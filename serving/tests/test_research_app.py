import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'serving/src'))

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from research_app import create_app
from experiments.run_ablation import synthetic_data,make_env,save_research_bundle
from models.normalization import FeatureNormalizer
from models.constrained_ppo import ResearchPPO,DurationAwareRolloutBuffer
from data.feature_engineer import FeatureEngineer
from utils.config_loader import load_config


@pytest.fixture
def research_client(tmp_path):
    torch.set_num_threads(1)
    data=synthetic_data(days=3)[0]['SYNTHETIC']
    config=load_config(ROOT/'env/configs/config.yaml')
    config['environment']['episode_days']=3
    norm=FeatureNormalizer.fit(data,list(FeatureEngineer.FEATURE_COLUMNS))
    env=make_env(data,config,'K',norm)
    model=ResearchPPO('MlpPolicy',env,n_steps=8,batch_size=8,gamma=config['agent']['ppo']['gamma'],
        rollout_buffer_class=DurationAwareRolloutBuffer,policy_kwargs={'net_arch':[8,8]},seed=1,verbose=0)
    with torch.no_grad():
        for parameter in model.policy.parameters():parameter.zero_()
        model.policy.action_net.bias[3]=1.
    save_research_bundle(model,norm,env,config,tmp_path,{'condition':'K'})
    return TestClient(create_app(tmp_path,'SYNTHETIC',frame=data))


def body(state):
    return {key:state[key] for key in ('session_id','model_sha256','cash','shares','cost_basis','decisions')}


def test_research_protocol_events_reproduce_cash_shares_basis_and_settlement(research_client):
    meta=research_client.get('/metadata').json()
    assert meta['contract']['n_actions']==6
    assert meta['broker_orders_enabled'] is False
    state=research_client.post('/reset').json()
    decisions=0
    while not state['done']:
        response=research_client.post('/step',json=body(state))
        assert response.status_code==200,response.text
        value=response.json();cash=state['cash'];shares=state['shares'];basis=state['cost_basis']
        for event in value['events']:
            previous=shares;quantity=event['quantity'];price=event['price']
            cash-=quantity*price+event['cost'];shares+=quantity
            if quantity>0:basis+=quantity*price
            elif previous:basis*=shares/previous
            if shares==0:basis=0.
        state=value['state'];decisions+=1
        assert state['cash']==pytest.approx(cash)
        assert state['shares']==shares
        assert state['cost_basis']==pytest.approx(basis)
        assert len(value['observation'])==meta['contract']['observation_shape'][0]
    assert decisions>3
    assert state['shares']==state['cost_basis']==0


def test_research_protocol_rejects_stale_duplicate_and_wrong_state(research_client):
    state=research_client.post('/reset').json()
    payload=body(state)
    assert research_client.post('/step',json=payload).status_code==200
    assert research_client.post('/step',json=payload).status_code==409
    current=research_client.get('/state').json();bad=body(current);bad['cash']+=1
    assert research_client.post('/step',json=bad).status_code==409
    assert research_client.get('/state').json()==current
    bad=body(current);bad['units_held']=1
    assert research_client.post('/step',json=bad).status_code==422
    assert research_client.post('/predict',json={}).status_code==404
