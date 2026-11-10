"""Loopback-only research replay protocol, independent of brokerage/order code."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import uuid

ROOT=Path(__file__).resolve().parents[2]
for directory in (ROOT,ROOT/'agent/src',ROOT/'env/src'):
    sys.path.insert(0,str(directory))

import numpy as np
import torch
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,ConfigDict,Field

from experiments.run_ablation import load_market,load_research_bundle
from experiments.finalize_research import fold_from_dict
from models.walk_forward import slice_walk_forward_fold
from env.research_env import TARGET_WEIGHTS
from env.controlled_exposure_env import ControlledExposureEnvironment


class ExpectedState(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    session_id:str
    model_sha256:str
    cash:float=Field(ge=0,strict=True)
    shares:int=Field(ge=0,strict=True)
    cost_basis:float=Field(ge=0,strict=True)
    decisions:int=Field(ge=0,strict=True)


def create_app(bundle,symbol='005930',split='test',frame=None):
    bundle=Path(bundle)
    manifest=json.loads((bundle/'research_manifest.json').read_text())
    if frame is None:
        data,_,_=load_market([symbol])
        frame=slice_walk_forward_fold(data[symbol],fold_from_dict(manifest['fold']),segment=split)
    model,env,_=load_research_bundle(bundle,frame)
    if not isinstance(env,ControlledExposureEnvironment):
        raise ValueError('this replay protocol requires a controlled target-exposure artifact')
    app=FastAPI(title='Local research dry-run protocol')
    lock=threading.RLock()
    session={'id':'','done':False,'episode':0,'decisions':0}
    initial_cash=env.initial_cash
    checksum=manifest['sha256']['research_model.zip']
    all_dates=env.available_dates

    def state():
        return {'session_id':session['id'],'model_sha256':checksum,'cash':float(env.cash),
            'shares':int(env.shares_held),'cost_basis':float(env.cost_basis),
            'done':session['done'],'decisions':session['decisions']}

    def reset():
        session.update(id=str(uuid.uuid4()),done=False,episode=0,decisions=0)
        env.initial_cash=initial_cash
        env.reset(seed=0,options={'start_date':all_dates[0],'episode_days':env.episode_days})
        return state()

    reset()

    @app.get('/metadata')
    def metadata():
        return {'api_contract':'research-dry-run-v1','research_only':True,'broker_orders_enabled':False,
            'model_sha256':checksum,'symbol':symbol,'split':split,'contract':env.contract(),
            'target_weights':list(TARGET_WEIGHTS),'initial_cash':initial_cash,
            'first_timestamp':str(frame.Timestamp.iloc[0]),'last_timestamp':str(frame.Timestamp.iloc[-1])}

    @app.get('/state')
    def current_state():
        with lock:
            return state()

    @app.post('/reset')
    def reset_session():
        with lock:
            return reset()

    @app.post('/step')
    def step(expected:ExpectedState):
        with lock:
            before=state()
            if expected.session_id!=session['id'] or expected.model_sha256!=checksum:
                raise HTTPException(409,'session or artifact changed')
            if expected.decisions!=session['decisions']:
                raise HTTPException(409,'stale or repeated decision request')
            for key in ('cash','shares','cost_basis'):
                if not math.isclose(getattr(expected,key),before[key],rel_tol=1e-12,abs_tol=1e-5):
                    raise HTTPException(409,'Java/Python portfolio mismatch: '+key)
            if session['done']:
                raise HTTPException(409,'replay already finished')
            observation=env._get_observation()
            masks=env.action_masks()
            with torch.no_grad():
                tensor,_=model.policy.obs_to_tensor(observation)
                probabilities=model.policy.get_distribution(tensor,action_masks=masks).distribution.probs.cpu().numpy().ravel()
            action=int(np.argmax(probabilities))
            mark=float(env.get_current_market_row()[env.price_col])
            first_step=env.current_step
            observed_fills={}
            original_fill=env._fill

            def trace_fill(requested):
                offset=env.current_step-first_step
                available=np.isfinite(float(env.get_current_market_row()[env.exec_price_col]))
                quantity,price,cost=original_fill(requested)
                observed_fills.setdefault(offset,[]).append({'reason':'target','requested_quantity':int(requested),
                    'quantity':int(quantity),'price':float(price),'cost':float(cost),'fill_available':bool(available)})
                return quantity,price,cost

            env._fill=trace_fill
            try:
                _,_,terminated,truncated,info=env.step(action)
            finally:
                env._fill=original_fill
            events=[]
            running_shares=before['shares']
            for offset,bar in enumerate(info['bar_records']):
                for event in observed_fills.get(offset,[]):
                    running_shares+=event['quantity']
                    events.append({**event,'bar_offset':offset,'timestamp':bar['valuation_timestamp'].isoformat()})
                closing_cost=float(bar['terminal_settlement_cost'])
                # Settlement is a valuation-time event, separate from a target
                # order at the earlier observed decision price.
                if running_shares and bar['shares']==0 and (closing_cost>0 or offset==len(info['bar_records'])-1):
                    events.append({'reason':'settlement','requested_quantity':-running_shares,
                        'quantity':-running_shares,'price':float(bar['valuation_price']),
                        'cost':closing_cost,'fill_available':True,'bar_offset':offset,
                        'timestamp':bar['valuation_timestamp'].isoformat()})
                    running_shares=0
                if running_shares!=int(bar['shares']):
                    raise RuntimeError('execution event stream does not reproduce Python shares')
            session['decisions']+=1
            if terminated or truncated:
                session['episode']+=env.episode_days
                if session['episode']>=len(all_dates):
                    session['done']=True
                else:
                    capital=env.cash
                    env.initial_cash=capital
                    env.reset(seed=0,options={'start_date':all_dates[session['episode']],
                                              'episode_days':env.episode_days})
            return {'decision_index':session['decisions'],'action':action,'target_weight':TARGET_WEIGHTS[action],
                'action_mask':masks.tolist(),'observation':observation.tolist(),
                'probabilities':probabilities.tolist(),'mark_price':mark,
                'rebalance_tolerance':env.rebalance_tolerance,
                'requested_quantity':next((event['requested_quantity'] for event in events if event['reason']=='target'),0),
                'events':events,'elapsed_bars':info['elapsed_bars'],'state':state()}
    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--symbol',default='005930')
    parser.add_argument('--split',choices=['validation','test'],default='test');parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    torch.set_num_threads(1)
    import uvicorn
    uvicorn.run(create_app(args.bundle,args.symbol,args.split),host='127.0.0.1',port=args.port)
