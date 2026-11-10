from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
import pytest

from experiments.predictive_diagnostics import causal_inputs,forward_targets
from experiments.run_ablation import make_env,synthetic_data
from models.normalization import FeatureNormalizer
from models.research_protocol import reference_calendar
from data.feature_engineer import FeatureEngineer
from utils.config_loader import load_config
from env.research_env import market_window


def test_predictive_inputs_match_rl_causal_windows_and_do_not_include_execution():
    data=synthetic_data(days=2)[0]['SYNTHETIC']
    normalizer=FeatureNormalizer.fit(data.iloc[:32],list(FeatureEngineer.FEATURE_COLUMNS))
    before=causal_inputs(data,normalizer,12)
    for index in (0,1,15,31,32,33,63):
        np.testing.assert_array_equal(before[index],market_window(data,list(FeatureEngineer.FEATURE_COLUMNS),index,12,normalizer))
    altered=data.copy();altered['ExecPrice']*=10
    altered.loc[20:,list(FeatureEngineer.FEATURE_COLUMNS)]=10000.
    after=causal_inputs(altered,normalizer,12)
    np.testing.assert_array_equal(before[:20],after[:20])


def test_forward_labels_use_execution_and_cannot_cross_session_or_gap():
    data=synthetic_data(days=2)[0]['SYNTHETIC']
    labels=forward_targets(data,3)
    assert np.isnan(labels[29:32]).all()
    assert labels[0]==pytest.approx(np.log(data.Close.iloc[3]/data.ExecPrice.iloc[0]))
    changed=data.copy();changed.loc[2:,'Timestamp']+=pd.Timedelta(minutes=5)
    assert np.isnan(forward_targets(changed,3)[0])


def test_independent_calendar_handles_holiday_end_bounds_and_sourced_2026_closures():
    from datetime import date
    sessions,provenance=reference_calendar(date(2026,6,1),date(2026,12,31))
    assert date(2026,6,3) not in sessions
    assert date(2026,7,17) not in sessions
    assert date(2026,12,31) not in sessions
    assert provenance['independent_of_price_availability']


def test_daily_settlement_preserves_multiday_risk_peak_and_blocks_overnight_entry():
    data=synthetic_data(days=3)[0]['SYNTHETIC']
    data['Close']=np.linspace(100,80,len(data));data['ExecPrice']=data.Close
    config=load_config(ROOT/'env/configs/config.yaml')
    config['environment']['episode_days']=3
    config['agent']['ppo']['gamma']=1.
    config['research']={'drawdown_threshold':0.,'risk_budget':.001}
    stats=FeatureNormalizer.fit(data,list(FeatureEngineer.FEATURE_COLUMNS))
    env=make_env(data,config,'K',stats)
    env.reset(seed=0)
    initial_peak=env._liquidation_peak
    while env.current_step<31:
        env.step(5)
    assert env.current_step==31
    assert env.shares_held==0
    assert env._liquidation_peak==initial_peak
    assert env._liquidation_value<initial_peak
    np.testing.assert_array_equal(env.action_masks(),[True,False,False,False,False,False])
    _,_,done,_,info=env.step(0)
    assert not done and env.current_step==32
    assert env.shares_held==0
    assert info['risk_cost']>0  # cash overnight does not erase cumulative loss
