from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import pandas as pd
from scripts.recover_research_data import merge_observed_rows


def test_recovery_adds_only_observed_requested_day_rows_and_preserves_existing_prices():
    timestamps=pd.date_range('2026-07-07 09:00',periods=3,freq='min',tz='Asia/Seoul')
    full=pd.DataFrame({'Timestamp':timestamps,'Open':100.,'High':100.,'Low':100.,'Close':100.,'Volume':10,'TradingValue':1000})
    old=full.iloc[[0,2]].copy()
    fetched=full.copy();fetched.loc[0,'Close']=200
    extra=full.iloc[[0]].copy();extra['Timestamp']+=pd.Timedelta(days=1)
    fetched=pd.concat([fetched,extra],ignore_index=True)
    merged=merge_observed_rows(old,fetched,timestamps[0].date())
    assert len(merged)==3
    assert merged.Close.tolist()==[100.,100.,100.]
