from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd

from scripts.stress_gap_audit import observed_features


def test_observed_stress_features_do_not_fabricate_bars_or_forward_fill_execution_across_halt():
    rng=np.random.default_rng(4);frames=[]
    days=pd.bdate_range('2026-01-05',periods=23,tz='Asia/Seoul')
    for day in days:
        times=pd.date_range(day+pd.Timedelta(hours=9),periods=380,freq='min')
        prices=100*np.exp(np.cumsum(rng.normal(0,.001,len(times))))
        frame=pd.DataFrame({'Timestamp':times,'Open':prices,'High':prices*1.001,'Low':prices*.999,
            'Close':prices,'Volume':100,'TradingValue':np.cumsum(prices*100)})
        auction=frame.iloc[[-1]].copy();auction['Timestamp']=day+pd.Timedelta(hours=15,minutes=30)
        frames.extend([frame,auction])
    raw=pd.concat(frames,ignore_index=True)
    start=days[-1]+pd.Timedelta(hours=11,minutes=20)
    end=start+pd.Timedelta(minutes=29)
    raw=raw.loc[~raw.Timestamp.between(start,end)].reset_index(drop=True)
    result=observed_features(raw)
    at_start=result.loc[result.Timestamp==start]
    assert len(at_start)==1
    assert at_start.ExecPrice.isna().all()
    assert not result.Timestamp.between(start+pd.Timedelta(minutes=5),end+pd.Timedelta(minutes=1)).any()
    changed=raw.copy();changed.loc[changed.Timestamp>end,'Close']*=2
    later=observed_features(changed)
    pd.testing.assert_frame_equal(result.loc[result.Timestamp<=start].reset_index(drop=True),
        later.loc[later.Timestamp<=start].reset_index(drop=True))
