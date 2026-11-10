from datetime import date
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.future_research_holdout import availability


def test_future_evaluation_requires_past_end_date_and_final_observed_session():
    days=pd.bdate_range('2026-11-10',periods=20)
    reservation={'sessions':[str(day.date()) for day in days],'end':str(days[-1].date()),'minimum_coverage':.95}
    data={'A':pd.DataFrame({'Timestamp':days[:-1]})}
    assert not availability(data,reservation,date(2026,11,9))['ready']
    assert not availability(data,reservation,date(2026,12,8))['ready'] # 95% alone cannot expose an unfinished last session
    data['A']=pd.DataFrame({'Timestamp':days})
    assert not availability(data,reservation,days[-1].date())['ready']
    assert availability(data,reservation,date(2026,12,8))['ready']
