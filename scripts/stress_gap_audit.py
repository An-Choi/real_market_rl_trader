"""Frozen-policy stress audit of observed sparse days, without filling or fitting."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for directory in (ROOT,ROOT/'agent/src',ROOT/'env/src'):
    sys.path.insert(0,str(directory))

import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace

from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer
from data.resample import add_minute_trading_value,split_closing_auction,resample_5min
from data.micro_features import add_micro_features
from data.context_features import add_context_features
from data.cross_day_features import add_cross_day_features
from experiments.run_ablation import load_research_bundle,make_env,evaluate,write_json
from scripts.recover_research_data import missing_runs


def observed_features(raw):
    """Existing causal formulas, no full-day rejection, no resampled fake prices.

    Unlike the usual forward-asof execution join, require a print at the exact
    next-minute execution label. A halt cannot silently queue an order into a
    future reopening price.
    """
    clean=add_minute_trading_value(raw)
    regular,auction=split_closing_auction(clean)
    bars=resample_5min(regular)
    bars=add_micro_features(bars);bars=add_context_features(bars,regular)
    bars=add_cross_day_features(bars,auction)
    prices=regular.set_index('Timestamp').Open
    bars['ExecPrice']=pd.to_datetime(bars.Timestamp).map(prices)
    columns=['Timestamp','Close','ExecPrice','Adv20',*FeatureEngineer.FEATURE_COLUMNS]
    return bars[columns].dropna(subset=list(FeatureEngineer.FEATURE_COLUMNS)).reset_index(drop=True)


def prepare(root,recovery):
    if root.exists():raise ValueError('stress feature revision exists')
    root.mkdir(parents=True)
    defects=json.loads((recovery/'recovery.json').read_text())
    loader=DataLoader(ROOT/'data/raw',ROOT/'data/processed')
    records=[]
    for symbol in ('005930','000660','034220','066570','009150'):
        raw=loader.load_raw_parquet_all(symbol,'1m')
        features=observed_features(raw)
        features.to_parquet(root/f'{symbol}.parquet',index=False)
        for item in defects:
            if item['symbol']!=symbol:continue
            target=date.fromisoformat(item['date'])
            frame=raw.loc[pd.to_datetime(raw.Timestamp).dt.date==target]
            records.append({'symbol':symbol,'date':str(target),
                'observed_missing_runs':[{'start':str(run[0]),'end':str(run[-1]),'minutes':len(run)} for run in missing_runs(frame)],
                'available_feature_bars':int((pd.to_datetime(features.Timestamp).dt.date==target).sum())})
    matches=[]
    for day in sorted({row['date'] for row in records}):
        group=[row for row in records if row['date']==day]
        signatures={json.dumps(row['observed_missing_runs'],sort_keys=True) for row in group}
        intervals=[]
        for row in group:
            intervals.append({minute for run in row['observed_missing_runs']
                for minute in pd.date_range(run['start'],run['end'],freq='min')})
        shared=set.intersection(*intervals) if intervals else set()
        matches.append({'date':day,'symbols':len(group),'same_missing_intervals':len(signatures)==1,
            'shared_missing_minutes':len(shared)})
    write_json(root/'manifest.json',{'prepared':True,'records':records,'synchronous_dates':matches,
        'feature_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob('*.parquet'))},
        'classification':'synchronous gaps can be marketwide trading halts; provider outage is not established',
        'event_correspondence_examples':[
            {'date':'2026-03-04','source':'https://www.donga.com/news/Economy/article/all/20260304/133461199/1'},
            {'date':'2026-07-07','source':'https://www.donga.com/news/Economy/article/all/20260707/134250744/1'},
            {'date':'2026-06-08','source':'https://file.alphasquare.co.kr/media/pdfs/market-report/820260609KB%EC%A6%9D%EA%B6%8C.pdf'}],
        'fully_adjudicated_halt_registry':False,'quality_gate_relaxed':False,
        'synthetic_price_rows':0,'execution_join':'exact observed next-minute label; no forward fill across halts',
        'feature_semantics':'existing formulas on observed bars; gaps reset market context; this is an exploratory distribution-shift audit',
        'sampling_bias_warning':'qualified historical studies excluded these sparse dates and cannot establish crash-regime robustness'})


class ConstantTarget:
    def __init__(self,target):self.target=target;self.policy=self
    def obs_to_tensor(self,observation):return torch.as_tensor(observation).reshape(1,-1),False
    def get_distribution(self,observation,action_masks):
        action=self.target if action_masks[self.target] else 0
        probs=torch.zeros((1,6));probs[0,action]=1.
        return SimpleNamespace(distribution=SimpleNamespace(probs=probs))


def audit(args):
    if args.output.exists():raise ValueError('stress audit exists')
    torch.set_num_threads(1);args.output.mkdir(parents=True)
    meta=json.loads((args.features/'manifest.json').read_text())
    for filename,checksum in meta['feature_sha256'].items():
        if hashlib.sha256((args.features/filename).read_bytes()).hexdigest()!=checksum:
            raise ValueError('stress feature revision changed')
    selection=json.loads((args.study/'frozen_evaluation/selection_freeze.json').read_text())
    study=json.loads((args.study/'study_manifest.json').read_text());fold=max(row['index'] for row in study['folds'])
    bundle=args.study/f'fold_{fold:02d}'/args.condition/f'seed_{args.seed}'
    proof=next(row for row in selection['artifacts'] if row['directory']==str(bundle.relative_to(args.study)))
    if (hashlib.sha256((bundle/'research_model.zip').read_bytes()).hexdigest()!=proof['model_sha256']
            or hashlib.sha256((bundle/'research_manifest.json').read_bytes()).hexdigest()!=proof['manifest_sha256']):
        raise ValueError('policy changed since pre-test freeze')
    write_json(args.output/'audit_freeze.json',{'condition':args.condition,'seed':args.seed,
        'model_sha256':proof['model_sha256'],'refitting':False,'selection_allowed':False,
        'manifest_sha256':proof['manifest_sha256'],
        'stress_features_sha256':hashlib.sha256((args.features/'manifest.json').read_bytes()).hexdigest()})
    rows=[]
    for symbol in ('005930','000660','034220','066570','009150'):
        data=pd.read_parquet(args.features/f'{symbol}.parquet')
        days=sorted(set(pd.to_datetime(data.Timestamp).dt.date))
        for target in sorted(date.fromisoformat(row['date']) for row in meta['records'] if row['symbol']==symbol):
            available=[day for day in days if day<=target]
            chosen=set(available[-20:])
            frame=data.loc[pd.to_datetime(data.Timestamp).dt.date.isin(chosen)].reset_index(drop=True)
            if target not in chosen or len(chosen)!=20:raise ValueError('incomplete stress history')
            model,env,manifest=load_research_bundle(bundle,frame)
            metrics,trace=evaluate(model,env)
            benchmark,_=evaluate(ConstantTarget(5),make_env(frame,manifest['config'],'I',env.normalizer))
            stress_trace=trace.loc[pd.to_datetime(trace.valuation_timestamp).dt.date==target]
            previous=trace.loc[pd.to_datetime(trace.valuation_timestamp).dt.date<target].portfolio_value
            day_initial=float(previous.iloc[-1]) if len(previous) else env.initial_cash
            row={'condition':args.condition,'seed':args.seed,'symbol':symbol,'stress_date':str(target),
                'window_start':str(min(chosen)),'window_sessions':len(chosen),
                'stress_day_return':float(stress_trace.portfolio_value.iloc[-1]/day_initial-1),
                'window_net_return':metrics['net_return'],'window_max_drawdown':metrics['bar_max_drawdown'],
                'cash_control_return':0.,'daily_full_target_control_return':benchmark['net_return'],
                'daily_full_target_control_max_drawdown':benchmark['bar_max_drawdown'],
                'mean_drawdown_excess':metrics['discounted_drawdown_excess'],
                'stress_day_trades':int(stress_trace.executed_trades.sum()),
                'stress_day_mean_exposure':float(stress_trace.exposure.mean()),
                'unavailable_fill_bars':metrics['unavailable_fill_bars']}
            rows.append(row)
            trace.to_csv(args.output/f'{symbol}_{target}_trace.csv',index=False)
        pd.DataFrame(rows).to_csv(args.output/'metrics.csv',index=False)
    write_json(args.output/'completion.json',{'completed':True,'policy_windows':len(rows),'refitting':False,
        'model_selection_changed':False,'quality_gate_passed':False,'synthetic_price_rows':0,
        'status':'exploratory stress audit, not a certified halt-aware dataset','report_generated':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--recovery',type=Path);parser.add_argument('--study',type=Path)
    parser.add_argument('--condition',default='I');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.prepare:
        if args.recovery is None:parser.error('--prepare requires --recovery')
        prepare(args.features,args.recovery)
    else:
        if args.study is None or args.output is None:parser.error('--study and --output required')
        audit(args)
