"""Seal a future evaluation before its start; refuse partial or changed inputs."""
from __future__ import annotations

import argparse
import copy
from datetime import date,datetime,timedelta
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
for directory in (ROOT,ROOT/'agent/src',ROOT/'env/src'):
    sys.path.insert(0,str(directory))

import pandas as pd
from experiments.run_ablation import load_market,load_research_bundle,evaluate,write_json,code_snapshot,data_fingerprints,make_env
from models.normalization import FeatureNormalizer


def today():
    return datetime.now(ZoneInfo('Asia/Seoul')).date()


def availability(data,reservation,as_of):
    end=date.fromisoformat(reservation['end'])
    expected={date.fromisoformat(day) for day in reservation['sessions']}
    coverage={}
    for symbol,frame in data.items():
        observed=set(pd.to_datetime(frame.Timestamp).dt.date)
        present=observed&expected
        coverage[symbol]={'ratio':len(present)/len(expected),'missing_days':[str(day) for day in sorted(expected-observed)],
            'last_session_available':end in observed}
    ready=(as_of>end and all(row['ratio']>=reservation['minimum_coverage'] and row['last_session_available']
                           for row in coverage.values()))
    return {'ready':ready,'as_of':str(as_of),'earliest_evaluation_date':str(end+timedelta(days=1)),
            'coverage':coverage,'scores_computed':False}


def reserve(study,plan_path,output):
    if output.exists():raise ValueError('future reservation exists; do not replace it')
    plan=json.loads(plan_path.read_text());reservation=plan['future_holdout']
    if today()>=date.fromisoformat(reservation['start']):
        raise ValueError('cannot retrospectively reserve an already-started holdout')
    selection=json.loads((study/'frozen_evaluation/selection_freeze.json').read_text())
    manifest=json.loads((study/'study_manifest.json').read_text())
    if code_snapshot()['python_source_sha256']!=manifest['code']['python_source_sha256']:
        raise ValueError('code changed since the historical study')
    primary=selection['selected_condition'];fold=max(row['index'] for row in manifest['folds'])
    bundles=[]
    for condition in manifest['arguments']['conditions'].split(','):
        for seed in map(int,manifest['arguments']['seeds'].split(',')):
            directory=study/f'fold_{fold:02d}'/condition/f'seed_{seed}'
            proof=next(row for row in selection['artifacts'] if row['directory']==str(directory.relative_to(study)))
            if hashlib.sha256((directory/'research_model.zip').read_bytes()).hexdigest()!=proof['model_sha256']:
                raise ValueError('selected policy changed after pre-test freeze')
            bundles.append({'directory':str(directory.resolve()),'condition':condition,'seed':seed,
                'model_sha256':proof['model_sha256'],'manifest_sha256':proof['manifest_sha256']})
    output.mkdir(parents=True)
    payload={'reserved_on':str(today()),'window':reservation,'primary_condition':primary,'fold':fold,
        'historical_selection_status':selection['selection_status'],'bundles':bundles,
        'historical_data_end':plan['calendar'][-1],'historical_fingerprints':plan['source_fingerprints'],
        'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'code':manifest['code'],'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evaluated':False,'selection_locked':True}
    write_json(output/'reservation.json',payload)
    return payload


def run_evaluation(output):
    reservation=json.loads((output/'reservation.json').read_text())
    if (output/'evaluation').exists():raise ValueError('future evaluation already exists; do not retune or overwrite')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=reservation['evaluator_sha256']:
        raise ValueError('future evaluator changed after reservation')
    if code_snapshot()['python_source_sha256']!=reservation['code']['python_source_sha256']:
        raise ValueError('research implementation changed after reservation')
    data,_,_=load_market(list(reservation['historical_fingerprints']))
    cutoff=date.fromisoformat(reservation['historical_data_end'])
    historical={symbol:frame.loc[pd.to_datetime(frame.Timestamp).dt.date<=cutoff].reset_index(drop=True)
                for symbol,frame in data.items()}
    if data_fingerprints(historical)!=reservation['historical_fingerprints']:
        raise ValueError('historical features changed; investigate before scoring future data')
    status=availability(data,reservation['window'],today())
    write_json(output/'availability.json',status)
    if not status['ready']:
        print(json.dumps(status));return False
    # No policy loading or prediction occurs until the complete future window
    # and its final observed session have passed the above gate.
    results=[];directory=output/'evaluation';directory.mkdir()
    expected={date.fromisoformat(day) for day in reservation['window']['sessions']}
    for bundle in reservation['bundles']:
        path=Path(bundle['directory'])
        for filename,key in (('research_model.zip','model_sha256'),('research_manifest.json','manifest_sha256')):
            if hashlib.sha256((path/filename).read_bytes()).hexdigest()!=bundle[key]:
                raise ValueError('sealed future artifact changed')
        for symbol,source in data.items():
            frame=source.loc[pd.to_datetime(source.Timestamp).dt.date.isin(expected)].reset_index(drop=True)
            model,env,_=load_research_bundle(path,frame);metrics,trace=evaluate(model,env)
            trace.to_csv(directory/f"{bundle['condition']}_seed_{bundle['seed']}_{symbol}.csv",index=False)
            results.append({'condition':bundle['condition'],'seed':bundle['seed'],'symbol':symbol,'metrics':metrics})
    write_json(directory/'results.json',results)
    first=Path(reservation['bundles'][0]['directory'])
    configuration=copy.deepcopy(json.loads((first/'research_manifest.json').read_text())['config'])
    normalizer=FeatureNormalizer.load(first/'normalization.json')
    configuration['environment'].update(episode_days=1,unit_fraction=.995)
    controls={}
    for symbol,source in data.items():
        frame=source.loc[pd.to_datetime(source.Timestamp).dt.date.isin(expected)].reset_index(drop=True)
        controls[symbol]={name:evaluate(None,make_env(frame,configuration,'R',normalizer),baseline=name)[0]
            for name in ('cash','intraday_buy_and_hold','intraday_mean_reversion')}
        controls[symbol]['buy_and_hold']=evaluate(None,make_env(frame,configuration,'A',normalizer),baseline='buy_and_hold')[0]
    write_json(directory/'fixed_controls.json',controls)
    write_json(directory/'completion.json',{'completed':True,'runs':len(results),'training_performed':False,
        'selection_changed':False,'report_generated':False})
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reserve',action='store_true');parser.add_argument('--study',type=Path)
    parser.add_argument('--plan',type=Path)
    args=parser.parse_args()
    if args.reserve:
        if args.study is None or args.plan is None:parser.error('--reserve requires --study and --plan')
        reserve(args.study,args.plan,args.output)
    else:run_evaluation(args.output)
