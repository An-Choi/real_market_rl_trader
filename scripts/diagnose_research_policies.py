"""Audit policy reliance and critic calibration on validation states only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
for directory in (ROOT,ROOT/'agent/src',ROOT/'env/src'):
    sys.path.insert(0,str(directory))

import numpy as np
import pandas as pd
import torch

from experiments.run_ablation import (load_market,load_research_bundle,make_env,evaluate,write_json,EXTENSIONS)
from experiments.finalize_research import fold_from_dict
from models.constrained_ppo import ResearchPPO,DurationAwareRolloutBuffer,initialize_cash_prior
from models.walk_forward import slice_walk_forward_fold


def probabilities(model,observations,masks):
    result=[]
    for start in range(0,len(observations),512):
        with torch.no_grad():
            tensor,_=model.policy.obs_to_tensor(observations[start:start+512])
            result.append(model.policy.get_distribution(tensor,action_masks=masks[start:start+512]).distribution.probs.cpu().numpy())
    return np.concatenate(result)


def visited_states(model,env):
    observations,masks,values,returns=[],[],[],[]
    initial=env.initial_cash;capital=initial
    try:
        for start in env.available_dates[::env.episode_days]:
            env.initial_cash=capital
            observation,_=env.reset(seed=0,options={'start_date':start,'episode_days':env.episode_days})
            rewards,durations=[],[]
            while True:
                mask=env.action_masks()
                with torch.no_grad():
                    tensor,_=model.policy.obs_to_tensor(observation)
                    distribution=model.policy.get_distribution(tensor,action_masks=mask).distribution.probs.cpu().numpy().ravel()
                    value=float(model.policy.predict_values(tensor).cpu().numpy().ravel()[0])
                observations.append(observation.copy());masks.append(mask);values.append(value)
                observation,reward,terminated,truncated,info=env.step(int(np.argmax(distribution)))
                rewards.append(reward);durations.append(info['elapsed_bars'])
                if terminated or truncated:break
            discounted=0.;episode_returns=[]
            for reward,duration in zip(reversed(rewards),reversed(durations)):
                discounted=reward+env.bar_gamma**duration*discounted
                episode_returns.append(discounted)
            returns.extend(reversed(episode_returns));capital=env.cash
    finally:env.initial_cash=initial
    return np.asarray(observations),np.asarray(masks),np.asarray(values),np.asarray(returns)


def sensitivity(model,observations,masks,contract,seed=42):
    features=contract['feature_columns'];window=contract['window_length']
    market_count=window*(len(features)+1)
    # Padding and forced boundary actions otherwise make reliance misleading.
    full=(observations[:,window*len(features):market_count].sum(axis=1)==window)&(masks.sum(axis=1)>1)
    selected=np.flatnonzero(full)
    if len(selected)>1500:selected=selected[np.linspace(0,len(selected)-1,1500,dtype=int)]
    if len(selected)<2:return []
    obs=observations[selected];mask=masks[selected];baseline=probabilities(model,obs,mask)
    groups={'market_all':list(range(market_count)),'account_all':list(range(market_count,obs.shape[1]))}
    groups.update({name:[lag*len(features)+index for lag in range(window)] for index,name in enumerate(features)})
    groups.update({name:[market_count+index] for index,name in enumerate(contract['account_fields'])})
    rng=np.random.default_rng(seed);rows=[]
    for name,indices in groups.items():
        identifiable=bool(np.any(np.std(obs[:,indices],axis=0)>1e-8))
        shifts,flips=[],[]
        for _ in range(3):
            changed=obs.copy();changed[:,indices]=obs[rng.permutation(len(obs))][:,indices]
            current=probabilities(model,changed,mask)
            shifts.append(float(np.mean(.5*np.abs(current-baseline).sum(axis=1))))
            flips.append(float(np.mean(current.argmax(axis=1)!=baseline.argmax(axis=1))))
        rows.append({'field_group':name,'identifiable_on_visited_states':identifiable,
            'mean_probability_shift':float(np.mean(shifts)),'action_flip_rate':float(np.mean(flips)),
            'sampled_decisions':len(obs)})
    return rows


def run(args):
    if args.output.exists():raise ValueError('output exists')
    args.output.mkdir(parents=True);torch.set_num_threads(1)
    study=args.study;manifest=json.loads((study/'study_manifest.json').read_text())
    results=json.loads((study/'results.json').read_text())
    seeds=None if args.seeds is None else set(map(int,args.seeds.split(',')))
    selected=[r for r in results if r['condition'] in args.conditions.split(',') and (seeds is None or r['seed'] in seeds)]
    if not selected:raise ValueError('no matching models')
    data,_,_=load_market(manifest['arguments']['symbols'].split(','))
    diagnostics,sensitivities,curves=[],[],[]
    for result in selected:
        condition,seed,fold_index=result['condition'],result['seed'],result['fold']
        directory=study/f'fold_{fold_index:02d}'/condition/f'seed_{seed}'
        bundle=json.loads((directory/'research_manifest.json').read_text());fold=fold_from_dict(bundle['fold'])
        all_obs,all_masks,all_values,all_returns=[],[],[],[]
        initial_metrics=[]
        initial_config=copy.deepcopy(bundle['config'])
        mode=EXTENSIONS[condition]['risk_mode']
        initial_config.setdefault('research',{})['risk_multiplier']=.1 if mode!='off' else 0.
        initial_model=None
        for symbol,source in data.items():
            frame=slice_walk_forward_fold(source,fold,segment='validation')
            model,env,_=load_research_bundle(directory,frame)
            obs,mask,values,returns=visited_states(model,env)
            all_obs.append(obs);all_masks.append(mask);all_values.append(values);all_returns.append(returns)
            if initial_model is None:
                ppo=copy.deepcopy(initial_config['agent']['ppo'])
                ppo.update(verbose=0,n_steps=8,batch_size=8,policy_kwargs={'net_arch':[result['hidden_width']]*2},
                    rollout_buffer_class=DurationAwareRolloutBuffer)
                initial_env=make_env(frame,initial_config,condition,env.normalizer)
                initial_model=ResearchPPO('MlpPolicy',initial_env,seed=seed,device='cpu',**ppo)
                initialize_cash_prior(initial_model,initial_config.get('research',{}).get('cash_logit_bias',0.))
            initial_metrics.append(evaluate(initial_model,make_env(frame,initial_config,condition,env.normalizer))[0])
        obs=np.concatenate(all_obs);masks=np.concatenate(all_masks);values=np.concatenate(all_values);returns=np.concatenate(all_returns)
        variance=float(np.var(returns))
        quality={'fold':fold_index,'condition':condition,'seed':seed,'visited_decisions':len(obs),
            'critic_mc_mse':float(np.mean((values-returns)**2)),
            'critic_mc_explained_variance':1-float(np.var(values-returns))/variance if variance>1e-12 else None,
            'value_mean':float(values.mean()),'deterministic_mc_return_mean':float(returns.mean()),
            'initial_validation_return':float(np.mean([row['net_return'] for row in initial_metrics])),
            'selected_validation_return':float(np.mean([row['net_return'] for row in result['splits']['validation'].values()])),
            'selected_training_step':result['best_validation_step'],'selected_risk_multiplier':result['risk_multiplier']}
        history=json.loads((directory/'validation.json').read_text())
        expected_initial=float(np.mean([row['net_return'] for row in history[0]['per_symbol'].values()]))
        quality['initial_policy_exactly_reproduced']=bool(np.isclose(quality['initial_validation_return'],expected_initial,rtol=0,atol=1e-12))
        if not quality['initial_policy_exactly_reproduced']:raise RuntimeError('initial policy reconstruction mismatch')
        diagnostics.append(quality)
        for row in sensitivity(model,obs,masks,bundle['contract']):
            sensitivities.append({'fold':fold_index,'condition':condition,'seed':seed,**row})
        for checkpoint in history:
            curves.append({'fold':fold_index,'condition':condition,'seed':seed,'timesteps':checkpoint['timesteps'],
                'training_bars':checkpoint.get('training_bars',checkpoint['timesteps']),
                'validation_score':checkpoint['score'],
                'mean_return':float(np.mean([row['net_return'] for row in checkpoint['per_symbol'].values()])),
                'mean_turnover':float(np.mean([row['turnover'] for row in checkpoint['per_symbol'].values()]))})
        pd.DataFrame(diagnostics).to_csv(args.output/'critic_and_initial_policy.csv',index=False)
        pd.DataFrame(sensitivities).to_csv(args.output/'policy_sensitivity.csv',index=False)
        pd.DataFrame(curves).to_csv(args.output/'learning_curves.csv',index=False)
        print(json.dumps({key:quality[key] for key in ('fold','condition','seed','initial_policy_exactly_reproduced')}),flush=True)
    write_json(args.output/'completion.json',{'completed':True,'models_audited':len(diagnostics),'test_evaluated':False,
        'sensitivity':'group permutation on fully observed validation states; reliance, not causal predictive value',
        'critic':'deterministic validation Monte Carlo audit; different from stochastic training explained variance',
        'initial_policy':'exact network width, seed and initial risk multiplier reconstructed and checked against checkpoint zero',
        'report_generated':False})
    export_optimizer_metrics(study,selected,args.output)


def export_optimizer_metrics(study,selected,output):
    tables=[]
    for result in selected:
        path=study/f"fold_{result['fold']:02d}"/result['condition']/f"seed_{result['seed']}"/'progress.csv'
        table=pd.read_csv(path)
        table.insert(0,'seed',result['seed']);table.insert(0,'condition',result['condition']);table.insert(0,'fold',result['fold'])
        tables.append(table)
    pd.concat(tables,ignore_index=True).to_csv(output/'ppo_optimizer_metrics.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--conditions',default='F,H')
    parser.add_argument('--seeds');parser.add_argument('--optimizer-only',action='store_true')
    args=parser.parse_args()
    if args.optimizer_only:
        selected=[row for row in json.loads((args.study/'results.json').read_text()) if row['condition'] in args.conditions.split(',')]
        export_optimizer_metrics(args.study,selected,args.output)
    else:run(args)
