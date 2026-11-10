"""Combine validated per-seed diagnostics and training curves, without a report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def combine(shards,study,output):
    if output.exists():raise ValueError('combined diagnostics exist')
    paths=sorted(path for path in shards.iterdir() if path.is_dir())
    names=('critic_and_initial_policy','policy_sensitivity','learning_curves','ppo_optimizer_metrics')
    tables={name:[] for name in names}
    for path in paths:
        status=json.loads((path/'completion.json').read_text())
        if not status['completed'] or status['test_evaluated']:raise ValueError('require complete validation diagnostics')
        for name in names:tables[name].append(pd.read_csv(path/(name+'.csv')))
    combined={name:pd.concat(parts,ignore_index=True) for name,parts in tables.items()}
    checks=combined['critic_and_initial_policy']
    expected=json.loads((study/'results.json').read_text())
    keys=set(zip(checks.fold,checks.condition,checks.seed))
    wanted={(row['fold'],row['condition'],row['seed']) for row in expected}
    if keys!=wanted or len(checks)!=len(expected) or not checks.initial_policy_exactly_reproduced.all():
        raise ValueError('missing, duplicate or unreproduced policy diagnostics')
    output.mkdir(parents=True)
    for name,table in combined.items():table.to_csv(output/(name+'.csv'),index=False)
    risks=[]
    for row in expected:
        history=json.loads((study/f"fold_{row['fold']:02d}"/row['condition']/f"seed_{row['seed']}"/'risk_training.json').read_text())
        bars=np.cumsum([point['bars'] for point in history])
        for point,total in zip(history,bars):
            risks.append({'fold':row['fold'],'condition':row['condition'],'seed':row['seed'],
                'training_bars':int(total),**point})
    risk=pd.DataFrame(risks);risk.to_csv(output/'risk_budget_updates.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    colors={'I':'tab:blue','J':'tab:orange','K':'tab:green'}
    for condition in sorted(checks.condition.unique()):
        color=colors.get(condition)
        for _,group in combined['learning_curves'].query('condition == @condition').groupby(['fold','seed']):
            axes[0,0].plot(group.training_bars,group.mean_return*100,color=color,alpha=.4)
        for _,group in risk.query('condition == @condition').groupby(['fold','seed']):
            axes[0,1].plot(group.training_bars,group.mean_risk_cost,color=color,alpha=.4)
            axes[1,0].plot(group.training_bars,group.multiplier_after,color=color,alpha=.4)
        optimizer=combined['ppo_optimizer_metrics']
        if 'train/explained_variance' in optimizer:
            for _,group in optimizer.query('condition == @condition').groupby(['fold','seed']):
                axes[1,1].plot(group['time/total_timesteps'],group['train/explained_variance'],color=color,alpha=.25)
        axes[0,0].plot([],[],color=color,label=condition)
    axes[0,0].set_ylabel('Validation mean return (%)');axes[0,0].legend()
    axes[0,1].set_ylabel('Mean drawdown excess cost');axes[0,1].axhline(float(risk.budget.iloc[0]),ls='--',color='black',label='budget')
    axes[0,1].legend();axes[1,0].set_ylabel('Risk multiplier');axes[1,1].set_ylabel('Training value explained variance')
    for axis in (axes[0,0],axes[0,1],axes[1,0]):axis.set_xlabel('Underlying training bars')
    axes[1,1].set_xlabel('PPO decisions');axes[1,1].axhline(0,color='black',ls='--')
    for axis in axes.ravel():axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'learning_and_risk.png',dpi=160);plt.close(fig)
    (output/'completion.json').write_text(json.dumps({'completed':True,'models_audited':len(checks),
        'initial_policies_exactly_reproduced':True,'test_evaluated':False,'report_generated':False},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shards',type=Path,required=True);parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    combine(args.shards,args.study,args.output)
