"""Reproducible figures/tables from four recorded runs; no invented aligned endpoints."""
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.optional-plot-runtime'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

OUT=ROOT/'results'
OUT.mkdir(exist_ok=True)
SOURCE=ROOT
result=json.loads((ROOT/'attestation/kaggle-final-results.json').read_text())
colors={'Baseline':'#2465ad','hybrid':'#cf3034'}
limits={'Baseline':63000,'hybrid':62683}
curves={}; provenance=[]; threshold_rows=[]
for run in result['runs']:
    model='Baseline' if run['slot']<4 else 'hybrid'
    name=('baseline' if run['slot']<4 else 'hybrid_darfix')+f'_seed{run["seed"]}_b32'
    path=ROOT/'artifacts/runs'/name/'metrics.jsonl'
    all_metrics=[json.loads(s) for s in path.read_text().splitlines()]
    evals=[m for m in all_metrics if m['kind']=='eval' and m['step']<=limits[model]]
    start=next(m['time'] for m in all_metrics if m['kind']=='start')
    console=json.loads((ROOT/'artifacts/runs'/name/'console.json').read_text())
    elapsed={}
    for m in console:
        match=re.search(r'\[EVAL\] step=(\d+)',m.get('data',''))
        if match:elapsed[int(match[1])]=m['time']
    curves[model,run['seed']]=dict(evals=evals,start=start,notebook_elapsed=elapsed,run=run)
    provenance.append(dict(slot=run['slot'],model=model,seed=run['seed'],commit=run['commit'],
        parameters=run['parameters'],cutoff_step=evals[-1]['step'],
        source=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

def stats(values):
    return dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)),per_seed=list(map(float,values)))
common=sorted(set.intersection(*(set(e['step'] for e in c['evals']) for c in curves.values())))
evolution=[]
for step in common:
    by={model:stats([next(e['val_bpb_est'] for e in curves[model,s]['evals'] if e['step']==step) for s in (1,2)]) for model in colors}
    diff=np.array(by['hybrid']['per_seed'])-np.array(by['Baseline']['per_seed'])
    pooled=math.sqrt((by['Baseline']['std']**2+by['hybrid']['std']**2)/2)
    evolution.append(dict(step=step,tokens=step*65536,baseline=by['Baseline'],hybrid=by['hybrid'],
        delta=float(np.mean(diff)),delta_std=float(np.std(diff,ddof=1)),pooled_std=pooled,
        abs_delta_over_pooled_std=abs(float(np.mean(diff)))/pooled,
        per_seed_delta=diff.tolist(),alignment='exact same logged step'))
endpoints={model:dict(steps=[curves[model,s]['evals'][-1]['step'] for s in (1,2)],
    **stats([curves[model,s]['evals'][-1]['val_bpb_est'] for s in (1,2)])) for model in colors}
end_delta=endpoints['hybrid']['mean']-endpoints['Baseline']['mean']
end_pooled=math.sqrt(sum(endpoints[m]['std']**2 for m in colors)/2)

thresholds=[1.10,1.08,1.06,1.02]
threshold_summary=[]
for target in thresholds:
    groups={}
    for model in colors:
        found=[]
        for seed in (1,2):
            c=curves[model,seed]
            ix=next((i for i,e in enumerate(c['evals']) if e['val_bpb_est']<=target),None)
            assert ix is not None,(target,model,seed)
            e=c['evals'][ix]
            previous=c['evals'][ix-1] if ix else None
            record=dict(target_bpb=target,model=model,seed=seed,step=e['step'],tokens=e['step']*65536,
                observed_bpb=e['val_bpb_est'],previous_above_step=previous['step'] if previous else None,
                previous_above_bpb=previous['val_bpb_est'] if previous else None,
                minutes_since_start_metric=(e['time']-c['start'])/60,
                notebook_minutes=c['notebook_elapsed'].get(e['step'],float('nan'))/60)
            threshold_rows.append(record);found.append(record)
        groups[model]={key:stats([r[key] for r in found]) for key in ('step','tokens','minutes_since_start_metric','notebook_minutes')}
    threshold_summary.append(dict(target_bpb=target,baseline=groups['Baseline'],hybrid=groups['hybrid'],
        tokens_reduction_pct=100*(1-groups['hybrid']['tokens']['mean']/groups['Baseline']['tokens']['mean']),
        notebook_time_reduction_pct=100*(1-groups['hybrid']['notebook_minutes']['mean']/groups['Baseline']['notebook_minutes']['mean'])))

per_seed=[dict(seed=s,all_common_steps_hybrid_better=all(e['per_seed_delta'][s-1]<0 for e in evolution),
    early_delta=evolution[0]['per_seed_delta'][s-1],last_common_delta=evolution[-1]['per_seed_delta'][s-1],
    endpoint_delta=endpoints['hybrid']['per_seed'][s-1]-endpoints['Baseline']['per_seed'][s-1]) for s in (1,2)]
regression=np.polyfit(np.log([e['step'] for e in evolution]),[e['delta'] for e in evolution],1)

def write_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
main_rows=[dict(step=str(e['step']),tokens=str(e['tokens']),
    baseline_s1=e['baseline']['per_seed'][0],baseline_s2=e['baseline']['per_seed'][1],baseline_mean=e['baseline']['mean'],baseline_std=e['baseline']['std'],
    hybrid_s1=e['hybrid']['per_seed'][0],hybrid_s2=e['hybrid']['per_seed'][1],hybrid_mean=e['hybrid']['mean'],hybrid_std=e['hybrid']['std'],
    delta=e['delta'],pooled_std=e['pooled_std'],abs_delta_over_pooled_std=e['abs_delta_over_pooled_std'],
    alignment='same step') for e in evolution]
main_rows.append(dict(step='B:63000; H:62673/62683',tokens='B:4128768000; H:4107337728/4107993088',
    baseline_s1=endpoints['Baseline']['per_seed'][0],baseline_s2=endpoints['Baseline']['per_seed'][1],baseline_mean=endpoints['Baseline']['mean'],baseline_std=endpoints['Baseline']['std'],
    hybrid_s1=endpoints['hybrid']['per_seed'][0],hybrid_s2=endpoints['hybrid']['per_seed'][1],hybrid_mean=endpoints['hybrid']['mean'],hybrid_std=endpoints['hybrid']['std'],
    delta=end_delta,pooled_std=end_pooled,abs_delta_over_pooled_std=abs(end_delta)/end_pooled,
    alignment='near-step endpoints; NOT exactly token matched'))
write_csv('table1-main-results.csv',main_rows)
write_csv('table2-token-efficiency-per-seed.csv',threshold_rows)
write_csv('table2-token-efficiency.csv',[dict(target_bpb=t['target_bpb'],
    baseline_steps_mean=t['baseline']['step']['mean'],baseline_steps_std=t['baseline']['step']['std'],
    hybrid_steps_mean=t['hybrid']['step']['mean'],hybrid_steps_std=t['hybrid']['step']['std'],
    baseline_tokens=t['baseline']['tokens']['mean'],hybrid_tokens=t['hybrid']['tokens']['mean'],
    tokens_reduction_pct=t['tokens_reduction_pct'],baseline_notebook_minutes=t['baseline']['notebook_minutes']['mean'],
    hybrid_notebook_minutes=t['hybrid']['notebook_minutes']['mean'],time_reduction_pct=t['notebook_time_reduction_pct']) for t in threshold_summary])

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.titlesize':11,
    'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':300})
def style(ax,log=True):
    if log:
        ax.set_xscale('log');ax.set_xlim(1000,70000)
        ax.set_xticks([1500,3000,10000,30000,63000]);ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel('Optimizer steps')
    ax.grid(True,alpha=.2);ax.spines[['top','right']].set_visible(False)
def mainplot(ax):
    for model in colors:
        vals=[e['baseline' if model=='Baseline' else 'hybrid'] for e in evolution]
        x=[e['step'] for e in evolution];y=[v['mean'] for v in vals];err=[v['std'] for v in vals]
        ep=endpoints[model]
        x.append(float(np.mean(ep['steps'])));y.append(ep['mean']);err.append(ep['std'])
        ax.errorbar(x,y,yerr=err,color=colors[model],label=model,fmt='o-',markersize=3,
            linewidth=1.4,elinewidth=1,capsize=2,zorder=3)
    style(ax);ax.set_ylim(.98,1.55);ax.set_ylabel('Validation BPB (estimated)')
    ax.legend(loc='upper right',frameon=False)
    ax.annotate(f'$\\Delta = {end_delta:.3f}$\nnear-step endpoints',
        xy=(np.mean(endpoints['hybrid']['steps']),endpoints['hybrid']['mean']),
        xytext=(2200,1.06),arrowprops=dict(arrowstyle='->',color='#333333'),fontsize=10)
def deltaplot(ax):
    x=np.array([e['step'] for e in evolution]);y=np.array([e['delta'] for e in evolution])
    ax.errorbar(x,y,yerr=[e['delta_std'] for e in evolution],fmt='o-',color=colors['hybrid'],capsize=2,markersize=3,label='Paired delta mean ± sample SD')
    ax.axhline(0,color='#555555',lw=1,ls='--');style(ax)
    ax.set_ylim(-.065,.004);ax.set_ylabel('Δ BPB (hybrid − Baseline)')
    ax.annotate('advantage peaks early,\nnarrows toward convergence',xy=(15000,evolution[9]['delta']),xytext=(2000,-.009),
        arrowprops=dict(arrowstyle='->',lw=.8),fontsize=9)
    ax.text(.04,.08,f'Descriptive OLS slope: {regression[0]:+.4f} BPB / ln(step)',transform=ax.transAxes,fontsize=8)
def bars(ax):
    x=np.arange(len(thresholds));width=.34
    for model,key,offset in [('Baseline','baseline',-.17),('hybrid','hybrid',.17)]:
        ax.bar(x+offset,[t[key]['tokens']['mean']/1e9 for t in threshold_summary],width,
            yerr=[t[key]['tokens']['std']/1e9 for t in threshold_summary],capsize=3,label=model,color=colors[model])
    for i,t in enumerate(threshold_summary):
        ax.text(i,t['baseline']['tokens']['mean']/1e9+.13,f'−{t["tokens_reduction_pct"]:.1f}%',ha='center',fontsize=9)
    ax.set_xticks(x,[f'≤ {t:.2f}' for t in thresholds]);ax.set_xlabel('Validation BPB target (estimated)')
    ax.set_ylabel('Tokens to first logged target (billions)');style(ax,False)
    ax.set_ylim(0,max(t['baseline']['tokens']['mean']/1e9 for t in threshold_summary)*1.18)
    ax.legend(frameon=False)
def save(fig,name):
    for suffix in ('png','pdf','svg'):fig.savefig(OUT/f'{name}.{suffix}',bbox_inches='tight')
    plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4.5));mainplot(ax);fig.tight_layout();save(fig,'figure1-val-bpb')
fig,ax=plt.subplots(figsize=(7,4.5));deltaplot(ax);fig.tight_layout();save(fig,'figure2-delta')
fig,ax=plt.subplots(figsize=(7,4.5));bars(ax);fig.tight_layout();save(fig,'figure3-token-efficiency')
fig,ax=plt.subplots(figsize=(7,4.5))
x=np.arange(len(thresholds))
for model,key,offset in [('Baseline','baseline',-.17),('hybrid','hybrid',.17)]:
    ax.bar(x+offset,[t[key]['notebook_minutes']['mean'] for t in threshold_summary],.34,
        yerr=[t[key]['notebook_minutes']['std'] for t in threshold_summary],capsize=3,color=colors[model],label=model)
for i,t in enumerate(threshold_summary):
    ax.text(i,t['baseline']['notebook_minutes']['mean']+22,f'−{t["notebook_time_reduction_pct"]:.1f}%',ha='center')
ax.set_xticks(x,[f'≤ {t:.2f}' for t in thresholds]);ax.set_xlabel('Validation BPB target (estimated)')
ax.set_ylabel('Notebook time to first logged target (minutes)');style(ax,False)
ax.set_ylim(0,max(t['baseline']['notebook_minutes']['mean'] for t in threshold_summary)*1.2)
ax.legend(frameon=False);fig.tight_layout();save(fig,'figure3-wall-time-to-target')
fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
for seed,ax in zip((1,2),axs):
    for model in colors:
        ev=curves[model,seed]['evals'];ax.plot([e['step'] for e in ev],[e['val_bpb_est'] for e in ev],color=colors[model],marker='o',ms=2.5,lw=1.4,label=model)
    style(ax);ax.set_ylim(.98,1.55);ax.set_title(f'Seed {seed}');ax.legend(frameon=False)
axs[0].set_ylabel('Validation BPB (estimated)');fig.tight_layout();save(fig,'figure4-per-seed')
fig,axs=plt.subplots(2,1,figsize=(7.2,8.5))
mainplot(axs[0]);axs[0].set_title('A. Validation BPB: mean ± sample SD (2 seeds)',loc='left')
bars(axs[1]);axs[1].set_title('B. Tokens to target: first logged evaluation',loc='left')
fig.tight_layout(h_pad=2);save(fig,'arxiv_figure')

attestation=dict(artifact='training-attestation',schema_version=2,date='2026-10-16',
    source_repository='https://github.com/Akseleu-J/atomic-gdn2',
    kernel_commit='0dbc59cf74883315d7adbebe9229c520663e0d58',
    environment=dict(hardware='TPU v5e-8',backend='tpu',compute_dtype='bfloat16',seeds=[1,2]),
    optimization=dict(batch_size=32,sequence_length=2048,learning_rate='wall-time cosine with first-100-step warmup',
        peak_lr=3e-4,weight_decay=0.1,grad_clip=1.0,session_limit_seconds=32040,
        optimizer='Muon for eligible matrices; AdamW for remaining parameters'),
    analysis_cutoff_only=True,original_training_runs_unchanged=True,
    dataset=result['runs'][0]['data'],
    metric=dict(name='estimated validation BPB',formula='CE / ln(2) / 4.2',bytes_per_token=4.2,
        validation_target_tokens=4390912,std='sample SD across two seeds, ddof=1',n_seeds=2),
    runs=[dict(model=model,seeds=[1,2],cutoff_steps=endpoints[model]['steps'],statistics=endpoints[model],
        artifacts=[p for p in provenance if p['model']==model]) for model in colors],
    matched_token_result=evolution[-1],near_step_endpoint_result=dict(baseline=endpoints['Baseline'],hybrid=endpoints['hybrid'],
        delta_bpb=end_delta,exact_token_match=False),per_step_evolution=evolution,
    per_seed_consistency=per_seed,time_to_target=dict(method='first observed evaluation <= threshold, no interpolation',
        per_seed=threshold_rows,summary=threshold_summary),
    effect_ratio_definition='abs(mean paired delta) / sqrt((baseline_sample_variance + hybrid_sample_variance)/2); descriptive, not a significance test',
    limitations=['Only two seeds; SD is not a confidence interval.',
        'Baseline cutoff 63000; hybrid endpoints 62673/62683. Endpoint comparison is not exactly token matched.',
        'Exact shared-step curve ends at 61500. No hybrid value at 63000 exists.',
        '1.5B tokens has no recorded evaluation; 22500 and 24000 bracket it. No value invented.',
        'Time-based LR differs at the same step between architectures.',
        'hybrid includes MLA and different parameter count; DAR effect is not isolated.',
        'Trend OLS is descriptive; adjacent evaluations are correlated.',
        'Time-to-target limited by 1500-step evaluation grid. No predicted savings used.',
        'Checkpoint binaries not restored in this audit.'])
(OUT/'attestation').mkdir(exist_ok=True)
(OUT/'attestation/training.json').write_text(json.dumps(attestation,indent=2),encoding='utf-8')
note=['# Графики и таблицы: Baseline и hybrid','',
    'Использованы только измерения baseline до 63000, hybrid до финальных 62673/62683. '
    'Оба hybrid seeds сохранены: усечение seed 2 строго на 62673 исключило бы его финальное измерение.',
    'std = sample SD, ddof=1, n=2. Error bars не являются confidence interval.', '',
    '## Таблица 1. Все общие точки и отдельные конечные измерения','',
    '| Step | Baseline mean ± std | hybrid mean ± std | Δ H−B | abs(Δ)/pooled std |',
    '|---|---:|---:|---:|---:|']
for r in main_rows:
    note.append(f'| {r["step"]} | {r["baseline_mean"]:.6f} ± {r["baseline_std"]:.6f} | {r["hybrid_mean"]:.6f} ± {r["hybrid_std"]:.6f} | {r["delta"]:+.6f} | {r["abs_delta_over_pooled_std"]:.2f} |')
note+=['','Последняя строка — близкие конечные шаги, не matched-token comparison. '
    'Поэтому после отсечения baseline новая Δ округляется до −0.012, не прежних −0.011. '
    'Точное сравнение при 1.5B отсутствует: его окружают 22500 и 24000 шагов.',
    'abs(Δ)/pooled std — описательное отношение, не t-statistic и не доказательство значимости.', '',
    '## Таблица 2. Достижение порога','',
    '| BPB ≤ | Baseline steps | hybrid steps | Снижение tokens | Baseline minutes | hybrid minutes | Снижение времени |',
    '|---:|---:|---:|---:|---:|---:|---:|']
for t in threshold_summary:
    note.append(f'| {t["target_bpb"]:.2f} | {t["baseline"]["step"]["mean"]:.0f} | {t["hybrid"]["step"]["mean"]:.0f} | {t["tokens_reduction_pct"]:.2f}% | {t["baseline"]["notebook_minutes"]["mean"]:.2f} | {t["hybrid"]["notebook_minutes"]["mean"]:.2f} | {t["notebook_time_reduction_pct"]:.2f}% |')
note+=['','Время — Kaggle notebook-relative timestamp первого записанного [EVAL], включая setup/compile, '
    'validation и checkpoint overhead до точки, без очереди. В CSV также время после start metric '
    'и per-seed показатели с предыдущей оценкой выше порога. Первое пересечение между evaluations не известно.', '',
    '## Проверка per-seed','']
for s in per_seed:
    note.append(f'- Seed {s["seed"]}: hybrid лучше на всех {len(common)} общих шагах: {s["all_common_steps_hybrid_better"]}. '
        f'Δ в начале {s["early_delta"]:+.6f}, на 61500 {s["last_common_delta"]:+.6f}, в конечных точках {s["endpoint_delta"]:+.6f}.')
note+=['','Оба seeds имеют одинаковый знак на всей измеренной кривой. Это не устраняет ограничение двух seeds '
    'и не доказывает преимущество на других задачах/корпусах. Сужение разрыва видно, но причина и '
    'предельное поведение при продолжении обучения из этих данных не устанавливаются.', '',
    '## Файлы','',
    '- figure1–4: PNG 300 dpi, PDF и SVG; arxiv_figure — обновлённые панели A и B.',
    '- table1-main-results.csv — все shared-step mean/std и исходные per-seed bpb.',
    '- table2-token-efficiency.csv и per-seed CSV — точные first-observed значения.',
    '- attestation/training.json — новые runs, seeds, matched_token_result, per_step_evolution.',
    '- Исторические arxiv_figure и attestation сохранены в archive/before-research-figures. '
    'Старый arxiv_figure.png фактически был PDF; новый файл соответствует расширению PNG.',
    '- Текст статьи не менялся. Изменения подготовлены локально, не опубликованы на GitHub.']
(OUT/'tables-and-methods.md').write_text('\n'.join(note)+'\n',encoding='utf-8')
# LaTeX table replacements are standalone snippets, not changes to the article.
tex=['\\begin{tabular}{rrrrr}','\\hline','Step & Baseline (mean $\\pm$ SD) & hybrid (mean $\\pm$ SD) & $\\Delta$ & $|\\Delta|/s_p$ \\\\','\\hline']
for r in main_rows:
    step=r['step'] if r['alignment']=='same step' else 'Endpoints$^{*}$'
    tex.append(f'{step} & ${r["baseline_mean"]:.6f} \\pm {r["baseline_std"]:.6f}$ & ${r["hybrid_mean"]:.6f} \\pm {r["hybrid_std"]:.6f}$ & {r["delta"]:+.6f} & {r["abs_delta_over_pooled_std"]:.2f} \\\\')
tex+=['\\hline','\\end{tabular}','% * Baseline 63000; hybrid 62673/62683. Not exactly token matched. Sample SD, n=2.']
(OUT/'table1-main-results.tex').write_text('\n'.join(tex),encoding='utf-8')
tex2=['\\begin{tabular}{rrrrrr}','\\hline',
    'BPB target & Baseline steps & hybrid steps & Token savings & Baseline min & hybrid min \\\\',
    '\\hline']
for t in threshold_summary:
    tex2.append(f'$\\leq {t["target_bpb"]:.2f}$ & '
        f'${t["baseline"]["step"]["mean"]:.0f} \\pm {t["baseline"]["step"]["std"]:.0f}$ & '
        f'${t["hybrid"]["step"]["mean"]:.0f} \\pm {t["hybrid"]["step"]["std"]:.0f}$ & '
        f'{t["tokens_reduction_pct"]:.2f}\\% & {t["baseline"]["notebook_minutes"]["mean"]:.2f} & '
        f'{t["hybrid"]["notebook_minutes"]["mean"]:.2f} \\\\')
tex2+=['\\hline','\\end{tabular}',
    '% First observed evaluation at or below target; no interpolation. Sample SD, n=2.']
(OUT/'table2-token-efficiency.tex').write_text('\n'.join(tex2),encoding='utf-8')
shutil.copy2(OUT/'attestation/training.json',SOURCE/'attestation/training.json')
summary=dict(endpoints=endpoints,endpoint_delta=end_delta,last_exact_common_step=common[-1],
    last_exact_common_delta=evolution[-1]['delta'],per_seed_consistency=per_seed,
    threshold_summary=threshold_summary,slope_delta_vs_ln_steps=float(regression[0]),
    std_ddof=1,expected_main_rows=len(evolution)+1)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with zipfile.ZipFile(ROOT/'artifacts/figures-tables.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in OUT.rglob('*'):
        if f.is_file():z.write(f,f.relative_to(OUT))
print(json.dumps(summary,indent=2))
