"""Generate data tables and a vector PGFPlots source from the saved results."""
from pathlib import Path
from statistics import median
import csv
import json

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results_v2/journal_followup'
S = json.loads((RESULTS/'summary.json').read_text())
family_map = json.loads((RESULTS/'family_map.json').read_text())
best_rows = list(csv.DictReader((RESULTS/'validation_best.csv').open(encoding='utf-8')))
F = []
for family in sorted(set(family_map.values())):
    tasks = [task for task,group in family_map.items() if group==family]
    item = dict(family=family,tasks=len(tasks))
    for method in ('mlp','eml_kan','kan'):
        per_task = [[r for r in best_rows if r['dataset']==task and r['method']==method] for task in tasks]
        item[method] = dict(nmse=median(median(float(r['test_nmse']) for r in group) for group in per_task),
                            cse=median(median(float(r['cse_tokens']) for r in group) for group in per_task))
    F.append(item)
(RESULTS/'family_summary.json').write_text(json.dumps(F,indent=2)+'\n',encoding='utf-8')
rows = list(csv.DictReader((RESULTS/'all_candidates.csv').open(encoding='utf-8')))
ceil = list(csv.DictReader((RESULTS/'validation_ceilings.csv').open(encoding='utf-8')))
cases = list(csv.DictReader((RESULTS/'fault_cases.csv').open(encoding='utf-8')))
LABELS = {'mlp':'MLP','eml_kan':'EML-KAN','kan':'RBF-KAN'}
NAMES = dict(exp_decay='Exp. decay',log_response='Log response',power_law='Power law',
             arrhenius_like='Arrhenius-like',michaelis_menten='Saturation',
             multiplicative_power='Product power',mixed_exp_log='Mixed exp/log',
             damped_log_response='Damped log',exp_saturation='Exp. saturation',
             two_var_decay='Two-variable decay',control_quadratic='Quadratic',
             control_kinetic_energy='Kinetic energy',control_sine='Sine',control_sine_cosine='Sine/cosine')


def sci(value):
    if float(value) == 0:return '0'
    a,e=f'{float(value):.2e}'.split('e')
    return a+r'\times10^{'+str(int(e))+'}'


def table(caption,label,columns,header,data):
    return ('\\begin{table}[tbp]\n\\centering\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
            '\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{'+columns+'}\n\\toprule\n'+header+'\\\\\n\\midrule\n'+
            '\n'.join(' & '.join(r)+r' \\' for r in data)+'\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n')


tables={}
data=[]
for m in ('mlp','eml_kan','kan'):
    q=S['method_summary'][m]
    data.append([LABELS[m],'$'+sci(q['median_test_nmse'])+'$',f"{q['median_test_r2']:.6f}",
                 '$'+sci(q['median_ood_nmse'])+'$',f"{q['median_raw_tokens']:.1f}",
                 f"{q['median_cse_tokens']:.1f}",f"{q['median_inference_ms']:.3f}"])
tables['best']=table('Best-validation selections: medians over 42 task--seed models per family. Each row can attain a different predictive quality. Latency is CPU float64 forward time per 1000 examples, with five warmups and thirty repetitions after training.','tab:best','lrrrrrr','Method & Test NMSE & Test $R^2$ & Wider NMSE & Raw & CSE & ms',data)
data=[]
for q in S['validation_ceilings']:
    data.append(['$'+sci(q['threshold'])+'$',str(q['mlp_feasible']),str(q['eml_kan_feasible']),str(q['kan_feasible']),str(q['paired']),f"{q['eml_shorter']}/{q['paired']}",f"{q['median_ratio']:.3f}"])
tables['ceilings']=table('Shared validation-error ceilings. Feasible counts are out of 42 task--seed settings. Each method selects its smallest feasible expanded-CSE graph. The ratio is EML-KAN/MLP among jointly feasible settings; below one favors EML-KAN. Pair sets differ across ceilings.','tab:ceilings','rrrrrrr','Val. NMSE & MLP & EML & RBF & Pairs & EML shorter & Median ratio',data)
data=[]
for q in F:
    data.append([q['family'],str(q['tasks'])]+['$'+sci(q[m]['nmse'])+'$ / '+f"{q[m]['cse']:.0f}" for m in ('mlp','eml_kan','kan')])
tables['families']=table('Function-family diagnostics for best-validation models. Cells give test NMSE / expanded-CSE tokens. Values are medians of per-task medians over three seeds; small groups limit generalization.','tab:families','lrrrr','Family & Tasks & MLP & EML-KAN & RBF-KAN',data)
tables['fidelity']=table('Worst absolute checkpoint discrepancy over every candidate, all three export variants, and 2000 audited inputs per candidate. Values are measured float64 discrepancies, not assertions of exact real-arithmetic equality.','tab:fidelity','lrrr','Method & Candidates & Maximum discrepancy & Failures',[[LABELS[m],'252','$'+sci(S['method_summary'][m]['maximum_export_error'])+'$','0'] for m in ('mlp','eml_kan','kan')])
data=[]
for m in ('mlp','eml_kan'):
    for fault,label in [('omit_input_centering','Input centering'),('hidden_assignment_offset','Hidden assignment'),('omit_target_inverse_scaling','Target scaling')]:
        rr=[r for r in cases if r['method']==m and r['fault']==fault]
        lo=min(int(r['comparisons']) for r in rr);hi=max(int(r['comparisons']) for r in rr)
        data.append([LABELS[m],label,'3/3',str(lo) if lo==hi else f'{lo}--{hi}','$'+sci(max(float(r['repaired_error']) for r in rr))+'$'])
tables['faults']=table('Full-checkpoint faults on mixed exp/log, seeds 0--2. Each row represents three trained models. Comparisons count assignment-value vectors examined to the first mismatch, not analyst time. All cases are repaired and rechecked.','tab:faults','llrrr','Model & Controlled fault & Localized & Comparisons & Repaired max. error',data)
tables['task_pairs']=table('Near-error pairs by function. Ratios are EML-KAN/MLP expanded-CSE tokens. Ranges span eligible seeds, not confidence intervals; missing pairs are listed in the artifact.','tab:task-pairs','lrrr','Function & Eligible seeds & Median ratio & Seed range',[[NAMES[q['dataset']],str(q['n']),f"{q['median']:.3f}",f"{q['min']:.3f}--{q['max']:.3f}"] for q in S['strict_matched']['task_summaries']])
formulae=[('Exp. decay','1',r'1.7e^{-0.8x_1}+0.25'),('Log response','1',r'1.2\log(1.5x_1+0.7)-0.15'),('Power law','1',r'0.8(x_1+0.2)^{1.35}+0.1'),('Arrhenius-like','1',r'2e^{-1.15/(x_1+0.6)}'),('Saturation','1',r'1.8x_1/(0.45+x_1)'),('Product power','2',r'1.1(x_1+0.15)^{0.7}(x_2+0.25)^{-0.45}'),('Mixed exp/log','1',r'0.9e^{0.55x_1}+0.65\log(1.2x_1+0.8)'),('Damped log','1',r'1.4e^{-0.55x_1}\log(1.7x_1+0.9)'),('Exp. saturation','1',r'1.3(1-e^{-1.1x_1})+0.05'),('Two-var. decay','2',r'0.75x_1e^{-0.65x_2}+0.2'),('Quadratic control','1',r'x_1^2+x_1+1'),('Kinetic-energy control','2',r'0.5x_1x_2^2'),('Sine control','1',r'\sin(2x_1)'),('Sine/cosine control','2',r'\sin(1.5x_1)+0.3\cos(2x_2)')]
tables['formulas']=table('Complete generated functions. The last four are new controls. Executable NumPy definitions and exact generated arrays are included in the artifact.','tab:formulas','lrl','Function & Inputs & Generating expression',[[a,b,'$'+c+'$'] for a,b,c in formulae])
(RESULTS/'final_tables.json').write_text(json.dumps(tables,indent=2),encoding='utf-8')

plot=[r'\documentclass[tikz,border=3pt,10pt]{standalone}',r'\usepackage{pgfplots}',r'\usepgfplotslibrary{groupplots}',r'\pgfplotsset{compat=1.18}',r'\definecolor{mlpblue}{HTML}{0072B2}',r'\definecolor{emlorange}{HTML}{D55E00}',r'\definecolor{rbfgreen}{HTML}{009E73}',r'\begin{document}',r'\begin{tikzpicture}',r'\begin{groupplot}[group style={group size=1 by 2,vertical sep=2.2cm},width=12cm,height=6.6cm,tick label style={font=\small},axis line style={black!70},grid=major,grid style={black!12}]',r'\nextgroupplot[xmode=log,ymode=log,xtick={300,600,1000,2000,3000},xticklabels={300,600,1000,2000,3000},xlabel={Expanded-CSE tokens},ylabel={Validation NMSE},title={(a) Mixed exp/log, seed 0},legend style={font=\small,at={(0.98,0.98)},anchor=north east,fill=white,draw=none}]']
for m,color,marker in [('mlp','mlpblue','square*'),('eml_kan','emlorange','*'),('kan','rbfgreen','triangle*')]:
    rr=[r for r in rows if r['method']==m and r['dataset']=='mixed_exp_log' and r['seed']=='0']
    coords=' '.join(f"({r['cse_tokens']},{float(r['val_nmse']):.12g})" for r in rr)
    plot += [rf'\addplot[only marks,mark={marker},mark size=2.3pt,color={color}] coordinates {{{coords}}};',rf'\addlegendentry{{{LABELS[m]}}}']
    front=[r for r in rr if not any(float(o['cse_tokens'])<=float(r['cse_tokens']) and float(o['val_nmse'])<=float(r['val_nmse']) and (float(o['cse_tokens'])<float(r['cse_tokens']) or float(o['val_nmse'])<float(r['val_nmse'])) for o in rr)]
    front.sort(key=lambda r:float(r['cse_tokens']))
    coords=' '.join(f"({r['cse_tokens']},{float(r['val_nmse']):.12g})" for r in front)
    plot.append(rf'\addplot[no marks,thin,color={color},forget plot] coordinates {{{coords}}};')
plot += [r'\nextgroupplot[xmode=log,xmin=0.000075,xmax=0.014,ymin=0,ymax=1.9,xtick={0.0001,0.001,0.01},xlabel={Shared validation NMSE ceiling},ylabel={EML-KAN / MLP token ratio},title={(b) Minimum-size feasible exports}]',r'\addplot[black,dashed,domain=0.000075:0.014,samples=2] {1};']
for threshold in (0.0001,0.001,0.01):
    vals=sorted(float(r['eml_mlp_cse_ratio']) for r in ceil if float(r['validation_ceiling'])==threshold and r.get('eml_mlp_cse_ratio'))
    def quantile(q):
        pos=(len(vals)-1)*q;lo=int(pos);hi=min(lo+1,len(vals)-1)
        return vals[lo]+(vals[hi]-vals[lo])*(pos-lo)
    lo,hi=quantile(.25),quantile(.75)
    plot += [rf'\addplot[emlorange,thick,no marks] coordinates {{({threshold},{lo}) ({threshold},{hi})}};',rf'\addplot[emlorange,only marks,mark=*,mark size=2.8pt] coordinates {{({threshold},{median(vals)})}};',rf'\node[font=\small,anchor={'south east' if threshold==0.01 else 'south west'}] at (axis cs:{threshold},{min(1.65,hi+.05)}) {{$n={len(vals)}$}};']
plot += [r'\end{groupplot}',r'\end{tikzpicture}',r'\end{document}']
(RESULTS/'capacity_plot.tex').write_text('\n'.join(plot)+'\n',encoding='utf-8')
print('Data tables and vector-plot source generated.')
