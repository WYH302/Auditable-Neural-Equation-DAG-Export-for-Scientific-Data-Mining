"""Validation-only selection, paired summaries, and full-checkpoint fault cases."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import time

import numpy as np
import torch

from graph import evaluate
from study import OUT, HERE, METHODS, create_model, make_data, normalize


def save(name, obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def csv_file(name, rows):
    if not rows:
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def flat(row):
    c=row["configuration"];g=row["graph"]
    r=dict(id=row["id"],**c,parameters=row["parameters"],
           train_seconds=row["training"]["training_seconds"],
           best_step=row["training"]["best_step"],steps=row["training"]["steps_completed"])
    for split in ("val","test","ood"):
        for key in ("mse","nmse","r2"):r[split+"_"+key]=row["metrics"][split][key]
    for key in ("raw_tokens","core_tokens","expanded_tokens","cse_tokens","assignments","cse_assignments"):
        r[key]=g[key]
    r["max_export_error"]=max(g["max_abs_errors"].values())
    return r


def load_checkpoint(row):
    c=row["configuration"]
    model=create_model(c["method"],len(row["scales"]["x_mean"]),c["width"],c["seed"])
    checkpoint=torch.load(OUT/"checkpoints"/(row["id"]+".pt"),map_location="cpu",weights_only=True)
    model.load_state_dict(checkpoint["state_dict"]);model.eval()
    return model


def representative_cases(best):
    torch.set_num_threads(1)
    cases=[]
    for method in ("mlp","eml_kan"):
        for seed in (0,1,2):
            row=best[("mixed_exp_log",seed,method)]
            model=load_checkpoint(row);s=row["scales"]
            x=np.linspace(0.05,3.0,512).reshape(-1,1)
            xn=(x-np.asarray(s["x_mean"]))/np.asarray(s["x_std"])
            with torch.no_grad():reference=model(torch.from_numpy(xn)).numpy().reshape(-1)*s["y_std"]+s["y_mean"]
            raw=json.loads((OUT/"graphs"/(row["id"]+".json")).read_text())["raw"]
            good=evaluate(raw,x,trace=True)
            tolerance=1e-9*max(1,float(np.max(np.abs(reference))))
            correct_error=float(np.max(np.abs(good["y"]-reference)))
            assert correct_error<=tolerance
            hidden=next(i for i,line in enumerate(raw) if not line.split("=",1)[0].strip().startswith("x"))
            for fault,target in (("omit_input_centering",0),("hidden_assignment_offset",hidden),("omit_target_inverse_scaling",len(raw)-1)):
                broken=raw.copy();name,rhs=raw[target].split("=",1);name=name.strip()
                if fault=="omit_input_centering":broken[target]=f"{name} = x1 / ({s['x_std'][0]:.17g})"
                elif fault=="hidden_assignment_offset":broken[target]=name+" = ("+rhs.strip()+") + 0.1"
                else:broken[target]="y = y_norm"
                bad=evaluate(broken,x,trace=True)
                first=None;comparisons=0
                t0=time.perf_counter()
                for node,v in good.items():
                    comparisons+=1
                    node_tolerance=1e-9*max(1,float(np.max(np.abs(v))))
                    if float(np.max(np.abs(bad[node]-v)))>node_tolerance:
                        first=node;break
                localization_us=(time.perf_counter()-t0)*1e6
                repaired=broken.copy();repaired[target]=raw[target]
                repair_error=float(np.max(np.abs(evaluate(repaired,x)-reference)))
                output_error=float(np.max(np.abs(bad['y']-reference)))
                result=dict(id=row['id'],method=method,seed=seed,fault=fault,
                            injected_node=name,first_mismatch=first,localized_correctly=first==name,
                            output_discrepancy_detected=output_error>tolerance,
                            assignments=len(raw),comparisons=comparisons,
                            localization_comparison_microseconds=localization_us,
                            correct_error=correct_error,corrupted_error=output_error,
                            repaired_error=repair_error,tolerance=tolerance,evaluation_rows=512)
                assert first==name and repair_error<=tolerance
                cases.append(result)
                folder=OUT/'fault_cases';folder.mkdir(exist_ok=True)
                (folder/f'{method}_s{seed}_{fault}.json').write_text(json.dumps(dict(result=result,correct=raw,corrupted=broken,repaired=repaired),indent=2),encoding='utf-8')
    csv_file('fault_cases.csv',cases)
    save('fault_case_summary.json',dict(cases=len(cases),detected=sum(x['output_discrepancy_detected'] for x in cases),localized=sum(x['localized_correctly'] for x in cases),max_correct_error=max(x['correct_error'] for x in cases),max_repaired_error=max(x['repaired_error'] for x in cases),rows=cases))
    return cases


def main():
    torch.set_num_threads(1)
    complete=json.loads((OUT/'completion.json').read_text())
    assert complete['completed']==756 and complete['failed']==0
    rows=[json.loads(f.read_text(encoding='utf-8')) for f in sorted((OUT/'trials').glob('*.json'))]
    assert len(rows)==756 and all(r['status']=='ok' for r in rows)
    csv_file('all_candidates.csv',[flat(r) for r in rows])
    groups={}
    for row in rows:
        c=row['configuration'];groups.setdefault((c['dataset'],c['seed'],c['method']),[]).append(row)
    assert len(groups)==14*3*3 and all(len(v)==6 for v in groups.values())
    best={key:min(v,key=lambda r:(r['metrics']['val']['nmse'],r['id'])) for key,v in groups.items()}
    csv_file('validation_best.csv',[flat(r) for r in best.values()])
    pairs=[];unmatched=[];ceilings=[]
    task_seeds=sorted({(key[0],key[1]) for key in groups})
    for dataset,seed in task_seeds:
        eligible=[]
        for a in groups[(dataset,seed,'mlp')]:
            for b in groups[(dataset,seed,'eml_kan')]:
                va,vb=a['metrics']['val']['nmse'],b['metrics']['val']['nmse']
                if max(va,vb)<=0.01 and max(va,vb)/max(min(va,vb),1e-15)<=1.25:
                    eligible.append((max(va,vb),abs(math.log(vb/va)),a['id'],b['id'],a,b))
        if eligible:
            _,_,_,_,a,b=min(eligible,key=lambda x:x[:4])
            pair=dict(dataset=dataset,seed=seed,mlp_id=a['id'],eml_id=b['id'],
                      mlp_val_nmse=a['metrics']['val']['nmse'],eml_val_nmse=b['metrics']['val']['nmse'],
                      mlp_test_nmse=a['metrics']['test']['nmse'],eml_test_nmse=b['metrics']['test']['nmse'],
                      mlp_cse=a['graph']['cse_tokens'],eml_cse=b['graph']['cse_tokens'],
                      cse_ratio=b['graph']['cse_tokens']/a['graph']['cse_tokens'],
                      raw_ratio=b['graph']['raw_tokens']/a['graph']['raw_tokens'])
            pairs.append(pair)
        else:unmatched.append(dict(dataset=dataset,seed=seed,reason='No pair below NMSE 0.01 and within factor 1.25'))
        for threshold in (0.01,0.001,0.0001):
            record=dict(dataset=dataset,seed=seed,validation_ceiling=threshold)
            for m in METHODS:
                candidates=[r for r in groups[(dataset,seed,m)] if r['metrics']['val']['nmse']<=threshold]
                record[m+'_feasible']=bool(candidates)
                if candidates:
                    selected=min(candidates,key=lambda r:(r['graph']['cse_tokens'],r['metrics']['val']['nmse'],r['id']))
                    record.update({m+'_id':selected['id'],m+'_cse':selected['graph']['cse_tokens'],m+'_test_nmse':selected['metrics']['test']['nmse']})
            if record['mlp_feasible'] and record['eml_kan_feasible']:
                record['eml_mlp_cse_ratio']=record['eml_kan_cse']/record['mlp_cse']
            ceilings.append(record)
    csv_file('matched_pairs.csv',pairs);csv_file('unmatched_task_seeds.csv',unmatched);csv_file('validation_ceilings.csv',ceilings)
    task_ratios={}
    for pair in pairs:task_ratios.setdefault(pair['dataset'],[]).append(pair['cse_ratio'])
    paired=dict(task_seed_pairs=len(pairs),tasks=len(task_ratios),eml_shorter=sum(r['cse_ratio']<1 for r in pairs),
                median_cse_ratio=median(r['cse_ratio'] for r in pairs) if pairs else None,
                median_task_median_cse_ratio=median(median(v) for v in task_ratios.values()) if pairs else None,
                task_summaries=[dict(dataset=k,n=len(v),median=median(v),min=min(v),max=max(v)) for k,v in sorted(task_ratios.items())],unmatched=unmatched)
    ceiling_summary=[]
    for threshold in (0.01,0.001,0.0001):
        rr=[r for r in ceilings if r['validation_ceiling']==threshold]
        pp=[r for r in rr if 'eml_mlp_cse_ratio' in r]
        ceiling_summary.append(dict(threshold=threshold,**{m+'_feasible':sum(r[m+'_feasible'] for r in rr) for m in METHODS},
                                    paired=len(pp),eml_shorter=sum(r['eml_mlp_cse_ratio']<1 for r in pp),
                                    median_ratio=median(r['eml_mlp_cse_ratio'] for r in pp) if pp else None))
    profiles=[];method_summary={}
    for method in METHODS:
        selected=[row for key,row in best.items() if key[2]==method]
        for row in selected:
            model=load_checkpoint(row);c=row['configuration'];data=make_data(c['dataset'],c['seed']);norm,_=normalize(data);x=norm['test'][0]
            with torch.no_grad():
                for _ in range(5):model(x)
                t=time.perf_counter()
                for _ in range(30):model(x)
                elapsed=(time.perf_counter()-t)/30*1000
            profiles.append(dict(id=row['id'],dataset=c['dataset'],seed=c['seed'],method=method,milliseconds_per_1000=elapsed))
        method_summary[method]=dict(selected_runs=len(selected),median_test_nmse=median(r['metrics']['test']['nmse'] for r in selected),
                                    median_test_r2=median(r['metrics']['test']['r2'] for r in selected),
                                    median_ood_nmse=median(r['metrics']['ood']['nmse'] for r in selected),
                                    median_raw_tokens=median(r['graph']['raw_tokens'] for r in selected),
                                    median_cse_tokens=median(r['graph']['cse_tokens'] for r in selected),
                                    maximum_export_error=max(max(r['graph']['max_abs_errors'].values()) for r in rows if r['configuration']['method']==method),
                                    median_inference_ms=median(r['milliseconds_per_1000'] for r in profiles if r['method']==method))
    csv_file('inference_profiles.csv',profiles)
    cases=representative_cases(best)
    summary=dict(trials=len(rows),tasks=len(task_seeds)//3,seeds=3,failures=0,method_summary=method_summary,
                 strict_matched=paired,validation_ceilings=ceiling_summary,
                 fault_cases=dict(total=len(cases),detected=sum(x['output_discrepancy_detected'] for x in cases),localized=sum(x['localized_correctly'] for x in cases),max_repaired_error=max(x['repaired_error'] for x in cases)))
    save('summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='strict_matched'},indent=2),flush=True)
    print('STRICT_MATCHED',json.dumps({k:v for k,v in paired.items() if k not in ('unmatched','task_summaries')},indent=2),flush=True)


if __name__=='__main__':main()
