"""Validation-only summaries of the complete external-data candidate set."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from study import OUT, METHODS


def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def flat(row):
    r=dict(id=row['id'],**row['configuration'],parameters=row['parameters'],best_step=row['training']['best_step'])
    for split in ('train','val','test'):
        for metric in ('mse','nmse','r2'):r[split+'_'+metric]=row['metrics'][split][metric]
    for key in ('raw_tokens','expanded_tokens','cse_tokens','assignments','cse_assignments'):r[key]=row['graph'][key]
    r['maximum_normalized_export_error']=max(row['graph']['normalized_max_errors'].values())
    return r


def main():
    rows=[json.loads(path.read_text(encoding='utf-8')) for path in sorted((OUT/'trials').glob('*.json'))]
    assert len(rows)==1296,len(rows)
    failures=[r for r in rows if r['status']!='ok'];assert not failures,failures[:2]
    groups={}
    for row in rows:
        c=row['configuration'];groups.setdefault((c['case'],c['seed'],c['method']),[]).append(row)
    assert len(groups)==24*3*3 and all(len(group)==6 for group in groups.values())
    best={key:min(group,key=lambda r:(r['metrics']['val']['nmse'],r['id'])) for key,group in groups.items()}
    write_csv('all_candidates.csv',[flat(row) for row in rows])
    write_csv('validation_best.csv',[flat(row) for row in best.values()])
    pairs=[];missing=[];ceilings=[]
    for case,seed in sorted(set((key[0],key[1]) for key in groups)):
        collection=groups[(case,seed,'mlp')][0]['configuration']['collection']
        possible=[]
        for a in groups[(case,seed,'mlp')]:
            for b in groups[(case,seed,'eml_kan')]:
                va=a['metrics']['val']['nmse'];vb=b['metrics']['val']['nmse']
                if max(va,vb)<=.01 and max(va,vb)/max(min(va,vb),1e-300)<=1.25:
                    possible.append((max(va,vb),abs(math.log(vb/va)),a['id'],b['id'],a,b))
        if possible:
            *_,a,b=min(possible,key=lambda value:value[:4])
            pairs.append(dict(collection=collection,case=case,seed=seed,mlp_id=a['id'],eml_id=b['id'],
                              mlp_val_nmse=a['metrics']['val']['nmse'],eml_val_nmse=b['metrics']['val']['nmse'],
                              mlp_test_nmse=a['metrics']['test']['nmse'],eml_test_nmse=b['metrics']['test']['nmse'],
                              mlp_cse=a['graph']['cse_tokens'],eml_cse=b['graph']['cse_tokens'],
                              ratio=b['graph']['cse_tokens']/a['graph']['cse_tokens']))
        else:missing.append(dict(collection=collection,case=case,seed=seed,reason='No pair meets both validation NMSE <= .01 and error ratio <= 1.25'))
        for threshold in (.01,.001,.0001):
            record=dict(collection=collection,case=case,seed=seed,ceiling=threshold)
            for method in METHODS:
                valid=[r for r in groups[(case,seed,method)] if r['metrics']['val']['nmse']<=threshold]
                record[method+'_feasible']=bool(valid)
                if valid:
                    selected=min(valid,key=lambda r:(r['graph']['cse_tokens'],r['metrics']['val']['nmse'],r['id']))
                    record.update({method+'_id':selected['id'],method+'_cse':selected['graph']['cse_tokens'],method+'_test_nmse':selected['metrics']['test']['nmse']})
            if record['mlp_feasible'] and record['eml_kan_feasible']:record['ratio']=record['eml_kan_cse']/record['mlp_cse']
            ceilings.append(record)
    write_csv('matched_pairs.csv',pairs);write_csv('missing_pairs.csv',missing);write_csv('validation_ceilings.csv',ceilings)
    summaries=[];ceiling_summaries=[];pair_summaries=[]
    for collection in ('Feynman-PMLB','SRSD-hard'):
        for method in METHODS:
            selected=[r for r in best.values() if r['configuration']['collection']==collection and r['configuration']['method']==method]
            all_method=[r for r in rows if r['configuration']['collection']==collection and r['configuration']['method']==method]
            summaries.append(dict(collection=collection,method=method,n=len(selected),median_test_nmse=median(r['metrics']['test']['nmse'] for r in selected),
                                  median_test_r2=median(r['metrics']['test']['r2'] for r in selected),median_raw=median(r['graph']['raw_tokens'] for r in selected),
                                  median_cse=median(r['graph']['cse_tokens'] for r in selected),feasible_1e2=sum(r['metrics']['val']['nmse']<=.01 for r in selected),
                                  max_normalized_export_error=max(max(r['graph']['normalized_max_errors'].values()) for r in all_method)))
        pp=[p for p in pairs if p['collection']==collection]
        per_task={}
        for p in pp:per_task.setdefault(p['case'],[]).append(p['ratio'])
        pair_summaries.append(dict(collection=collection,pairs=len(pp),tasks=len(per_task),eml_shorter=sum(p['ratio']<1 for p in pp),
                                   pooled_median=median(p['ratio'] for p in pp) if pp else None,
                                   task_balanced_median=median(median(values) for values in per_task.values()) if pp else None))
        for threshold in (.01,.001,.0001):
            group=[r for r in ceilings if r['collection']==collection and r['ceiling']==threshold]
            eligible=[r for r in group if 'ratio' in r]
            ceiling_summaries.append(dict(collection=collection,ceiling=threshold,**{method+'_feasible':sum(r[method+'_feasible'] for r in group) for method in METHODS},
                                           pairs=len(eligible),eml_shorter=sum(r['ratio']<1 for r in eligible),median_ratio=median(r['ratio'] for r in eligible) if eligible else None))
    result=dict(candidates=len(rows),tasks=24,seeds=3,failed=len(failures),best_validation=summaries,near_error=pair_summaries,ceilings=ceiling_summaries,
                max_normalized_export_error=max(max(r['graph']['normalized_max_errors'].values()) for r in rows),
                numerical_failures=0,source_provenance_sha256=hashlib.sha256((OUT/'data_provenance.json').read_bytes()).hexdigest())
    save('summary.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
