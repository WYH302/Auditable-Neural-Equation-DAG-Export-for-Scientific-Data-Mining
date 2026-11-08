"""Resumable external-data study with identified official data and saved weights."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'journal_followup'))
from models import MLP, EMLKAN, RBFKAN
from graph import measured, evaluate

OUT=ROOT/'results_v2/journal_external'
METHODS=('mlp','eml_kan','kan')
WIDTHS={'mlp':(4,8,16),'eml_kan':(2,4,8),'kan':(2,4,8)}
PROTOCOL=dict(version=1,methods=METHODS,widths=WIDTHS,learning_rates=(.001,.003),seeds=(0,1,2),
              maximum_updates=600,batch_size=256,sampling='Seeded draws with replacement from the full 8000-row training split',
              validation_every=20,patience_updates=120,weight_decay=1e-4,gradient_clip=1.,
              penalties=dict(gate=1e-4,coefficient=1e-5,affine_constants=1e-6),dtype='float64',device='cpu',
              workers=4,torch_threads_per_worker=1,precision='17-digit graph constants',
              normalized_export_tolerance='max|checkpoint-export|/training_y_std <= 1e-9*max(1,max|checkpoint|/training_y_std)',
              selection='Validation NMSE only; test never selects configurations. Near-error pairs require both validation NMSE <= .01 and a maximum/minimum ratio <= 1.25, then minimize maximum validation NMSE and log-error gap; ties use IDs.',
              ceilings=(.01,.001,.0001),ceiling_selection='Smallest expanded-CSE graph satisfying the shared validation-error ceiling; ties use validation NMSE and ID.',
              scope='New external-data rerun, not recovery of historical unidentified checkpoint summaries.')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def fingerprint():
    h=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode())
    for path in (HERE/'study.py',HERE/'selection.json',HERE.parent/'journal_followup/models.py',HERE.parent/'journal_followup/graph.py'):
        h.update(path.read_bytes())
    return h.hexdigest()


def load_data(case,seed):
    suffix=f'__s{seed}' if case['collection']=='Feynman-PMLB' else ''
    with np.load(OUT/'data'/(case['key']+suffix+'.npz')) as arrays:
        data={split:(arrays[split+'_x'],arrays[split+'_y']) for split in ('train','val','test')}
    return data


def normalize(data):
    x,y=data['train']
    xm=x.mean(axis=0);xs=x.std(axis=0);ym=float(y.mean());ys=float(y.std())
    assert np.isfinite(xs).all() and math.isfinite(ys) and ys>0
    xs=np.where(xs>0,xs,1.)
    # Preserve physical scales: a floor such as 1e-12 would corrupt small-unit targets.
    normalized={s:(torch.from_numpy((x-xm)/xs),torch.from_numpy((y-ym)/ys)) for s,(x,y) in data.items()}
    assert all(torch.isfinite(a).all() and torch.isfinite(b).all() for a,b in normalized.values())
    return normalized,dict(x_mean=xm.tolist(),x_std=xs.tolist(),y_mean=ym,y_std=ys)


def create_model(method,dimension,width,seed):
    torch.manual_seed(seed)
    if method=='mlp':return MLP(dimension,width=width,depth=2).double()
    if method=='eml_kan':return EMLKAN(dimension,width=width,edge_depth=2,tau=2.).double()
    return RBFKAN(dimension,width=width,grid_size=16).double()


def identifier(case,seed,method,width,lr):
    return f"{case['key']}__s{seed}__{method}__w{width}__lr{lr:g}"


def run_trial(item,pilot=False):
    torch.set_num_threads(1)
    case,seed,method,width,lr=item
    name=identifier(*item)
    path=OUT/'trials'/(name+'.json')
    signature=fingerprint()
    if path.exists() and not pilot:
        cached=json.loads(path.read_text(encoding='utf-8'))
        if cached.get('signature')==signature and cached['status']=='ok':return dict(id=name,status='ok',cached=True)
    configuration=dict(case=case['key'],collection=case['collection'],dataset_id=case['id'],dimension=case['dimension'],seed=seed,method=method,width=width,learning_rate=lr)
    base=dict(id=name,configuration=configuration,signature=signature)
    start=time.perf_counter()
    try:
        data=load_data(case,seed);normalized,scales=normalize(data)
        model=create_model(method,case['dimension'],width,seed)
        checkpoint_path=OUT/'checkpoints'/(name+'.pt')
        if checkpoint_path.exists() and not pilot:
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
            assert checkpoint['signature']==signature
            model.load_state_dict(checkpoint['state_dict']);fit=checkpoint['training']
        else:
            optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
            rng=torch.Generator().manual_seed(10000+seed)
            x,y=normalized['train'];vx,vy=normalized['val']
            best=float('inf');state=None;best_step=0;maximum=40 if pilot else 600
            for step in range(1,maximum+1):
                model.train();optimizer.zero_grad(set_to_none=True)
                indices=torch.randint(len(x),(256,),generator=rng)
                loss=torch.mean((model(x[indices])-y[indices])**2)
                if hasattr(model,'regularization_loss'):
                    gate,coefficient,constant=model.regularization_loss()
                    loss=loss+1e-4*gate+1e-5*coefficient+1e-6*constant
                assert torch.isfinite(loss),'nonfinite loss'
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'nonfinite gradient'
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if step%20==0 or step==maximum:
                    model.eval()
                    with torch.no_grad():score=float(torch.mean((model(vx)-vy)**2))
                    if score<best:
                        best=score;best_step=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    if step-best_step>=120:break
                if time.perf_counter()-start>120:raise TimeoutError('Training trial exceeded 120 seconds')
            assert state is not None
            model.load_state_dict(state)
            fit=dict(best_step=best_step,steps=step,seconds=time.perf_counter()-start,nonfinite_steps=0)
            if not pilot:
                checkpoint_path.parent.mkdir(exist_ok=True)
                temporary=checkpoint_path.with_suffix('.partial')
                torch.save(dict(state_dict=model.state_dict(),signature=signature,configuration=configuration,scales=scales,training=fit),temporary)
                temporary.replace(checkpoint_path)
        model.eval();predictions={};metrics={}
        with torch.no_grad():
            for split,(x,truth) in normalized.items():
                pn=model(x).numpy().reshape(-1);yn=truth.numpy().reshape(-1)
                normalized_mse=float(np.mean((pn-yn)**2));variance=float(np.var(yn))
                assert variance>0 and math.isfinite(variance)
                nmse=normalized_mse/variance
                predictions[split]=pn*scales['y_std']+scales['y_mean']
                mse=normalized_mse*scales['y_std']**2
                metrics[split]=dict(mse=mse,nmse=nmse,r2=1-nmse)
        check_x=np.vstack((data['val'][0],data['test'][0]))
        reference=np.concatenate((predictions['val'],predictions['test']))
        with np.errstate(over='ignore',under='ignore'):
            graph,variants=measured(model,method,scales,check_x,reference)
            errors={kind:float(np.max(np.abs(evaluate(lines,check_x)-reference)))/scales['y_std'] for kind,lines in variants.items()}
        tolerance=1e-9*max(1.,float(np.max(np.abs(reference)))/scales['y_std'])
        assert all(math.isfinite(value) and value<=tolerance for value in errors.values()),(errors,tolerance)
        graph.update(normalized_max_errors=errors,normalized_tolerance=tolerance)
        result=dict(**base,status='ok',training=fit,metrics=metrics,scales=scales,graph=graph,parameters=sum(p.numel() for p in model.parameters()),audited_rows=2000)
        if not pilot:
            save(OUT/'graphs'/(name+'.json'),variants)
            save(path,result)
        return dict(id=name,status='ok',seconds=round(time.perf_counter()-start,3),validation_nmse=metrics['val']['nmse'],test_nmse=metrics['test']['nmse'],cse_tokens=graph['cse_tokens'],normalized_export_error=max(errors.values()))
    except Exception as exc:
        result=dict(**base,status='failed',error=repr(exc))
        if not pilot:save(path,result)
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    provenance=json.loads((OUT/'data_provenance.json').read_text(encoding='utf-8'));cases=provenance['cases']
    assert len(cases)==24
    protocol=dict(PROTOCOL,signature=fingerprint(),numpy_version=np.__version__,torch_version=torch.__version__,selection=provenance['selection'])
    save(OUT/'protocol.json',protocol)
    if args.pilot:
        case=next(c for c in cases if c['id']=='feynman-ii.11.20')
        rows=[run_trial((case,0,m,4,.003),pilot=True) for m in METHODS]
        save(OUT/'pilot.json',rows)
        print(json.dumps(rows,indent=2),flush=True)
        assert all(r['status']=='ok' for r in rows)
        return
    items=[(case,seed,method,width,lr) for case in cases for seed in range(3) for method in METHODS for width in WIDTHS[method] for lr in (.001,.003)]
    start=time.perf_counter();failures=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index,future in enumerate(as_completed([pool.submit(run_trial,item) for item in items]),1):
            row=future.result();failures+=int(row['status']!='ok')
            if index%12==0 or row['status']!='ok':print(json.dumps(dict(completed=index,total=len(items),failures=failures,elapsed_seconds=round(time.perf_counter()-start,1),last=row)),flush=True)
    save(OUT/'completion.json',dict(completed=len(items),failed=failures,seconds=time.perf_counter()-start))


if __name__=='__main__':main()
