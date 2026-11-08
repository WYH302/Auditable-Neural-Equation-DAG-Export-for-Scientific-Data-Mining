"""Check saved external splits, checkpoints, metrics and graph variants offline."""
import hashlib
import json
import math
import numpy as np
import torch
from study import OUT, fingerprint, load_data, normalize, create_model
from graph import export, evaluate, token_count


def main():
    torch.set_num_threads(1)
    provenance=json.loads((OUT/'data_provenance.json').read_text(encoding='utf-8'))
    cases={case['key']:case for case in provenance['cases']}
    for row in provenance['split_archives']:
        assert hashlib.sha256((OUT/'data'/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'trials').glob('*.json'))]
    assert len(rows)==1296 and all(row['status']=='ok' for row in rows)
    worst=0.
    for index,row in enumerate(rows,1):
        assert row['signature']==fingerprint()
        c=row['configuration'];data=load_data(cases[c['case']],c['seed']);normal,scales=normalize(data)
        assert scales==row['scales']
        checkpoint=torch.load(OUT/'checkpoints'/(row['id']+'.pt'),map_location='cpu',weights_only=True)
        assert checkpoint['signature']==row['signature']
        model=create_model(c['method'],c['dimension'],c['width'],c['seed'])
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        predictions={}
        with torch.no_grad():
            for split,(x,truth) in normal.items():
                prediction=model(x).numpy().reshape(-1);yn=truth.numpy().reshape(-1)
                nmse=float(np.mean((prediction-yn)**2)/np.var(yn))
                np.testing.assert_allclose(nmse,row['metrics'][split]['nmse'],rtol=1e-10,atol=1e-12)
                predictions[split]=prediction*scales['y_std']+scales['y_mean']
        variants=json.loads((OUT/'graphs'/(row['id']+'.json')).read_text(encoding='utf-8'))
        assert export(model,c['method'],scales)[0]==variants['raw']
        x=np.vstack((data['val'][0],data['test'][0]));reference=np.concatenate((predictions['val'],predictions['test']))
        with np.errstate(over='ignore',under='ignore'):
            for kind,lines in variants.items():
                assert token_count(';\n'.join(lines))==row['graph'][kind+'_tokens']
                error=float(np.max(np.abs(evaluate(lines,x)-reference)))/scales['y_std']
                assert math.isfinite(error) and error<=row['graph']['normalized_tolerance']
                worst=max(worst,error)
        if index%216==0:print('Verified external candidates',index,flush=True)
    result=dict(candidates=len(rows),tasks=len(cases),exact_splits=len(provenance['split_archives']),graph_variants=3*len(rows),inputs_per_candidate=2000,maximum_error_in_training_target_std_units=worst)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
