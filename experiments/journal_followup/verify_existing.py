"""Verify saved arrays, checkpoints, exports and faults without retraining."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from graph import evaluate, export, token_count
from study import OUT, HERE, create_model, fingerprint, make_data, normalize


def main():
    torch.set_num_threads(1)
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((OUT/'trials').glob('*.json'))]
    assert len(rows)==756
    checked_data = set()
    maximum = 0.0
    for index, row in enumerate(rows):
        c = row['configuration']
        assert row['status']=='ok' and row['protocol_sha256']==fingerprint()
        assert row['export_sha256']==hashlib.sha256((HERE/'graph.py').read_bytes()).hexdigest()
        key = (c['dataset'],c['seed'])
        with np.load(OUT/'data'/f'{key[0]}__s{key[1]}.npz') as arrays:
            data = {split:(arrays[split+'_x'],arrays[split+'_y']) for split in ('train','val','test','ood')}
        if key not in checked_data:
            generated = make_data(*key)
            for split in data:
                for a,b in zip(data[split],generated[split]):np.testing.assert_array_equal(a,b)
            checked_data.add(key)
        normalized, scales = normalize(data)
        assert scales==row['scales']
        checkpoint = torch.load(OUT/'checkpoints'/(row['id']+'.pt'),map_location='cpu',weights_only=True)
        assert checkpoint['protocol_sha256']==fingerprint()
        model = create_model(c['method'],len(scales['x_mean']),c['width'],c['seed'])
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        predictions = {}
        with torch.no_grad():
            for split,(x,_) in normalized.items():
                prediction = model(x).numpy().reshape(-1)*scales['y_std']+scales['y_mean']
                predictions[split] = prediction
                truth = data[split][1].reshape(-1)
                mse = float(np.mean((prediction-truth)**2))
                np.testing.assert_allclose(mse,row['metrics'][split]['mse'],rtol=1e-10,atol=1e-15)
        graphs = json.loads((OUT/'graphs'/(row['id']+'.json')).read_text(encoding='utf-8'))
        assert export(model,c['method'],scales)[0]==graphs['raw']
        x = np.vstack([data['test'][0],data['ood'][0]])
        reference = np.concatenate([predictions['test'],predictions['ood']])
        for kind,lines in graphs.items():
            assert token_count(';\n'.join(lines))==row['graph'][kind+'_tokens']
            error = float(np.max(np.abs(evaluate(lines,x)-reference)))
            assert error<=row['graph']['fidelity_tolerance']
            maximum = max(maximum,error)
        if (index+1)%126==0:print('Verified candidates',index+1,flush=True)
    faults = list((OUT/'fault_cases').glob('*.json'))
    assert len(faults)==18
    for path in faults:
        case = json.loads(path.read_text(encoding='utf-8'))
        x = np.linspace(.05,3.,512).reshape(-1,1)
        good = evaluate(case['correct'],x,trace=True)
        bad = evaluate(case['corrupted'],x,trace=True)
        first = next(name for name,value in good.items() if np.max(np.abs(bad[name]-value))>1e-9*max(1,float(np.max(np.abs(value)))))
        assert first==case['result']['injected_node']
        np.testing.assert_array_equal(evaluate(case['repaired'],x),good['y'])
    report = dict(candidates=len(rows),exact_splits=len(checked_data),graph_variants=3*len(rows),
                  audited_inputs_per_candidate=2000,maximum_absolute_error=maximum,fault_cases=len(faults))
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
