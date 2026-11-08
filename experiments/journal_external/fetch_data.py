"""Download pinned official SRSD splits and pinned PMLB Feynman data."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'results_v2/journal_external'
RAW=ROOT/'data/journal_external'


def download(item):
    name,url=item
    path=RAW/name
    if not path.exists():
        last=None
        for attempt in range(3):
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'EML-KAN-reproducibility/1.0'})
                value=urllib.request.urlopen(request,timeout=35).read()
                if url.endswith('.gz'):assert value[:2]==b'\x1f\x8b','Expected gzip, received a pointer or error page'
                path.parent.mkdir(parents=True,exist_ok=True)
                temp=path.with_suffix(path.suffix+'.partial')
                temp.write_bytes(value);temp.replace(path)
                break
            except Exception as exc:last=exc
        else:raise RuntimeError(f'{name}: {last}')
    value=path.read_bytes()
    return dict(path=name,url=url,bytes=len(value),sha256=hashlib.sha256(value).hexdigest())


def main():
    spec=json.loads((HERE/'selection.json').read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    revision=spec['srsd_revision']
    hf=f'https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_hard/resolve/{revision}/'
    jobs=[('srsd/README.md',hf+'README.md'),('srsd/supp_info.json',hf+'supp_info.json')]
    for row in spec['srsd']:
        for split in ('train','val','test'):
            path=split+'/'+row['id']+'.txt'
            jobs.append(('srsd/'+path,hf+path))
    rev=spec['pmlb_revision']
    for row in spec['pmlb']:
        name=row['id']
        jobs.append(('pmlb/'+name+'.tsv.gz',f'https://media.githubusercontent.com/media/EpistasisLab/pmlb/{rev}/datasets/{name}/{name}.tsv.gz'))
        jobs.append(('pmlb/'+name+'_metadata.yaml',f'https://raw.githubusercontent.com/EpistasisLab/pmlb/{rev}/datasets/{name}/metadata.yaml'))
    records=[]
    with ThreadPoolExecutor(6) as pool:
        for index,future in enumerate(as_completed([pool.submit(download,item) for item in jobs]),1):
            record=future.result();records.append(record)
            if index%10==0:print('Downloaded',index,'/',len(jobs),flush=True)
    data_dir=OUT/'data';data_dir.mkdir(exist_ok=True)
    cases=[]
    for row in spec['srsd']:
        arrays={}
        for split,expected in [('train',8000),('val',1000),('test',1000)]:
            values=np.loadtxt(RAW/'srsd'/split/(row['id']+'.txt'),dtype=np.float64)
            assert values.shape==(expected,row['dimension']+1),(row['id'],split,values.shape)
            assert np.isfinite(values).all()
            arrays[split+'_x']=values[:,:-1];arrays[split+'_y']=values[:,[-1]]
        name='srsd__'+row['id']
        np.savez_compressed(data_dir/(name+'.npz'),**arrays)
        cases.append(dict(key=name,collection='SRSD-hard',id=row['id'],dimension=row['dimension'],formula=row['formula'],source_rows=10000,training_rows=8000,validation_rows=1000,test_rows=1000,split='Official 8000/1000/1000, identical for all model seeds'))
    for row in spec['pmlb']:
        name=row['id']
        with gzip.open(RAW/'pmlb'/(name+'.tsv.gz'),'rt') as f:
            columns=f.readline().strip().split('\t')
            target=columns.index('target')
            values=np.loadtxt(f,delimiter='\t',dtype=np.float64)
        features=[i for i in range(len(columns)) if i!=target]
        assert len(features)==row['dimension'] and np.isfinite(values).all()
        assert len(values)>=10000
        key='pmlb__'+name
        for seed in range(3):
            indices=np.random.default_rng(seed).permutation(len(values))[:10000]
            selected=values[indices]
            blocks={'train':selected[:8000],'val':selected[8000:9000],'test':selected[9000:]}
            arrays={split+'_'+kind:v for split,block in blocks.items() for kind,v in [('x',block[:,features]),('y',block[:,[target]])]}
            arrays['source_indices']=indices
            np.savez_compressed(data_dir/(key+f'__s{seed}.npz'),**arrays)
        cases.append(dict(key=key,collection='Feynman-PMLB',id=name,dimension=row['dimension'],columns=columns,target_column='target',source_rows=len(values),training_rows=8000,validation_rows=1000,test_rows=1000,split='Seeded permutation of pinned source, first 10000 partitioned 8000/1000/1000; indices saved'))
    result=dict(selection=spec,sources=sorted(records,key=lambda r:r['path']),cases=cases,
                split_archives=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(data_dir.glob('*.npz'))])
    (OUT/'data_provenance.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Prepared',len(cases),'identified tasks and',len(result['split_archives']),'exact split archives',flush=True)


if __name__=='__main__':main()
