"""Single-request inference only; no cohorts, splits, labels, or evaluation."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from .inputs import validate
from .evidence import run_episode,dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True);p.add_argument('--image-root',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--registry',required=True);p.add_argument('--output',required=True)
    p.add_argument('--report-gpus',required=True,help='Two physical GPU indices separated by a comma')
    p.add_argument('--tool-gpu',required=True,help='Separate physical GPU index')
    a=p.parse_args();devices=a.report_gpus.split(',')
    if len(devices)!=2 or len(set(devices+[a.tool_gpu]))!=3:raise ValueError('Two reporter GPUs and one distinct tool GPU required')
    root=Path(a.image_root).resolve();row=json.loads(Path(a.request).read_text());validate(row)
    out=Path(a.output).resolve()
    os.environ['CUDA_VISIBLE_DEVICES']=a.report_gpus
    from .policy import ReportPolicy
    policy=ReportPolicy(a.checkpoint,root)
    def backend(request,action,path):
        file=path.parent/(path.name+'_request.json');dump(file,dict(row=request,action=action))
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=a.tool_gpu
        with path.parent.joinpath(path.name+'_tool.log').open('x') as log:
            subprocess.run([sys.executable,'-m','retina2cardio.tool','--request',str(file),'--image-root',str(root),
                '--registry',str(Path(a.registry).resolve()),'--output',str(path)],env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
        return json.loads((path/'tool_return.json').read_text())
    result=run_episode(row,policy,backend,out)
    print(json.dumps(dict(episode=str(out/'episode.json'),structured_report_valid=result['report_qc'] is not None,
        clinical_quality_verified=False)))


if __name__=='__main__':main()
