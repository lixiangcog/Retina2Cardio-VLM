"""GPU-isolated actual specialist invocation."""
import argparse,json
from pathlib import Path
from .tools import SpecialistBackend
from .evidence import dump


def main():
    p=argparse.ArgumentParser()
    for k in ('request','image-root','registry','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();request=json.loads(Path(a.request).read_text());registry=json.loads(Path(a.registry).read_text())
    output=Path(a.output);backend=SpecialistBackend(a.image_root,registry,output.parent)
    result=backend(request['row'],request['action'],output);dump(output/'tool_return.json',result)


if __name__=='__main__':main()
