"""Sequential matched full/nominal runs. Fresh weights and identical data seeds."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--subcarriers',type=int,default=8)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=(args.output or root/'artifacts'/datetime.now().strftime('paired_joint_%Y%m%d_%H%M%S_%f')).resolve()
    summary={}
    for mode in ('full','nominal'):
        run=output/mode
        command=[sys.executable,'-m','main.train','--geometry-mode',mode,
            '--seed',str(args.seed),'--device',args.device,'--batch-size',str(args.batch_size),
            '--epochs',str(args.epochs),'--subcarriers',str(args.subcarriers),'--output',str(run)]
        if args.quick:
            command.append('--quick')
        print(f'\nTRAINING {mode.upper()}',flush=True)
        subprocess.run(command,cwd=root,check=True)
        summary[mode]=json.loads((run/'evaluation_results.json').read_text())
    (output/'comparison.json').write_text(json.dumps(summary,indent=2))
    print('\nJOINT ENCODER COMPARISON',flush=True)
    print(json.dumps(summary,indent=2),flush=True)
    print(f'Saved: {output}',flush=True)


if __name__=='__main__':
    main()
