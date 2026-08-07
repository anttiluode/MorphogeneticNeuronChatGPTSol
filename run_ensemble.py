#!/usr/bin/env python3
"""Convenience launcher for the three v0.2 experiments."""
import argparse, subprocess, sys

p=argparse.ArgumentParser()
p.add_argument('--seeds',type=int,default=8)
p.add_argument('--device',default='cuda')
a=p.parse_args()
cmds=[
    [sys.executable,'experiments/exp_noncommuting_growth.py','--seeds',str(a.seeds),'--device',a.device],
    [sys.executable,'experiments/exp_order_fossil.py','--seeds',str(a.seeds),'--device',a.device,'--mode','symmetric'],
    [sys.executable,'experiments/exp_link_channel.py','--seeds',str(a.seeds),'--device',a.device,'--mu-a','0.15','--chis','0','1','2','3','4','6'],
]
for c in cmds:
    print('\n>', ' '.join(c), flush=True)
    subprocess.run(c,check=True)
