#!/usr/bin/env python3
"""Direct mechanism test: do symmetric plasticity updates fail to commute?

For two spatial stimuli A and B, compare equal-dose growth:
    M_AB = F_B(F_A(M0))
    M_BA = F_A(F_B(M0))

Transport is reciprocal (theta=0). If M_AB != M_BA above a stimulus-blind null,
then temporal order has become geometry through path-dependent plasticity rather
than local nonreciprocity.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from mgn import Config, MorphogeneticGeometricNeuron
from mgn.analysis import summary


def visit(model,patch,dwell,reset_fast=False):
    if reset_fast:
        model.reset_fast_state()
    for local_t in range(dwell):
        model.step_patch(patch,local_t,grow=True,read_links=False,write_links=False)


def cfg_replace(cfg, **kw):
    d=cfg.as_dict(); d.update(kw); return Config(**d)


def run_order(cfg,first,second,repeats,active=True,reset_fast=False):
    if not active:
        cfg=cfg_replace(cfg,symmetric_write_gain=0.0,scalar_write_gain=0.0,structural_phase_floor=1.0)
    m=MorphogeneticGeometricNeuron(cfg,'symmetric')
    for _ in range(repeats):
        visit(m,first,cfg.sequence_dwell,reset_fast); visit(m,second,cfg.sequence_dwell,reset_fast)
    return m


def distance(a,b):
    A=a.Ms.detach().cpu().numpy(); B=b.Ms.detach().cpu().numpy()
    return float(np.sqrt(np.mean((A-B)**2))/(np.sqrt(np.mean(0.5*(A*A+B*B)))+1e-12))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=8)
    p.add_argument('--size',type=int,default=96)
    p.add_argument('--device',default='cuda')
    p.add_argument('--dwell',type=int,default=40)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--patch-a',type=int,default=1)
    p.add_argument('--patch-b',type=int,default=3)
    p.add_argument('--noise',type=float,default=0.08)
    p.add_argument('--out',default='runs/noncommuting_growth')
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    active=[]; slow_only=[]; null=[]; rows=[]
    print('NONCOMMUTING SYMMETRIC GROWTH')
    print(f'  A={a.patch_a}, B={a.patch_b}, dwell={a.dwell}, repeats={a.repeats}; exact equal dose')
    for sd in range(a.seeds):
        cfg=Config(seed=sd,device=a.device,size=a.size,sequence_dwell=a.dwell,substrate_disorder=a.noise)
        ab=run_order(cfg,a.patch_a,a.patch_b,a.repeats,True)
        ba=run_order(cfg,a.patch_b,a.patch_a,a.repeats,True)
        # Reset psi/history at every visit but keep Ms/U: isolates slow stigmergic path dependence.
        sab=run_order(cfg,a.patch_a,a.patch_b,a.repeats,True,True)
        sba=run_order(cfg,a.patch_b,a.patch_a,a.repeats,True,True)
        nab=run_order(cfg,a.patch_a,a.patch_b,a.repeats,False,True)
        nba=run_order(cfg,a.patch_b,a.patch_a,a.repeats,False,True)
        da=distance(ab,ba); ds=distance(sab,sba); dn=distance(nab,nba)
        active.append(da); slow_only.append(ds); null.append(dn)
        rows.append({'seed':sd,'active_commutator':da,'slow_stigmergic_commutator':ds,'null_commutator':dn})
        print(f'  seed {sd:02d}: carry {da:.6f}   reset-fast/slow-only {ds:.6f}   null {dn:.3e}')
    sa=summary(active); ss=summary(slow_only); sn=summary(null)
    result={'active_with_fast_carry':sa,'slow_stigmergic_only':ss,'null':sn,'rows':rows,'patches':[a.patch_a,a.patch_b],
            'dwell':a.dwell,'repeats':a.repeats}
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('\n--- result ---')
    print(f"with fast carry      {sa['mean']:.6f} ±{sa['sd']:.6f}")
    print(f"reset-fast slow only {ss['mean']:.6f} ±{ss['sd']:.6f}")
    print(f"stimulus-blind null  {sn['mean']:.6e} ±{sn['sd']:.6e}")
    print('The reset-fast arm is the sharp stigmergic test: if it remains above null,')
    print('the FIRST structural write changes the substrate seen by the SECOND even when')
    print('the fast wave itself is erased between visits. That is slow developmental memory.')

if __name__=='__main__': main()
