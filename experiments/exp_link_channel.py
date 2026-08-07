#!/usr/bin/env python3
"""Does a directed-link memory add a LOCAL arrow channel beyond reciprocal morphology?

Train link_noread so theta is written but cannot alter psi or Ms during growth.
That makes the morphology exactly the symmetric control in the ideal implementation.
After freezing, evaluate:
  delete:      same Ms, theta=0
  full:        same Ms, grown theta read at chi
  crossplant:  theta from one training direction transplanted onto the opposite morphology

The decisive sign test is not 'nonzero'. It is antisymmetry:
  contribution on forward-trained state ~= - contribution on reverse-trained state.
A chi sweep asks whether this second channel can add to, cancel, or reverse the S fossil.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from mgn import Config, MorphogeneticGeometricNeuron
from mgn.analysis import summary
from mgn.operators import plaquette_flux


def probe_state(cfg, mode, state, chi, read_links):
    m=MorphogeneticGeometricNeuron(cfg,mode); m.load_slow_state(state)
    dR,rf,rr=m.preference(read_links=read_links,chi=chi)
    return dR


def zero_links(state):
    q={k:v.clone() for k,v in state.items()}
    q['theta_x'].zero_(); q['theta_y'].zero_(); return q


def transplant_links(base, donor):
    q={k:v.clone() for k,v in base.items()}
    q['theta_x']=donor['theta_x'].clone(); q['theta_y']=donor['theta_y'].clone(); return q


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=6)
    p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--dwell',type=int,default=40)
    p.add_argument('--size',type=int,default=96)
    p.add_argument('--device',default='cuda')
    p.add_argument('--noise',type=float,default=0.08)
    p.add_argument('--mu-a',type=float,default=0.05,dest='mu_a')
    p.add_argument('--chis',type=float,nargs='+',default=[0,1,2,3,4,6])
    p.add_argument('--out',default='runs/link_channel')
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    # Keep trained states once; sweep read strength without retraining.
    all_states=[]
    print('DIRECTED-LINK ARROW EXPERIMENT')
    print(f'  complete cycles={a.cycles}, equal dose={a.cycles*a.dwell}/patch, carrier reset, mu_A={a.mu_a}')
    for seed in range(a.seeds):
        cfg=Config(seed=seed,device=a.device,size=a.size,train_cycles=a.cycles,
                   sequence_dwell=a.dwell,substrate_disorder=a.noise,arrow_mu=a.mu_a)
        states={}
        for order in ('forward','reverse'):
            m=MorphogeneticGeometricNeuron(cfg,'link_noread')
            m.train_cycles(order,a.cycles)
            st=m.clone_slow_state(); states[order]=st
            flux=float(plaquette_flux(st['theta_x'],st['theta_y']).abs().mean().item())
            mag=0.5*(float(st['theta_x'].abs().mean())+float(st['theta_y'].abs().mean()))
            print(f'  seed {seed:02d} {order:7s}: mass {m.mass():.1f}  |theta| {mag:.5f}  |loop flux| {flux:.5f}')
        all_states.append((cfg,states))

    sweep={}
    for chi in a.chis:
        arm_contrib=[]; learned_full=[]; learned_delete=[]; cross_contrib=[]; biases=[]
        rows=[]
        for seed,(cfg,states) in enumerate(all_states):
            sf,sr=states['forward'],states['reverse']
            df0=probe_state(cfg,'link_noread',zero_links(sf),chi,False)
            dr0=probe_state(cfg,'link_noread',zero_links(sr),chi,False)
            dff=probe_state(cfg,'link_noread',sf,chi,True)
            drr=probe_state(cfg,'link_noread',sr,chi,True)
            # Direct read contribution on each trained state.
            cf=dff-df0; cr=drr-dr0
            arm=0.5*(cf-cr); bias=0.5*(cf+cr)
            arm_contrib.append(arm); biases.append(bias)
            learned_full.append(0.5*(dff-drr)); learned_delete.append(0.5*(df0-dr0))

            # Cross-transplant: forward theta onto reverse Ms and vice versa.
            xf=transplant_links(sr,sf); xr=transplant_links(sf,sr)
            dxf=probe_state(cfg,'link_noread',xf,chi,True)
            dxr=probe_state(cfg,'link_noread',xr,chi,True)
            # compare each against its same morphology with theta deleted
            cxf=dxf-dr0; cxr=dxr-df0
            cross=0.5*(cxf-cxr)
            cross_contrib.append(cross)
            rows.append(dict(seed=seed,delete_f=df0,delete_r=dr0,full_f=dff,full_r=drr,
                             direct_f=cf,direct_r=cr,direct_arrow=arm,direct_bias=bias,
                             cross_forward=cxf,cross_reverse=cxr,cross_arrow=cross))

        sd=summary(arm_contrib); sx=summary(cross_contrib)
        sfm=summary(learned_full); sdel=summary(learned_delete); sb=summary(biases)
        sweep[str(chi)]={'direct_link_arrow':sd,'cross_transplant_arrow':sx,
                         'full_learned_effect':sfm,'delete_learned_effect':sdel,
                         'direct_bias':sb,'rows':rows}
        print(f"\nchi={chi:g}: S/delete effect {sdel['mean']:+.5f}  full {sfm['mean']:+.5f}  "
              f"DIRECT LINK {sd['mean']:+.5f} ±{sd['sd']:.5f}  "
              f"p={sd['signflip_p']:.4f} same-sign {sd['same_sign']}/{sd['n']}  "
              f"crossplant {sx['mean']:+.5f}")

    (out/'result.json').write_text(json.dumps({'mu_a':a.mu_a,'chis':a.chis,'sweep':sweep},indent=2),encoding='utf-8')
    print('\nDecision rule: a local arrow channel earns its place only if the direct contribution')
    print('changes sign with training direction (low bias), survives cross-transplant, and is')
    print('stable across seeds/configurations. A chi-dependent zero crossing of the TOTAL effect')
    print('would show that global morphology and local link arrow are independent channels.')

if __name__=='__main__':
    main()
