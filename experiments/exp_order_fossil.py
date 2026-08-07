#!/usr/bin/env python3
"""Strict test: can time-even reciprocal growth fossilise sequence order?

The protocol fixes the two confounds found in v0.1 and in the independent Claude build:
  * training is an integer number of complete cycles -> identical per-patch dose;
  * carrier phase resets at each patch visit -> patch identity is not tied to carrier epoch.

Primary statistic:
  dR(structure) = (R_forward - R_reverse)/(R_forward + R_reverse)
  learned effect = (dR_forward-trained - dR_reverse-trained)/2
  bias           = (dR_forward-trained + dR_reverse-trained)/2

Run the S arm first. If it survives seed/random-order controls, the simple result is:
a reciprocal morphology can encode temporal order globally through path-dependent growth.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from mgn import Config, MorphogeneticGeometricNeuron
from mgn.analysis import summary, mirror_metrics


def train_score(cfg, mode, order):
    m=MorphogeneticGeometricNeuron(cfg,mode)
    m.train_cycles(order,cfg.train_cycles)
    dR,rf,rr=m.preference(read_links=False)
    return m,dR,rf,rr


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seeds',type=int,default=8)
    p.add_argument('--mode',choices=['scalar','symmetric'],default='symmetric')
    p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--dwell',type=int,default=40)
    p.add_argument('--size',type=int,default=96)
    p.add_argument('--device',default='cuda')
    p.add_argument('--noise',type=float,default=0.08)
    p.add_argument('--out',default='runs/order_fossil')
    p.add_argument('--no-random',action='store_true')
    a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)

    effects=[]; biases=[]; random_scores=[]; rows=[]
    print('STRICT ORDER-FOSSIL EXPERIMENT')
    print(f'  mode={a.mode} cycles={a.cycles} dwell={a.dwell} size={a.size} seeds={a.seeds}')
    print(f'  exact dose per patch = {a.cycles*a.dwell} frames; carrier resets every visit\n')

    # Positive-control calibration: paint a strongly asymmetric reciprocal morphology.
    # If the probe cannot see this, a null learned effect is instrument-blind.
    ccal=Config(seed=0,device=a.device,size=a.size,sequence_dwell=a.dwell,
                substrate_disorder=0.0,train_cycles=a.cycles)
    cal=MorphogeneticGeometricNeuron(ccal,'symmetric')
    import torch
    cal.Ms=(0.05+0.90*torch.sigmoid((cal.xx-cal.cx)/2.0))*cal.edge_window
    cal_dR,_,_=cal.preference(read_links=False)
    print(f'  painted-asymmetry positive control dR = {cal_dR:+.5f}\n')

    for seed in range(a.seeds):
        cfg=Config(seed=seed,device=a.device,size=a.size,train_cycles=a.cycles,
                   sequence_dwell=a.dwell,substrate_disorder=a.noise)
        mf,df,rf_f,rr_f=train_score(cfg,a.mode,'forward')
        mr,dr,rf_r,rr_r=train_score(cfg,a.mode,'reverse')
        eff=0.5*(df-dr); bias=0.5*(df+dr)
        effects.append(eff); biases.append(bias)
        mm=mirror_metrics(mf.Ms.detach().cpu().numpy(),mr.Ms.detach().cpu().numpy())
        rnd=float('nan')
        if not a.no_random:
            mq,dq,_,_=train_score(cfg,a.mode,'random'); rnd=dq; random_scores.append(dq)
        row=dict(seed=seed,dR_fwdtrain=df,dR_revtrain=dr,effect=eff,bias=bias,
                 mass_fwd=mf.mass(),mass_rev=mr.mass(),random_dR=rnd,**mm)
        rows.append(row)
        print(f"  seed {seed:02d}: dR fwd {df:+.5f} rev {dr:+.5f}  effect {eff:+.5f}  "
              f"bias {bias:+.5f}  mass {mf.mass():.1f}/{mr.mass():.1f}  "
              f"mirror corr {mm['mirror_corr']:+.3f}" +
              (f"  random {rnd:+.5f}" if not a.no_random else ''))

    s=summary(effects); b=summary(biases)
    result={'protocol':{'mode':a.mode,'cycles':a.cycles,'dwell':a.dwell,'size':a.size,
                        'dose_per_patch':a.cycles*a.dwell,'carrier_reset_each_visit':True},
            'learned_effect':s,'bias':b,'rows':rows}
    print('\n--- strict verdict inputs ---')
    print(f"learned effect {s['mean']:+.6f}  seed SD {s['sd']:.6f}  effect/SD {s['effect_over_seed_sd']:+.2f}  "
          f"exact sign-flip p {s['signflip_p']:.4f}  same-sign {s['same_sign']}/{s['n']}")
    print(f"bias           {b['mean']:+.6f}  seed SD {b['sd']:.6f}")
    if random_scores:
        rs=float(np.std(random_scores,ddof=1)) if len(random_scores)>1 else 0.0
        rm=float(np.mean(random_scores))
        result['random_control']={'mean':rm,'sd':rs,'effect_over_random_spread':abs(s['mean'])/(rs+1e-12)}
        print(f'random-order dR mean {rm:+.6f}  spread ±{rs:.6f}  |effect|/spread {abs(s["mean"])/(rs+1e-12):.2f}')
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

    print('\nInterpretation rule:')
    print('  * nonzero alone proves nothing; judge against seed and random-order spread;')
    print('  * if the S arm survives here, call it GLOBAL ORDER FOSSILISATION, not local nonreciprocity;')
    print('  * if it dies, the earlier S result was a protocol artifact.')

if __name__=='__main__':
    main()
