#!/usr/bin/env python3
from __future__ import annotations
import argparse
import cv2
import numpy as np

from mgn import Config, MorphogeneticGeometricNeuron


def norm(a):
    a=np.asarray(a,float); lo=np.percentile(a,2); hi=np.percentile(a,98)
    return np.clip((a-lo)/(hi-lo+1e-9),0,1)


def colormap(a, cmap):
    u=(255*norm(a)).astype(np.uint8)
    return cv2.applyColorMap(u,cmap)


def panel(img,title):
    out=img.copy(); cv2.putText(out,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA); return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='cuda'); p.add_argument('--size',type=int,default=96)
    p.add_argument('--mode',choices=['scalar','symmetric','link','link_noread'],default='link')
    p.add_argument('--seed',type=int,default=7); p.add_argument('--dwell',type=int,default=40)
    p.add_argument('--steps-per-frame',type=int,default=2)
    a=p.parse_args()
    cfg=Config(device=a.device,size=a.size,seed=a.seed,sequence_dwell=a.dwell)
    m=MorphogeneticGeometricNeuron(cfg,a.mode)
    order='forward'; grow=True; read_links=(a.mode=='link')
    print('controls: q quit | r reset | d reverse order | g freeze growth | a toggle link read | s save')
    while True:
        for _ in range(a.steps_per_frame): m.step(order,grow=grow,read_links=read_links)
        st=m.state_numpy(); flux=np.abs(st['flux'])
        p1=panel(colormap(np.abs(st['psi']),cv2.COLORMAP_TURBO),'|psi| fast field')
        p2=panel(colormap(st['Ms'],cv2.COLORMAP_BONE),'M_S grown body')
        # pad plaquette flux to image size
        fl=np.pad(flux,((0,1),(0,1)))
        p3=panel(colormap(fl,cv2.COLORMAP_MAGMA),'|loop flux| directed memory')
        p4=panel(colormap(st['front'],cv2.COLORMAP_VIRIDIS),'growth front')
        top=np.hstack([p1,p2]); bot=np.hstack([p3,p4]); frame=np.vstack([top,bot])
        met=m.metrics(); txt=f"{a.mode}  {order}  grow={'on' if grow else 'off'}  linkread={'on' if read_links else 'off'}  mass {met['mass']:.1f}  flux {met['plaquette_flux']:.4f}"
        cv2.putText(frame,txt,(8,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        cv2.imshow('Morphogenetic Geometric Neuron v0.2',frame)
        k=cv2.waitKey(1)&0xff
        if k in (ord('q'),27): break
        if k==ord('r'): m=MorphogeneticGeometricNeuron(cfg,a.mode)
        if k==ord('d'): order='reverse' if order=='forward' else 'forward'
        if k==ord('g'): grow=not grow
        if k==ord('a'): read_links=not read_links
        if k==ord('s'): cv2.imwrite('mgn_capture.png',frame); print('wrote mgn_capture.png')
    cv2.destroyAllWindows()

if __name__=='__main__': main()
