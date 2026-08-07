#!/usr/bin/env python3
import numpy as np
import torch

from mgn import Config, MorphogeneticGeometricNeuron
from mgn.operators import (
    covariant_laplacian, link_current, plaquette_flux, scalar_gradient_links,
)

print('Morphogenetic Geometric Neuron v0.2 selftest')

# 1. Conservative link operator: constant field is stationary with zero link phase.
n=24
psi=torch.ones((n,n),dtype=torch.complex64)
kx=torch.rand((n,n-1))+0.2
ky=torch.rand((n-1,n))+0.2
L=covariant_laplacian(psi,kx,ky)
e=float(L.abs().max())
print(f'  conservative constant-field error  {e:.3e}')
assert e < 1e-6

# 2. Current is odd under conjugation/time reversal.
yy,xx=torch.meshgrid(torch.arange(n),torch.arange(n),indexing='ij')
z=torch.exp(1j*(0.21*xx+0.13*yy)).to(torch.complex64)
jx,jy=link_current(z,kx,ky)
cx,cy=link_current(torch.conj(z),kx,ky)
err=max(float((jx+cx).abs().max()),float((jy+cy).abs().max()))
print(f'  directed-link current reversal error {err:.3e}')
assert err < 1e-5

# 3. Pure scalar-gradient link memory is gauge-trivial: zero plaquette flux.
phi=torch.randn((n,n))
tx,ty=scalar_gradient_links(phi)
f=float(plaquette_flux(tx,ty).abs().max())
print(f'  pure-gradient plaquette flux        {f:.3e}')
assert f < 2e-6

# 4. A genuinely directed loop has non-zero gauge-invariant flux.
tx=torch.zeros((n,n-1)); ty=torch.zeros((n-1,n))
tx[8,8]=0.4
f=float(plaquette_flux(tx,ty).abs().max())
print(f'  directed-link loop flux             {f:.3f}')
assert f > 0.3

# 5. Strict cycle exposure: endpoint stopping can no longer change the dose.
cfg=Config(size=48,device='cpu',train_cycles=3,sequence_dwell=12,
           substrate_disorder=0.0,opportunity_substeps=4)
m=MorphogeneticGeometricNeuron(cfg,'symmetric')
counts=m.exposure_counts(3)
print(f'  strict-cycle exposure counts        {counts}')
assert len(set(counts)) == 1 and counts[0] == 36

# 6. Stability guard catches Claude's quiet explicit-Euler failure mode.
try:
    MorphogeneticGeometricNeuron(Config(size=32,device='cpu',phase_dt=0.20,substrate_disorder=0.0),'symmetric')
    raise AssertionError('stability guard failed to fire')
except ValueError:
    print('  explicit-Euler stability guard      PASS')

# 7. Exact geometric centre + phase-reset drive: an UNTRAINED symmetric medium
# must not prefer forward vs reverse. This is the mirror/readout calibration.
cfg=Config(size=48,device='cpu',substrate_disorder=0.0,sequence_dwell=10,
           train_cycles=1,probe_cycles=2,probe_warmup_cycles=1,
           opportunity_substeps=3,phase_dt=0.008)
m=MorphogeneticGeometricNeuron(cfg,'symmetric')
p,rf,rr=m.preference(read_links=False)
print(f'  untrained mirror preference         {p:+.6f}')
assert abs(p) < 2e-4

# 8. One complete growth cycle in a clean substrate should mirror under reversal.
f=MorphogeneticGeometricNeuron(cfg,'symmetric'); f.train_cycles('forward',1)
r=MorphogeneticGeometricNeuron(cfg,'symmetric'); r.train_cycles('reverse',1)
A=f.Ms.detach().cpu().numpy(); B=np.fliplr(r.Ms.detach().cpu().numpy())
rrmse=np.sqrt(np.mean((A-B)**2))/(np.sqrt(np.mean(A*A))+1e-12)
print(f'  clean forward/reverse morphology mirror rRMSE {rrmse:.5f}')
assert rrmse < 0.08

print('SELFTEST PASS')
