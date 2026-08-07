from __future__ import annotations
import math
import numpy as np
import torch

from .config import Config
from .operators import (
    EPS, choose_device, mean_filter2, max_filter2, normalise01, grad2_zero,
    link_conductivities, covariant_laplacian, link_current, plaquette_flux,
    jacobi_opportunity,
)

MODES = ('scalar', 'symmetric', 'link', 'link_noread')


class MorphogeneticGeometricNeuron:
    """A field that grows geometry from pulse history and then computes through it.

    Fast state
        psi(y,x,t): complex activity.

    Slow reciprocal state
        Ms(y,x): structural/material memory. It changes positive symmetric link
        conductivities and can therefore store global path history without making
        transport locally nonreciprocal.

    Slow directed state
        theta_x, theta_y: signed phases stored ON DIRECTED LINKS. Reverse links
        use -theta automatically. Non-zero plaquette sums are the gauge-invariant
        local arrow / loop flux. This replaces v0.1's scalar Ma stream function.

    Modes
        scalar       : intensity-gated growth, no directed memory
        symmetric    : time-even lag support gates growth, no directed memory
        link         : symmetric growth + directed-link write + link read
        link_noread  : symmetric growth + directed-link write, but links do not
                       affect psi during growth. Best mode for clean transplant/read tests.
    """

    def __init__(self, cfg: Config | None = None, mode: str = 'symmetric'):
        self.cfg = cfg or Config()
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.mode = mode
        self.device = choose_device(self.cfg.device)
        self.rng = np.random.default_rng(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self._build_geometry()
        self.reset()

    def _build_geometry(self):
        c = self.cfg; n = c.size
        yy, xx = torch.meshgrid(
            torch.arange(n, device=self.device, dtype=torch.float32),
            torch.arange(n, device=self.device, dtype=torch.float32), indexing='ij')
        self.yy, self.xx = yy, xx
        self.cx = self.cy = 0.5 * (n - 1)
        self.rr = torch.sqrt((xx-self.cx).square() + (yy-self.cy).square() + EPS)
        self.root_mask = torch.exp(-self.rr.square()/(2*(c.root_radius*0.70)**2))
        self.soma_mask = self.root_mask / (self.root_mask.sum()+EPS)

        radius = n*c.patch_radius_frac
        masks=[]; xy=[]
        # patch 0 top, 1 right, 2 bottom, 3 left: reverse is x-mirror of forward.
        for k in range(c.n_patches):
            a = 2*math.pi*k/c.n_patches - math.pi/2
            x = self.cx + radius*math.cos(a)
            y = self.cy + radius*math.sin(a)
            m = torch.exp(-((xx-x).square()+(yy-y).square())/(2*c.patch_sigma*c.patch_sigma))
            masks.append(m/(m.max()+EPS)); xy.append((x,y))
        self.patch_masks = torch.stack(masks)
        self.patch_xy = xy

        q = torch.arange(n, device=self.device, dtype=torch.float32)
        edge_dist = torch.minimum(q, (n-1)-q)
        w = (edge_dist/max(float(c.edge_margin),1.0)).clamp(0,1)
        w = w*w*(3-2*w)
        self.edge_window = w[:,None]*w[None,:]

        self.outer_reservoir = torch.zeros((n,n), device=self.device)
        self.outer_reservoir[:2,:]=1; self.outer_reservoir[-2:,:]=1
        self.outer_reservoir[:,:2]=1; self.outer_reservoir[:,-2:]=1

    def reset(self):
        c=self.cfg; n=c.size
        self.Ms = torch.sigmoid((c.root_radius-self.rr)*3.0) * self.edge_window
        self.theta_x = torch.zeros((n,n-1), device=self.device)
        self.theta_y = torch.zeros((n-1,n), device=self.device)
        self.opportunity = self.outer_reservoir.clone()
        self.front = torch.zeros_like(self.Ms)

        # Quenched disorder acts from t=0 on the whole medium AND growth, so seeds differ.
        if c.substrate_disorder > 0:
            z = torch.randn((n,n), device=self.device)
            z = mean_filter2(z, c.substrate_smoothing)
            z = (z-z.mean())/(z.std()+EPS)
            self.substrate = torch.exp(c.substrate_disorder*z).clamp(c.substrate_min,c.substrate_max)
        else:
            self.substrate = torch.ones((n,n), device=self.device)

        self._check_stability()
        self.reset_fast_state()

    def reset_fast_state(self):
        c=self.cfg; n=c.size
        self.psi = torch.zeros((n,n), dtype=torch.complex64, device=self.device)
        self.t = 0
        self.history=[]
        self.overlap_history=[]
        self.sym = torch.zeros((n,n), device=self.device)
        self.global_arrow = 0.0
        self._arrow_scale = 1e-3
        self.last_jx = torch.zeros((n,n-1), device=self.device)
        self.last_jy = torch.zeros((n-1,n), device=self.device)

    def _check_stability(self):
        c=self.cfg
        # Conservative Gershgorin-style upper bound for square-grid link Laplacian.
        kmax = (c.base_conductivity+c.structure_conductivity) * float(self.substrate.max().item())
        lam = 8.0*kmax
        denom = lam*(c.diffusion*c.diffusion + c.dispersion*c.dispersion) + EPS
        dt_max = 2.0*c.diffusion/denom
        self.dt_max = dt_max
        if c.phase_dt > c.stability_safety*dt_max:
            raise ValueError(
                f'phase_dt={c.phase_dt:.5f} exceeds safety*explicit-Euler bound '
                f'{c.stability_safety*dt_max:.5f} (raw bound {dt_max:.5f}). '
                f'Lower phase_dt/dispersion/conductivity or raise diffusion.')

    # ---------- drive ----------
    def period(self):
        return self.cfg.n_patches*self.cfg.sequence_dwell

    def sequence(self, order_mode: str, cycle: int = 0):
        n=self.cfg.n_patches
        seq=list(range(n))
        if order_mode == 'forward':
            return seq
        if order_mode == 'reverse':
            return [seq[0]] + seq[1:][::-1]
        if order_mode == 'random':
            rng=np.random.default_rng(self.cfg.seed*1000003 + 7919*cycle + 17)
            return list(rng.permutation(n).astype(int))
        raise ValueError('order_mode must be forward, reverse, or random')

    def active_patch(self, order_mode: str):
        dwell_idx=self.t//self.cfg.sequence_dwell
        cycle=dwell_idx//self.cfg.n_patches
        pos=dwell_idx%self.cfg.n_patches
        return self.sequence(order_mode,cycle)[pos]

    def exposure_counts(self, cycles: int):
        return [cycles*self.cfg.sequence_dwell]*self.cfg.n_patches

    def _source_patch(self, k: int, local_t: int):
        c=self.cfg
        root_phase=c.root_frequency*self.t
        src=c.root_drive*self.root_mask*torch.exp(1j*torch.tensor(root_phase,device=self.device))
        phase=c.patch_frequency*local_t
        src=src+c.patch_drive*self.patch_masks[int(k)]*torch.exp(1j*torch.tensor(phase,device=self.device))
        return src

    def _source(self, order_mode: str):
        c=self.cfg
        k=self.active_patch(order_mode)
        local_t = self.t % c.sequence_dwell if c.carrier_reset_each_visit else self.t
        return self._source_patch(k,local_t)

    # ---------- field ----------
    def _links(self):
        c=self.cfg
        return link_conductivities(self.Ms,self.substrate,
                                   c.base_conductivity,c.structure_conductivity,c.structure_power)

    def _default_read_links(self):
        return self.mode == 'link'

    @torch.no_grad()
    def _advance(self, order_mode: str, read_links: bool | None = None, chi: float | None = None, source_override=None):
        c=self.cfg
        read_links=self._default_read_links() if read_links is None else bool(read_links)
        chi=c.chi if chi is None else float(chi)
        kx,ky=self._links()
        tx=self.theta_x if read_links else None
        ty=self.theta_y if read_links else None
        lap=covariant_laplacian(self.psi,kx,ky,tx,ty,chi if read_links else 0.0)
        src=self._source(order_mode) if source_override is None else source_override
        self.psi=self.psi+c.phase_dt*((c.diffusion+1j*c.dispersion)*lap-c.damping*self.psi+src)
        self.psi=self.psi/(1.0+c.saturation*self.psi.abs().square())
        self.psi=self.psi*self.edge_window

        # Gauge-covariant current through the same links.
        self.last_jx,self.last_jy=link_current(self.psi,kx,ky,tx,ty,chi if read_links else 0.0)

        # Time-even S and V5-style global arrow monitor.
        self.history.append(self.psi.clone())
        if len(self.history)>c.lag+1:
            self.history.pop(0)
        if len(self.history)>=c.lag+1:
            old=self.history[0]
            Ct=self.psi*torch.conj(old)
            agree=0.5*(1.0+Ct.real/(Ct.abs()+EPS))
            support=normalise01(torch.sqrt(self.psi.abs()*old.abs()+EPS),0.97)
            raw=(agree*support).clamp(0,1)
            self.sym=(1-c.symmetric_smoothing)*self.sym+c.symmetric_smoothing*raw

        re=self.psi.real
        rr=torch.stack([(re*m).sum()/(m.sum()+EPS) for m in self.patch_masks])
        self.overlap_history.append(rr.detach().clone())
        if len(self.overlap_history)>c.lag+1:
            self.overlap_history.pop(0)
        if len(self.overlap_history)>=c.lag+1:
            r=self.overlap_history[-1]; ro=self.overlap_history[0]
            z=r+1j*torch.roll(r,-1); zo=ro+1j*torch.roll(ro,-1)
            L=float((z*torch.conj(zo)).imag.mean().item())
            self.global_arrow=(1-c.arrow_smoothing)*self.global_arrow+c.arrow_smoothing*L

    @torch.no_grad()
    def _write_directed_links(self):
        if self.mode not in ('link','link_noread'):
            return
        c=self.cfg
        mx=torch.sqrt((self.Ms[:,:-1]*self.Ms[:,1:]).clamp(min=0))
        my=torch.sqrt((self.Ms[:-1,:]*self.Ms[1:,:]).clamp(min=0))
        vals=torch.cat([self.last_jx.abs().flatten(),self.last_jy.abs().flatten()])
        q=float(torch.quantile(vals,c.arrow_current_quantile).item())+EPS
        self._arrow_scale=(1-c.arrow_scale_ema)*self._arrow_scale+c.arrow_scale_ema*q
        sx=torch.tanh(self.last_jx/(self._arrow_scale+EPS))*mx
        sy=torch.tanh(self.last_jy/(self._arrow_scale+EPS))*my
        self.theta_x=((1-c.arrow_decay)*self.theta_x+c.arrow_mu*sx).clamp(-c.arrow_clip,c.arrow_clip)
        self.theta_y=((1-c.arrow_decay)*self.theta_y+c.arrow_mu*sy).clamp(-c.arrow_clip,c.arrow_clip)

    # ---------- growth ----------
    @torch.no_grad()
    def _solve_opportunity(self):
        c=self.cfg
        sink=(1.0-self.Ms).clamp(0,1).pow(c.scaffold_sink_power)
        self.opportunity=jacobi_opportunity(
            self.opportunity,sink,self.outer_reservoir,
            c.opportunity_relaxation,c.opportunity_leak,c.opportunity_substeps)

    @torch.no_grad()
    def _grow(self):
        c=self.cfg
        solid=(self.Ms>=c.solid_threshold).float()
        boundary=(max_filter2(solid,1)-solid).clamp(0,1)*self.edge_window
        gx,gy=grad2_zero(self.opportunity)
        pressure=torch.sqrt(gx.square()+gy.square()+EPS)
        vals=pressure[boundary>0]
        scale=torch.quantile(vals,c.flux_quantile)+EPS if vals.numel() else pressure.max()+EPS
        pressure=(pressure/scale).clamp(0,1)

        if self.mode == 'scalar':
            amp=normalise01(self.psi.abs(),0.97)
            gate=(c.structural_phase_floor+c.scalar_write_gain*amp).clamp(0,1)
        else:
            gate=(c.structural_phase_floor+c.symmetric_write_gain*self.sym).clamp(0,1)

        growth=boundary*pressure.pow(c.growth_eta)*gate*self.substrate
        if growth.max()>EPS:
            growth=(growth/(growth.max()+EPS)).pow(c.competition_power)
        self.front=growth
        self.Ms=(self.Ms+c.deposit_rate*growth*(1-self.Ms)).clamp(0,1)*self.edge_window

    @torch.no_grad()
    def step(self, order_mode: str='forward', grow: bool=True,
             read_links: bool | None=None, write_links: bool=True,
             chi: float | None=None, source_override=None):
        self._advance(order_mode,read_links=read_links,chi=chi,source_override=source_override)
        if write_links:
            self._write_directed_links()
        if grow:
            self._solve_opportunity(); self._grow()
        self.t+=1
        return self.metrics()

    @torch.no_grad()
    def step_patch(self, patch: int, local_t: int, grow: bool=True,
                   read_links: bool | None=None, write_links: bool=True,
                   chi: float | None=None):
        # Controlled visit step for commutator experiments. local_t is explicit,
        # so every visit can begin at the identical carrier phase.
        src=self._source_patch(int(patch),int(local_t))
        return self.step('forward',grow=grow,read_links=read_links,
                         write_links=write_links,chi=chi,source_override=src)

    # ---------- lifecycle/probe ----------
    def train_cycles(self, order_mode: str='forward', cycles: int | None=None):
        cycles=self.cfg.train_cycles if cycles is None else int(cycles)
        rows=[]
        # No stopping criterion is allowed to decide which stimuli get presented.
        for _ in range(cycles*self.period()):
            rows.append(self.step(order_mode,grow=True,write_links=True))
        return rows

    def clone_slow_state(self):
        return {k:v.clone() for k,v in {
            'Ms':self.Ms,'theta_x':self.theta_x,'theta_y':self.theta_y,'substrate':self.substrate
        }.items()}

    def load_slow_state(self,state):
        self.Ms=state['Ms'].clone().to(self.device)
        self.theta_x=state.get('theta_x',torch.zeros_like(self.theta_x)).clone().to(self.device)
        self.theta_y=state.get('theta_y',torch.zeros_like(self.theta_y)).clone().to(self.device)
        self.substrate=state['substrate'].clone().to(self.device)
        self._check_stability()

    def zero_link_memory(self):
        self.theta_x.zero_(); self.theta_y.zero_()

    @torch.no_grad()
    def probe(self, order_mode: str, cycles: int | None=None,
              read_links: bool | None=None, chi: float | None=None):
        c=self.cfg
        cycles=c.probe_cycles if cycles is None else int(cycles)
        self.reset_fast_state()
        warm=c.probe_warmup_cycles*self.period()
        total=cycles*self.period()
        E=[]; L=[]
        for i in range(total):
            self.step(order_mode,grow=False,read_links=read_links,write_links=False,chi=chi)
            if i>=warm:
                E.append(float((self.soma_mask*self.psi.abs().square()).sum().item()))
                L.append(self.global_arrow)
        return {'root_energy':float(np.mean(E)), 'directed_L':float(np.mean(L))}

    def preference(self, read_links: bool | None=None, chi: float | None=None):
        st=self.clone_slow_state()
        a=MorphogeneticGeometricNeuron(self.cfg,self.mode); a.load_slow_state(st)
        b=MorphogeneticGeometricNeuron(self.cfg,self.mode); b.load_slow_state(st)
        rf=a.probe('forward',read_links=read_links,chi=chi)['root_energy']
        rr=b.probe('reverse',read_links=read_links,chi=chi)['root_energy']
        return (rf-rr)/(rf+rr+1e-12),rf,rr

    # ---------- metrics ----------
    def mass(self):
        return float(self.Ms.sum().item())

    def occupied_fraction(self):
        return float((self.Ms>=self.cfg.occupied_threshold).float().mean().item())

    def metrics(self):
        flux=plaquette_flux(self.theta_x,self.theta_y)
        return {
            'step':int(self.t),
            'mass':self.mass(),
            'occupied':self.occupied_fraction(),
            'link_memory':float(0.5*(self.theta_x.abs().mean()+self.theta_y.abs().mean()).item()),
            'plaquette_flux':float(flux.abs().mean().item()),
            'global_arrow':float(self.global_arrow),
            'dt_max':float(self.dt_max),
        }

    def state_numpy(self):
        return {
            'Ms':self.Ms.detach().cpu().numpy(),
            'theta_x':self.theta_x.detach().cpu().numpy(),
            'theta_y':self.theta_y.detach().cpu().numpy(),
            'flux':plaquette_flux(self.theta_x,self.theta_y).detach().cpu().numpy(),
            'psi':self.psi.detach().cpu().numpy(),
            'front':self.front.detach().cpu().numpy(),
            'opportunity':self.opportunity.detach().cpu().numpy(),
            'substrate':self.substrate.detach().cpu().numpy(),
        }
