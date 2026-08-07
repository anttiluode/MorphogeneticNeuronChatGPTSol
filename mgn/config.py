from dataclasses import dataclass, asdict


@dataclass
class Config:
    # Geometry/runtime
    size: int = 96
    seed: int = 7
    device: str = 'cuda'
    root_radius: float = 3.2
    edge_margin: int = 4

    # Four patches on a ring. Training always uses COMPLETE cycles.
    n_patches: int = 4
    patch_radius_frac: float = 0.34
    patch_sigma: float = 4.0
    sequence_dwell: int = 40
    train_cycles: int = 3
    probe_cycles: int = 3
    probe_warmup_cycles: int = 1
    carrier_reset_each_visit: bool = True

    # Fast complex field
    phase_dt: float = 0.012
    diffusion: float = 0.12
    dispersion: float = 0.70
    damping: float = 0.012
    saturation: float = 0.004
    base_conductivity: float = 0.55
    structure_conductivity: float = 1.10
    structure_power: float = 1.20
    substrate_disorder: float = 0.08
    substrate_smoothing: int = 1
    substrate_min: float = 0.65
    substrate_max: float = 1.35
    stability_safety: float = 0.85

    root_drive: float = 0.0
    patch_drive: float = 1.0
    root_frequency: float = 0.065
    patch_frequency: float = 0.55

    # Temporal observables
    lag: int = 10
    symmetric_smoothing: float = 0.16
    arrow_smoothing: float = 0.16

    # Opportunity/Laplacian growth
    opportunity_substeps: int = 12
    opportunity_relaxation: float = 0.60
    opportunity_leak: float = 0.0004
    scaffold_sink_power: float = 5.0
    solid_threshold: float = 0.48
    occupied_threshold: float = 0.44
    flux_quantile: float = 0.94
    growth_eta: float = 3.2
    competition_power: float = 1.35
    deposit_rate: float = 0.20

    # S-write growth gate
    structural_phase_floor: float = 0.55
    symmetric_write_gain: float = 0.45
    scalar_write_gain: float = 0.45

    # Directed-link A memory
    arrow_mu: float = 0.018
    arrow_decay: float = 0.0010
    arrow_clip: float = 1.0
    arrow_current_quantile: float = 0.97
    arrow_scale_ema: float = 0.06
    chi: float = 2.0

    def as_dict(self):
        return asdict(self)
