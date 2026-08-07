from __future__ import annotations
import itertools
import numpy as np


def exact_signflip_p(values):
    d = np.asarray(values, float)
    n = len(d)
    if n == 0:
        return float('nan')
    obs = abs(d.mean())
    if n <= 16:
        vals = [abs(np.mean(d*np.asarray(bits)))
                for bits in itertools.product((-1.0,1.0), repeat=n)]
        return float(np.mean(np.asarray(vals) >= obs-1e-12))
    rng = np.random.default_rng(0)
    vals = [abs(np.mean(d*rng.choice([-1.0,1.0], size=n))) for _ in range(20000)]
    return float(np.mean(np.asarray(vals) >= obs))


def summary(values):
    v = np.asarray(values, float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return {
        'n': int(len(v)),
        'mean': float(v.mean()) if len(v) else float('nan'),
        'sd': sd,
        'effect_over_seed_sd': float(v.mean()/(sd+1e-12)) if len(v) else float('nan'),
        'signflip_p': exact_signflip_p(v),
        'same_sign': int(max((v>0).sum(), (v<0).sum())) if len(v) else 0,
    }


def mirror_metrics(a, b):
    """Compare a to left-right mirror of b (the forward/reverse ring symmetry)."""
    a = np.asarray(a, float)
    bm = np.fliplr(np.asarray(b, float))
    denom = np.sqrt(np.mean(a*a)) + 1e-12
    rrmse = float(np.sqrt(np.mean((a-bm)**2))/denom)
    aa = a.ravel()-a.mean(); bb = bm.ravel()-bm.mean()
    corr = float(np.dot(aa,bb)/(np.linalg.norm(aa)*np.linalg.norm(bb)+1e-12))
    return {'mirror_rrmse': rrmse, 'mirror_corr': corr}
