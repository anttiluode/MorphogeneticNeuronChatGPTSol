from __future__ import annotations
import torch
import torch.nn.functional as F

EPS = 1e-9


def choose_device(requested: str) -> torch.device:
    if requested.startswith('cuda') and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device('cpu')


def mean_filter2(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = 2 * radius + 1
    return F.avg_pool2d(x[None, None], k, stride=1, padding=radius)[0, 0]


def max_filter2(x: torch.Tensor, radius: int = 1) -> torch.Tensor:
    k = 2 * radius + 1
    return F.max_pool2d(x[None, None], k, stride=1, padding=radius)[0, 0]


def normalise01(x: torch.Tensor, q: float = 0.98) -> torch.Tensor:
    vals = x.flatten()
    scale = torch.quantile(vals, q) if vals.numel() else x.max()
    return (x / (scale + EPS)).clamp(0.0, 1.0)


def grad2_zero(x: torch.Tensor):
    """Central gradient with one-sided zero-flux-ish edges; no periodic roll."""
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
    gx[:, 0] = x[:, 1] - x[:, 0]
    gx[:, -1] = x[:, -1] - x[:, -2]
    gy[1:-1, :] = 0.5 * (x[2:, :] - x[:-2, :])
    gy[0, :] = x[1, :] - x[0, :]
    gy[-1, :] = x[-1, :] - x[-2, :]
    return gx, gy


def link_conductivities(ms: torch.Tensor, substrate: torch.Tensor,
                        k0: float, k1: float, power: float):
    """Symmetric positive conductivities on +x and +y links."""
    mx = 0.5 * (ms[:, :-1] + ms[:, 1:])
    my = 0.5 * (ms[:-1, :] + ms[1:, :])
    sx = 0.5 * (substrate[:, :-1] + substrate[:, 1:])
    sy = 0.5 * (substrate[:-1, :] + substrate[1:, :])
    kx = (k0 + k1 * mx.clamp(min=0).pow(power)) * sx
    ky = (k0 + k1 * my.clamp(min=0).pow(power)) * sy
    return kx, ky


def covariant_laplacian(psi: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor,
                         theta_x: torch.Tensor | None = None,
                         theta_y: torch.Tensor | None = None,
                         chi: float = 0.0) -> torch.Tensor:
    """Conservative nearest-neighbour Hermitian link operator.

    theta_x[y,x] is the stored phase on the directed edge (y,x)->(y,x+1).
    theta_y[y,x] is the stored phase on the directed edge (y,x)->(y+1,x).
    Reverse edges use the conjugate phase automatically.
    """
    if theta_x is None:
        ex = torch.ones_like(kx, dtype=psi.dtype)
    else:
        ex = torch.exp(1j * chi * theta_x)
    if theta_y is None:
        ey = torch.ones_like(ky, dtype=psi.dtype)
    else:
        ey = torch.exp(1j * chi * theta_y)

    out = torch.zeros_like(psi)
    # Horizontal edge i(left) <-> j(right)
    out[:, :-1] += kx * (ex * psi[:, 1:] - psi[:, :-1])
    out[:, 1:] += kx * (torch.conj(ex) * psi[:, :-1] - psi[:, 1:])
    # Vertical edge i(top) <-> j(bottom)
    out[:-1, :] += ky * (ey * psi[1:, :] - psi[:-1, :])
    out[1:, :] += ky * (torch.conj(ey) * psi[:-1, :] - psi[1:, :])
    return out


def link_current(psi: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor,
                 theta_x: torch.Tensor | None = None,
                 theta_y: torch.Tensor | None = None,
                 chi: float = 0.0):
    """Gauge-covariant oriented link current; odd under complex conjugation."""
    ex = torch.ones_like(kx, dtype=psi.dtype) if theta_x is None else torch.exp(1j * chi * theta_x)
    ey = torch.ones_like(ky, dtype=psi.dtype) if theta_y is None else torch.exp(1j * chi * theta_y)
    jx = kx * (torch.conj(psi[:, :-1]) * ex * psi[:, 1:]).imag
    jy = ky * (torch.conj(psi[:-1, :]) * ey * psi[1:, :]).imag
    return jx, jy


def plaquette_flux(theta_x: torch.Tensor, theta_y: torch.Tensor) -> torch.Tensor:
    """Gauge-invariant oriented loop sum around each elementary plaquette."""
    return (theta_x[:-1, :] + theta_y[:, 1:] - theta_x[1:, :] - theta_y[:, :-1])


def scalar_gradient_links(phi: torch.Tensor):
    """Pure-gauge directed links. Their plaquette flux is zero up to roundoff."""
    return phi[:, 1:] - phi[:, :-1], phi[1:, :] - phi[:-1, :]


def jacobi_opportunity(p: torch.Tensor, sink: torch.Tensor, reservoir: torch.Tensor,
                       relaxation: float, leak: float, steps: int) -> torch.Tensor:
    """No-wrap Jacobi relaxation for the scalar opportunity field."""
    for _ in range(steps):
        neigh = torch.zeros_like(p)
        count = torch.zeros_like(p)
        neigh[:, 1:] += p[:, :-1]; count[:, 1:] += 1
        neigh[:, :-1] += p[:, 1:]; count[:, :-1] += 1
        neigh[1:, :] += p[:-1, :]; count[1:, :] += 1
        neigh[:-1, :] += p[1:, :]; count[:-1, :] += 1
        neigh = neigh / count.clamp(min=1)
        p = (1-relaxation) * p + relaxation * neigh
        p = p * sink * (1-leak)
        p = torch.maximum(p, reservoir)
    return p.clamp(0, 1)
