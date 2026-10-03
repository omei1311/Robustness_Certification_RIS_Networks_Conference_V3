"""Aggregate complex-ball S-procedure with a fixed congruence preconditioner."""
from dataclasses import dataclass
import numpy as np
from journal_sim.core.uncertainty import stack_users


def quadratic_matrix(w, user, cfg):
    l, k = user
    A = np.zeros((cfg.L * cfg.M, cfg.L * cfg.M), complex)
    for i in range(cfg.L):
        block = sum((-cfg.gamma * np.outer(w[i, j], w[i, j].conj())
                     for j in range(cfg.K)), np.zeros((cfg.M, cfg.M), complex))
        if i == l:
            block += (1 + cfg.gamma) * np.outer(w[i, k], w[i, k].conj())
        sl = slice(i * cfg.M, (i + 1) * cfg.M)
        A[sl, sl] = block
    return (A + A.conj().T) / 2


@dataclass(frozen=True)
class UserLMI:
    A: np.ndarray
    h: np.ndarray
    radius: float
    noise_term: float
    h_scale: float
    scale: float
    user_index: tuple

    @property
    def bottom_base(self):
        return float(np.vdot(self.h, self.A @ self.h).real - self.noise_term)

    def raw(self, lambda_value):
        d = len(self.h)
        M = np.empty((d + 1, d + 1), complex)
        M[:d, :d] = self.A + lambda_value * np.eye(d)
        M[:d, d] = self.A @ self.h
        M[d, :d] = M[:d, d].conj()
        M[d, d] = self.bottom_base - lambda_value * self.radius ** 2
        return (M + M.conj().T) / 2

    def balanced(self, mu):
        """D M(lambda) D / scale, D=diag(h_scale*I,1).

        mu=lambda*h_scale**2/scale. Scale and D do not depend on lambda.
        Thus scalar fast search maximizes a concave minimum eigenvalue.
        """
        d = len(self.h)
        M = np.empty((d + 1, d + 1), complex)
        M[:d, :d] = self.h_scale ** 2 * self.A / self.scale + mu * np.eye(d)
        M[:d, d] = self.h_scale * (self.A @ self.h) / self.scale
        M[d, :d] = M[:d, d].conj()
        M[d, d] = self.bottom_base / self.scale - mu * (self.radius / self.h_scale) ** 2
        return (M + M.conj().T) / 2

    def lambda_from_mu(self, mu):
        return float(mu * self.scale / self.h_scale ** 2)


def make_lmi(w, H_hat, epsilon, user, cfg):
    if np.shape(w) != (cfg.L, cfg.K, cfg.M) or np.shape(H_hat) != (cfg.L, cfg.L, cfg.K, cfg.M):
        raise ValueError("oracle array shape mismatch")
    if not np.isfinite(w).all() or not np.isfinite(H_hat).all() or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("finite arrays and nonnegative radius required")
    h = stack_users(np.asarray(H_hat))[user]
    A = quadratic_matrix(np.asarray(w), user, cfg)
    s = max(float(np.linalg.norm(h)), cfg.radius_floor)
    scale = max(float(np.linalg.norm(A, 2)) * s ** 2, cfg.gamma * cfg.noise_power)
    return UserLMI(A, h, epsilon * s, cfg.gamma * cfg.noise_power, s, scale, user)
