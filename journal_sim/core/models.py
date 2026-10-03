import hashlib
import numpy as np
from dataclasses import dataclass


def phase_set(bits):
    return np.exp(2j * np.pi * np.arange(2 ** bits) / (2 ** bits))


def phase_indices(theta, bits):
    theta = np.asarray(theta, complex)
    return np.rint(np.mod(np.angle(theta), 2 * np.pi) * (2 ** bits) / (2 * np.pi)).astype(int) % (2 ** bits)


def quantize_theta(theta, bits):
    return phase_set(bits)[phase_indices(theta, bits)]


def array_digest(*arrays):
    digest = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Configuration:
    w: np.ndarray
    theta: np.ndarray

    def __post_init__(self):
        for name in ("w", "theta"):
            a = np.array(getattr(self, name), dtype=np.complex128, copy=True)
            if not np.isfinite(a).all():
                raise ValueError("nonfinite configuration")
            a.setflags(write=False)
            object.__setattr__(self, name, a)

    @property
    def configuration_id(self):
        # Exact bytes: even a tiny W update invalidates an old certificate.
        return array_digest(self.w, self.theta)

    def validate(self, cfg):
        if self.w.shape != (cfg.L, cfg.K, cfg.M) or self.theta.shape != (cfg.N,):
            raise ValueError("configuration dimensions disagree")
        if not np.allclose(self.theta, quantize_theta(self.theta, cfg.bits), atol=1e-12, rtol=0):
            raise ValueError("RIS phases must be discrete unit-modulus states")
        if np.any(np.sum(abs(self.w) ** 2, axis=(1, 2)) > cfg.p_max * (1 + 1e-12)):
            raise ValueError("BS power budget exceeded")
        return self
