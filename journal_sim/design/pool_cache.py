"""Per-(seed, mobility) candidate pool cache shared across policies.

For a fixed (seed, mobility, time_index) the estimated CSI, the candidate
seed formula and every generated configuration/certificate are policy
independent: build_candidate_pool never sees the policy. Only the decision
to redesign, the previous theta, the transition cost and the final selection
depend on the policy. This provider builds each pool at most once per
(seed, mobility, time_index, observation, config fingerprint) and hands the
same read-only candidates to every policy; selection stays policy-local.

The cache lives exactly one run_paired_policies call (one seed x one
mobility): it is never shared across seeds, mobilities or runs, so stale
configurations and trajectory cross-contamination are impossible by
construction. Cache hits return zeroed cost statistics - the pool's
original build cost is preserved separately for audit, never re-billed.
"""
from dataclasses import dataclass
from journal_sim.config import config_fingerprint
from journal_sim.core.models import array_digest
from .candidate_pool import build_candidate_pool

# Cost counters zeroed on cache hits; descriptive fields are kept for audit.
POOL_COST_KEYS = ("candidate_generation_runtime", "certificate_runtime", "strict_validation_runtime",
                  "fast_oracle_calls", "strict_oracle_calls", "primary_solver_calls",
                  "fallback_solver_calls", "solver_error_count", "solver_inaccurate_count",
                  "end_to_end_runtime")


@dataclass(frozen=True)
class PoolResult:
    candidates: list
    request_stats: dict
    original_stats: dict
    cache_hit: bool
    cache_key: tuple

    @property
    def cache_key_repr(self):
        return "|".join(str(part) for part in self.cache_key)


class CandidatePoolProvider:
    def __init__(self, builder=None):
        # Resolve the module-level builder lazily so tests can patch
        # journal_sim.design.pool_cache.build_candidate_pool.
        self._builder = builder if builder is not None else build_candidate_pool
        self._cache = {}
        self._arrays = {}
        self.requests = 0
        self.builds = 0
        self.cache_hits = 0

    def get_or_build(self, estimated_channel, cfg, seed, time_index, epsilon_est=0.):
        """One pool per (seed, mobility, time_index, observation, config).

        epsilon_est is stamped once at build time so cached candidates stay
        read-only for every policy; a disagreeing requester is a bug.
        """
        observation_id = array_digest(estimated_channel.h_bu, estimated_channel.h_ru,
                                      estimated_channel.h_br)
        key = (seed, cfg.mobility_level, time_index, observation_id, config_fingerprint(cfg))
        self.requests += 1
        entry = self._cache.get(key)
        if entry is None:
            pool, stats = self._builder(estimated_channel, cfg, seed, time_index)
            for c in pool:
                c.epsilon_est = float(epsilon_est)
            self._cache[key] = (pool, stats)
            self.builds += 1
            for c in pool:
                self._arrays[f"pool_t{time_index}_c{c.index}_w"] = c.configuration.w
                self._arrays[f"pool_t{time_index}_c{c.index}_theta"] = c.configuration.theta
                self._arrays[f"pool_t{time_index}_c{c.index}_H_hat"] = c.H_hat
            return PoolResult(pool, stats, stats, False, key)
        pool, original_stats = entry
        for c in pool:
            if c.epsilon_est != float(epsilon_est):
                raise ValueError("cached candidate pool stamped with a different epsilon_est")
        zeroed = {name: 0 for name in POOL_COST_KEYS if name in original_stats}
        zeroed["end_to_end_runtime"] = 0.
        request_stats = dict(original_stats, **zeroed)
        self.cache_hits += 1
        return PoolResult(pool, request_stats, original_stats, True, key)

    def stats(self):
        return dict(candidate_pool_requests=self.requests,
                    candidate_pool_builds=self.builds,
                    candidate_pool_cache_hits=self.cache_hits,
                    candidate_pool_cache_hit_rate=(self.cache_hits / self.requests if self.requests else 0.))

    def collected_arrays(self):
        """Candidate arrays stored once per pool, not once per policy."""
        return dict(self._arrays)
