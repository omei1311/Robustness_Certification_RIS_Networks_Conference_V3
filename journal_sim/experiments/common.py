"""Run manifests, checkpointed raw records and immutable run directories.

Parallelism contract (seed-level only):
- Only the outer loop over independent seeds may run in parallel; one seed
  (all its mobilities, policies, candidates and SDPs) is executed start to
  finish by exactly one worker process.
- ``workers`` is a runtime option (ExecutionOptions), deliberately excluded
  from JournalConfig and from every scientific fingerprint.
- Workers never write shared outputs; they stage their large arrays to
  ``_worker_staging/seed_<seed>_arrays.npz`` and the parent merges them in
  ``cfg.seeds`` order, so completion order cannot influence recorded order.
- Windows spawn is used, worker callables are resolved from their real
  importable module (``python -m`` support), and BLAS thread env vars are
  pinned (without overriding user settings) to avoid oversubscription.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from csv import DictWriter
from dataclasses import dataclass, fields
from datetime import datetime, timezone, timedelta
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import cvxpy
import scipy
from journal_sim.config import JournalConfig, smoke_config, config_fingerprint

ROOT = Path(__file__).resolve().parents[2]

BLAS_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass(frozen=True)
class ExecutionOptions:
    """Runtime-only knobs; never part of any scientific fingerprint."""
    workers: int = 1

    def validate(self):
        if not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("workers must be a positive integer")
        return self


def serializable(value):
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, payload):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(serializable(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # Windows briefly locks freshly written files (defender/indexer); retry
    # the atomic replace a few times before giving up.
    for attempt in range(6):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.2 * (attempt + 1))


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(serializable(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else serializable(v) for k, v in row.items()})


def commit_id():
    head = ROOT / ".git/HEAD"
    if not head.exists():
        return None
    value = head.read_text().strip()
    if not value.startswith("ref: "):
        return value
    ref = value[5:]
    path = ROOT / ".git" / ref
    if path.exists():
        return path.read_text().strip()
    packed = ROOT / ".git/packed-refs"
    if packed.exists():
        for line in packed.read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    return None


def source_hashes():
    paths = [p for p in (ROOT / "journal_sim").rglob("*") if p.is_file() and p.suffix in (".py", ".npz", ".json") and "__pycache__" not in p.parts]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def working_tree_status():
    """(dirty, method): True/False when git is usable, (None, reason) if not.

    Formal paper results require dirty=false; when git is unavailable the
    manifest records null and source_hashes remain the authoritative state.
    """
    git = shutil.which("git")
    if git is None:
        return None, "git_unavailable"
    try:
        status = subprocess.run([git, "status", "--porcelain"], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None, "git_error"
    if status.returncode != 0:
        return None, "git_error"
    return bool(status.stdout.strip()), "git_status_porcelain"


def load_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "config" in payload:
        payload = payload["config"]
    def tuples(v):
        return tuple(tuples(x) for x in v) if isinstance(v, list) else v
    # Saved configs may carry retired keys; only known fields are loaded.
    known = {f.name for f in fields(JournalConfig)}
    payload = {k: v for k, v in payload.items() if k in known}
    defaults = JournalConfig()
    return JournalConfig(**{k: tuples(v) if isinstance(getattr(defaults, k), tuple) else v for k, v in payload.items()}).validate()


def cli_config(description):
    parser = ArgumentParser(description=description)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=str)
    parser.add_argument("--n-seeds", type=int)
    parser.add_argument("--time-steps", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes over independent seeds (runtime only; 1 = serial)")
    parser.add_argument("--direct-intercell", choices=("on", "off"))
    parser.add_argument("--mobility", choices=("slow", "medium", "fast"))
    parser.add_argument("--selection", choices=("lifetime_aware", "wee_only", "robustness_only", "stability_aware"))
    parser.add_argument("--output-root", type=str)
    parser.add_argument("--csi-calibration", type=str, help="Exp1 joint_radius.json; hash is recorded in full config")
    parser.add_argument("--drift-calibration", type=str, help="Exp1b drift_rate.json; hash is recorded in full config")
    parser.add_argument("--epsilon-est", type=float, help="Explicit pre-calibrated radius; no online Monte Carlo")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be a positive integer")
    cfg = load_config(args.config) if args.config else smoke_config() if args.smoke else JournalConfig().validate()
    overrides = {}
    if args.smoke:
        overrides["smoke"] = True
    if args.n_seeds is not None:
        if args.n_seeds < 1:
            parser.error("n-seeds must be positive")
        overrides["seeds"] = tuple(range(cfg.seeds[0], cfg.seeds[0] + args.n_seeds))
    for arg, field in (("time_steps", "time_steps"), ("pool_size", "pool_size"),
                       ("mobility", "mobility_level"), ("selection", "selection_rule"), ("output_root", "output_root")):
        value = getattr(args, arg)
        if value is not None:
            overrides[field] = value
    if args.direct_intercell:
        overrides["include_direct_intercell"] = args.direct_intercell == "on"
    if args.epsilon_est is not None:
        overrides.update(epsilon_est=args.epsilon_est, csi_calibration_file=None, csi_calibration_sha256=None)
    cfg = cfg.with_overrides(**overrides)
    if args.csi_calibration:
        from journal_sim.dynamics.offline_calibration import bind_artifact
        cfg = bind_artifact(cfg, args.csi_calibration)
    elif cfg.csi_calibration_file and not cfg.csi_calibration_sha256:
        from journal_sim.dynamics.offline_calibration import bind_artifact
        cfg = bind_artifact(cfg, cfg.csi_calibration_file)
    if args.drift_calibration:
        from journal_sim.dynamics.drift_calibration import bind_drift_artifact
        cfg = bind_drift_artifact(cfg, args.drift_calibration)
    elif cfg.drift_calibration_file and not cfg.drift_calibration_sha256:
        from journal_sim.dynamics.drift_calibration import bind_drift_artifact
        cfg = bind_drift_artifact(cfg, cfg.drift_calibration_file)
    return cfg, ExecutionOptions(workers=args.workers).validate()


def _resolve_run_seed_module(run_seed):
    """Real importable module for a ``python -m``-launched run_seed callable."""
    module_name = getattr(run_seed, "__module__", "")
    if module_name != "__main__":
        return module_name
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is None or not spec.name:
        raise ValueError("parallel workers need an importable experiment module; "
                         "run experiments via python -m journal_sim.experiments.<experiment>")
    return spec.name


def _run_seed_worker(payload):
    """Spawn-side entry: rebuild the experiment callable, stage large arrays.

    The worker only writes its own staging file; shared records/manifest/
    arrays are written exclusively by the parent process.
    """
    import importlib
    module = importlib.import_module(payload["module_name"])
    result = getattr(module, "run_seed")(payload["cfg"], payload["seed"])
    arrays = result.pop("arrays", {}) or {}
    staging = Path(payload["staging_path"])
    staging.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(staging, **arrays)
    result["staging_file"] = str(staging)
    result["staged_array_keys"] = sorted(arrays)
    return result


def execute(experiment, cfg, run_seed, plot, workers=1):
    options = ExecutionOptions(workers=workers).validate()
    timestamp = datetime.now(timezone(timedelta(hours=8)))
    fingerprint = config_fingerprint(cfg)
    mode = "smoke" if cfg.smoke else "formal"
    output = Path(cfg.output_root)
    if not output.is_absolute():
        output = ROOT / output
    output = output / experiment / mode / (timestamp.strftime("%Y%m%dT%H%M%S%f") + "_" + fingerprint[:12])
    output.mkdir(parents=True, exist_ok=False)
    parallel = options.workers > 1 and len(cfg.seeds) > 1
    if parallel:
        # Pin BLAS threads before spawning workers; never override user settings.
        for name in BLAS_THREAD_ENV:
            os.environ.setdefault(name, "1")
    execution = dict(parallel=parallel,
                     workers_requested=options.workers,
                     workers_effective=min(options.workers, len(cfg.seeds)) if parallel else 1,
                     multiprocessing_context="spawn" if parallel else None,
                     runtime_measurement_valid=not parallel,
                     blas_thread_env={name: os.environ.get(name) for name in BLAS_THREAD_ENV})
    staging_dir = output / "_worker_staging"
    dirty, dirty_method = working_tree_status()
    if mode == "formal" and dirty:
        print("WARNING: formal run started from a dirty working tree.\n"
              "git_commit alone does not uniquely identify source state; "
              "final paper results require a clean tree.", flush=True)
    manifest = dict(experiment=experiment, mode=mode, config=cfg.to_dict(),
                    seeds=cfg.seeds, fingerprint=fingerprint, timestamp=timestamp.isoformat(),
                    git_commit=commit_id(),
                    git=dict(commit=commit_id(), dirty=dirty, method=dirty_method),
                    source_hashes=source_hashes(),
                    versions=dict(numpy=np.__version__, scipy=scipy.__version__, cvxpy=cvxpy.__version__),
                    execution=execution,
                    note="Smoke results verify execution only. All requested seeds and failed/uncertain outcomes remain in raw records.")
    rows, details, extra, arrays = [], [], [], {}
    interrupted = False

    save_json(output / "manifest.json", {**manifest, "seed_details": [], "details": [], "run_complete": False})

    def ingest(seed, result):
        seed_rows = result["records"]
        rows.extend(dict(seed=seed, **{k: v for k, v in row.items() if k != "seed"}) for row in seed_rows)
        details.append(dict(seed=seed, status=result.get("status", "COMPLETED"),
                            summary=result.get("summary", result.get("summaries"))))
        extra.extend(result.get("details", result.get("designs", [])))
        staged = result.get("staging_file")
        if staged is not None:
            with np.load(staged) as data:
                for key in data.files:
                    arrays[f"seed{seed}_{key}"] = data[key]
        for key, value in result.get("arrays", {}).items():
            arrays[f"seed{seed}_{key}"] = value

    def record_failure(seed, exc):
        failure = dict(seed=seed, status="FAILED", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
        rows.append(failure)
        details.append(failure)

    def mark_remaining_interrupted():
        # Keep already-finished seeds; cancelled ones are recorded, never dropped.
        done = {d.get("seed") for d in details}
        for seed in cfg.seeds:
            if seed not in done:
                details.append(dict(seed=seed, status="INTERRUPTED",
                                    error="KeyboardInterrupt: cancelled before completion"))

    def checkpoint():
        save_csv(output / "records.csv", rows)
        save_json(output / "manifest.json", {**manifest, "seed_details": details, "details": extra,
                                             "run_complete": len(details) == len(cfg.seeds)})
        np.savez_compressed(output / "arrays.npz", **arrays)

    def announce(seed):
        print(f"[{experiment}] seed={seed} status={details[-1]['status']}", flush=True)

    if not parallel:
        # Serial path: no executor, straightforward debugging, valid runtime benchmark.
        try:
            for seed in cfg.seeds:
                try:
                    ingest(seed, run_seed(cfg, seed))
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    record_failure(seed, exc)
                announce(seed)
                checkpoint()
        except KeyboardInterrupt:
            interrupted = True
            mark_remaining_interrupted()
            checkpoint()
    else:
        module_name = _resolve_run_seed_module(run_seed)
        executor = ProcessPoolExecutor(max_workers=options.workers,
                                       mp_context=multiprocessing.get_context("spawn"))
        futures = {}
        try:
            futures = {seed: executor.submit(_run_seed_worker, dict(
                module_name=module_name, cfg=cfg, seed=seed,
                staging_path=str(staging_dir / f"seed_{seed}_arrays.npz"))) for seed in cfg.seeds}
            # Collect strictly in cfg.seeds order; completion order cannot leak out.
            for seed in cfg.seeds:
                try:
                    ingest(seed, futures[seed].result())
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    record_failure(seed, exc)
                announce(seed)
                checkpoint()
        except KeyboardInterrupt:
            interrupted = True
            for future in futures.values():
                future.cancel()
            mark_remaining_interrupted()
            checkpoint()
        finally:
            executor.shutdown(wait=not interrupted, cancel_futures=True)

    if interrupted:
        # Do not swallow Ctrl+C: save what completed, report honestly, re-raise.
        save_json(output / "manifest.json", {**manifest, "seed_details": details, "details": extra,
                                             "run_complete": False, "success": False})
        print(output, flush=True)
        raise KeyboardInterrupt

    try:
        plot(rows, details, output, cfg)
    except Exception as exc:
        manifest["plot_error"] = f"{type(exc).__name__}: {exc}"
    success = not manifest.get("plot_error") and all(d["status"] == "COMPLETED" for d in details)
    save_json(output / "manifest.json", {**manifest, "seed_details": details, "details": extra,
                                         "run_complete": True, "success": success})
    if success and staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    print(output, flush=True)
    return output, success


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .2})
    return plt


def save_plot(fig, output, name="plot"):
    fig.tight_layout()
    fig.savefig(output / (name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output / (name + ".pdf"), bbox_inches="tight")
