"""Run manifests, checkpointed raw records and immutable run directories."""
import argparse
import csv
from dataclasses import fields
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
import cvxpy
import scipy
from journal_sim.config import JournalConfig, smoke_config, config_fingerprint

ROOT = Path(__file__).resolve().parents[2]


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
    tmp.replace(path)


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
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


def load_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "config" in payload:
        payload = payload["config"]
    def tuples(v):
        return tuple(tuples(x) for x in v) if isinstance(v, list) else v
    defaults = JournalConfig()
    return JournalConfig(**{k: tuples(v) if isinstance(getattr(defaults, k), tuple) else v for k, v in payload.items()}).validate()


def cli_config(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=str)
    parser.add_argument("--n-seeds", type=int)
    parser.add_argument("--time-steps", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--direct-intercell", choices=("on", "off"))
    parser.add_argument("--mobility", choices=("slow", "medium", "fast"))
    parser.add_argument("--selection", choices=("lifetime_aware", "wee_only", "robustness_only", "stability_aware"))
    parser.add_argument("--output-root", type=str)
    args = parser.parse_args()
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
    return cfg.with_overrides(**overrides)


def execute(experiment, cfg, run_seed, plot):
    timestamp = datetime.now(timezone(timedelta(hours=8)))
    fingerprint = config_fingerprint(cfg)
    mode = "smoke" if cfg.smoke else "formal"
    output = Path(cfg.output_root)
    if not output.is_absolute():
        output = ROOT / output
    output = output / experiment / mode / (timestamp.strftime("%Y%m%dT%H%M%S%f") + "_" + fingerprint[:12])
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(experiment=experiment, mode=mode, config=cfg.to_dict(),
                    seeds=cfg.seeds, fingerprint=fingerprint, timestamp=timestamp.isoformat(),
                    git_commit=commit_id(), source_hashes=source_hashes(),
                    versions=dict(numpy=np.__version__, scipy=scipy.__version__, cvxpy=cvxpy.__version__),
                    note="Smoke results verify execution only. All requested seeds and failed/uncertain outcomes remain in raw records.")
    rows, details, extra, arrays = [], [], [], {}
    save_json(output / "manifest.json", {**manifest, "seed_details": [], "details": [], "run_complete": False})
    for seed in cfg.seeds:
        try:
            result = run_seed(cfg, seed)
            seed_rows = result["records"]
            seed_status = result.get("status", "COMPLETED")
            rows.extend(dict(seed=seed, **{k: v for k, v in row.items() if k != "seed"}) for row in seed_rows)
            details.append(dict(seed=seed, status=seed_status, summary=result.get("summary", result.get("summaries"))))
            extra.extend(result.get("details", result.get("designs", [])))
            for key, value in result.get("arrays", {}).items():
                arrays[f"seed{seed}_{key}"] = value
        except Exception as exc:
            failure = dict(seed=seed, status="FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            rows.append(failure)
            details.append(failure)
        save_csv(output / "records.csv", rows)
        save_json(output / "manifest.json", {**manifest, "seed_details": details, "details": extra,
                                             "run_complete": len(details) == len(cfg.seeds)})
        np.savez_compressed(output / "arrays.npz", **arrays)
        print(f"[{experiment}] seed={seed} status={details[-1]['status']}", flush=True)
    try:
        plot(rows, details, output, cfg)
    except Exception as exc:
        manifest["plot_error"] = f"{type(exc).__name__}: {exc}"
    success = not manifest.get("plot_error") and all(d["status"] == "COMPLETED" for d in details)
    save_json(output / "manifest.json", {**manifest, "seed_details": details, "details": extra,
                                         "run_complete": True, "success": success})
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


def save_plot(fig, output):
    fig.tight_layout()
    fig.savefig(output / "plot.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / "plot.pdf", bbox_inches="tight")
