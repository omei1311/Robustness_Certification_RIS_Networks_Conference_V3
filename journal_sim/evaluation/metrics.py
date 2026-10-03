"""Energy in joules, rate in bit/s, runtime accounted separately."""
import numpy as np


def long_term_metrics(records, cfg):
    duration = len(records) * cfg.slot_duration
    bits = sum(r["rate"] * cfg.slot_duration for r in records)
    tx = sum(r["transmission_circuit_power"] * cfg.slot_duration for r in records)
    ris = sum(r["ris_static_state_power"] * cfg.slot_duration for r in records)
    switched = sum(r["switching_energy"] for r in records)
    controller = sum(r["controller_energy"] for r in records)
    total = tx + ris + switched + controller
    installs = [r["time_index"] for r in records if r.get("installed")]
    intervals = np.diff(installs + [len(records)]) * cfg.slot_duration if installs else []
    return dict(total_bits=bits, transmission_circuit_energy=tx, ris_static_state_energy=ris,
                ris_switching_energy=switched, controller_energy=controller, total_energy=total,
                long_term_ee=bits / total if total > 0 else None, duration=duration,
                qos_outage_rate=sum(not r["qos_hold"] for r in records) / len(records) if records else None,
                number_of_reconfigurations=sum(bool(r.get("reconfigured")) for r in records),
                number_of_initializations=sum(bool(r.get("installed")) and not bool(r.get("reconfigured")) for r in records),
                mean_reuse_interval=float(np.mean(intervals)) if len(intervals) else None,
                n_changed_elements=sum(r["n_changed_elements"] for r in records),
                n_changed_bits=sum(r["n_changed_bits"] for r in records),
                algorithm_calls=sum(r.get("algorithm_calls", 0) for r in records),
                fast_oracle_calls=sum(r.get("fast_oracle_calls", 0) for r in records),
                strict_oracle_calls=sum(r.get("strict_oracle_calls", 0) for r in records),
                primary_solver_calls=sum(r.get("primary_solver_calls", 0) for r in records),
                fallback_solver_calls=sum(r.get("fallback_solver_calls", 0) for r in records),
                solver_error_count=sum(r.get("solver_error_count", 0) for r in records),
                solver_inaccurate_count=sum(r.get("solver_inaccurate_count", 0) for r in records),
                runtime=sum(r["runtime"] for r in records),
                failed_design_steps=sum(r.get("design_status") == "NO_ELIGIBLE_CANDIDATE" for r in records),
                empty_configuration_steps=sum(r["configuration_id"] is None for r in records))
