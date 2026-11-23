#!/usr/bin/env python3
"""
PySTARC command to build simulation inputs and calculate association rate constants.

Usage
-----
    python run_pystarc.py config.xml
    python run_pystarc.py config.xml --n-gpus 4
    python run_pystarc.py config.xml --n-gpus 4 --resume
    python run_pystarc.py config.xml --setup
    # Generate a blank configuration file:
    python run_pystarc.py --template [config.xml]

Requirements
------------
    conda activate PySTARC
"""

from pystarc.analysis.convergence import (
    analyse_convergence,
    print_convergence,
    save_convergence,
)
from pystarc.pipeline.input_parser import write_template
from pystarc.pipeline.input_parser import parse
from pystarc.pipeline.pipeline import run
import xml.etree.ElementTree as ET
from pathlib import Path
import subprocess
import threading
import datetime
import signal
import json
import time
import math
import sys
import csv
import os


class _Tee:
    """Write to both stdout and a log file simultaneously."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._stream.flush()
        self._log_file.write(data)
        self._log_file.flush()

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    # Pass through any attribute the real stdout has
    def __getattr__(self, name):
        return getattr(self._stream, name)


def _parse_cli(argv):
    setup_mode = False
    resume_mode = False
    n_gpus_override = None
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--setup":
            setup_mode = True
            i += 1
        elif a == "--resume":
            resume_mode = True
            i += 1
        elif a == "--n-gpus":
            if i + 1 >= len(argv):
                sys.exit("ERROR: --n-gpus requires a positive integer")
            try:
                n_gpus_override = int(argv[i + 1])
            except ValueError:
                sys.exit(f"ERROR: --n-gpus must be an integer, got {argv[i + 1]!r}")
            if n_gpus_override < 1:
                sys.exit("ERROR: --n-gpus must be >= 1")
            i += 2
        elif a.startswith("--n-gpus="):
            val = a.split("=", 1)[1]
            try:
                n_gpus_override = int(val)
            except ValueError:
                sys.exit(f"ERROR: --n-gpus must be an integer, got {val!r}")
            if n_gpus_override < 1:
                sys.exit("ERROR: --n-gpus must be >= 1")
            i += 1
        else:
            positional.append(a)
            i += 1
    return positional, setup_mode, n_gpus_override, resume_mode


def _env_with_repo_pythonpath(base_env=None) -> dict:
    """Return a copy of base_env with the repository root prepended to PYTHONPATH."""
    repo_root = str(Path(__file__).resolve().parent)
    env = dict(os.environ if base_env is None else base_env)
    existing_pp = env.get("PYTHONPATH", "")
    if existing_pp:
        parts = existing_pp.split(os.pathsep)
        if repo_root not in parts:
            env["PYTHONPATH"] = repo_root + os.pathsep + existing_pp
    else:
        env["PYTHONPATH"] = repo_root
    return env


def _resolve_visible_gpus(n_gpus: int):
    """Return visible GPU identifiers from CUDA_VISIBLE_DEVICES, or None when unset."""
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return None
    raw = os.environ["CUDA_VISIBLE_DEVICES"]
    visible = [d.strip() for d in raw.split(",") if d.strip()]
    if not visible:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is set but empty; no GPUs are visible for "
            f"{n_gpus}-GPU execution."
        )
    if len(visible) < n_gpus:
        print(
            f"  Warning: CUDA_VISIBLE_DEVICES exposes {len(visible)} GPU(s) "
            f"({','.join(visible)}) for {n_gpus} shards; shards will wrap "
            "across visible GPUs.",
            flush=True,
        )
    return visible


def _shard_is_complete(shard_dir: Path, expected_n: int) -> bool:
    """Return True when results.json is valid with expected_n trajectories and trajectories.csv exists."""
    rj_path = shard_dir / "results.json"
    traj_path = shard_dir / "trajectories.csv"
    if not rj_path.is_file() or not traj_path.is_file():
        return False
    try:
        data = json.loads(rj_path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    if "n_trajectories" in data:
        got_n = data["n_trajectories"]
    elif isinstance(data.get("summary"), dict) and "n_trajectories" in data["summary"]:
        got_n = data["summary"]["n_trajectories"]
    else:
        return False
    try:
        return int(got_n) == int(expected_n)
    except (TypeError, ValueError):
        return False


def _terminate_children(procs, grace_sec: float = 5.0) -> None:
    """Send SIGTERM to all live child processes, wait up to grace_sec, SIGKILL stragglers, and reap all."""
    live = [(i, p) for i, p in procs if p.poll() is None]
    if live:
        for _, p in live:
            try:
                p.terminate()
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace_sec
        while time.monotonic() < deadline:
            if all(p.poll() is not None for _, p in live):
                break
            time.sleep(0.05)
        for _, p in live:
            if p.poll() is None:
                try:
                    p.kill()
                except ProcessLookupError:
                    pass
    for _, p in procs:
        try:
            p.wait(timeout=5.0)
        except Exception:
            pass


def _detect_gpu_name(use_cupy: bool = True) -> str:
    """Return the active GPU name and memory string, respecting CUDA_VISIBLE_DEVICES."""
    if use_cupy:
        try:
            import cupy as cp

            dev = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(dev.id)
            raw_name = props.get("name", b"")
            name = (
                raw_name.decode("utf-8", errors="replace")
                if isinstance(raw_name, (bytes, bytearray))
                else str(raw_name)
            ).strip()
            total_mb = int(props.get("totalGlobalMem", 0)) // (1024 * 1024)
            if name:
                return f"{name}, {total_mb} MiB" if total_mb > 0 else name
        except Exception:
            pass
    first_dev = None
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cvd = [
            d.strip()
            for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            if d.strip()
        ]
        if not cvd:
            return "N/A"
        first_dev = cvd[0]
    try:
        if first_dev is not None:
            r_id = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    first_dev,
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
            )
            if r_id.returncode == 0 and r_id.stdout.strip():
                return r_id.stdout.strip().splitlines()[0].strip()
        r_all = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if r_all.returncode == 0 and r_all.stdout.strip():
            lines = [
                ln.strip() for ln in r_all.stdout.strip().splitlines() if ln.strip()
            ]
            if first_dev is not None and first_dev.isdigit():
                idx = int(first_dev)
                if 0 <= idx < len(lines):
                    return lines[idx]
            if lines:
                return lines[0]
    except Exception:
        pass
    return "N/A"


def _run_multi_gpu(
    input_xml_path: Path, work_dir: Path, n_gpus: int, resume: bool = False
):
    from pystarc.multi_GPU.multi_GPU_runs import split_multi_gpu

    split_meta = None
    split_path = work_dir / "split.json"
    if resume and split_path.is_file():
        try:
            meta = json.loads(split_path.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            meta = None
        if (
            isinstance(meta, dict)
            and int(meta.get("n_splits", -1)) == n_gpus
            and isinstance(meta.get("shards"), list)
            and len(meta["shards"]) == n_gpus
            and all(
                isinstance(s, dict)
                and "dir" in s
                and (work_dir / s["dir"] / "input.xml").is_file()
                for s in meta["shards"]
            )
        ):
            split_meta = meta
            print(
                f"  [resume] reusing existing {n_gpus}-way split in {work_dir}",
                flush=True,
            )

    if split_meta is None:
        visible = _resolve_visible_gpus(n_gpus)
        rc = split_multi_gpu(str(input_xml_path), n_splits=n_gpus)
        if rc != 0:
            sys.exit(rc)
        if split_path.is_file():
            split_meta = json.loads(split_path.read_text())
        shards_spec = (
            split_meta["shards"]
            if isinstance(split_meta, dict) and "shards" in split_meta
            else [{"dir": f"bd_{i}"} for i in range(1, n_gpus + 1)]
        )
        to_run = [
            (idx, work_dir / s["dir"]) for idx, s in enumerate(shards_spec, start=1)
        ]
    else:
        to_run = []
        for idx, s in enumerate(split_meta["shards"], start=1):
            shard_dir = work_dir / s["dir"]
            expected_n = int(s.get("n_trajectories", -1))
            if expected_n >= 0 and _shard_is_complete(shard_dir, expected_n):
                print(
                    f"  [resume] skipping completed shard {s['dir']} "
                    f"({expected_n:,} trajectories)",
                    flush=True,
                )
                continue
            to_run.append((idx, shard_dir))
        visible = _resolve_visible_gpus(n_gpus) if to_run else None

    runner = str(Path(__file__).resolve())
    base_env = _env_with_repo_pythonpath()
    procs = []
    in_main_thread = threading.current_thread() is threading.main_thread()
    prev_sigterm = None
    prev_sigint = None
    if in_main_thread:

        def _on_signal(signum, _frame):
            raise SystemExit(128 + signum)

        prev_sigterm = signal.signal(signal.SIGTERM, _on_signal)
        prev_sigint = signal.signal(signal.SIGINT, _on_signal)

    try:
        for k, (i, shard_dir) in enumerate(to_run):
            env = dict(base_env)
            env["CUDA_VISIBLE_DEVICES"] = (
                visible[k % len(visible)] if visible is not None else str(k)
            )
            procs.append(
                (
                    i,
                    subprocess.Popen(
                        [sys.executable, runner, "input.xml"],
                        cwd=str(shard_dir),
                        env=env,
                    ),
                )
            )
        failed = []
        pending = list(procs)
        while pending:
            still_running = []
            for i, p in pending:
                code = p.poll()
                if code is None:
                    still_running.append((i, p))
                elif code != 0:
                    failed.append((i, code))
            if failed:
                print(
                    f"ERROR: shard(s) failed early: {failed}; terminating remaining shards.",
                    flush=True,
                )
                _terminate_children(procs, grace_sec=5.0)
                sys.exit(f"ERROR: multi-GPU shard failure: {failed}")
            pending = still_running
            if pending:
                time.sleep(0.1)
    finally:
        if in_main_thread:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            _terminate_children(procs, grace_sec=5.0)
        finally:
            if in_main_thread:
                signal.signal(signal.SIGTERM, prev_sigterm)
                signal.signal(signal.SIGINT, prev_sigint)

    combiner = str(
        Path(__file__).resolve().parent / "pystarc" / "multi_GPU" / "combine_data.py"
    )
    rc = subprocess.run(
        [sys.executable, combiner, "--bd-sims", str(work_dir)],
        env=base_env,
    ).returncode
    if rc != 0:
        sys.exit(rc)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    if sys.argv[1] == "--template":
        out = sys.argv[2] if len(sys.argv) > 2 else "config.xml"
        write_template(out)
        print(f"Edit {out} then run:  python run_pystarc.py {out}")
        sys.exit(0)
    positional, setup_mode, n_gpus_override, resume_mode = _parse_cli(sys.argv[1:])
    if not positional:
        print(__doc__)
        sys.exit(1)
    xml_path = Path(positional[0])
    if not xml_path.exists():
        print(f"ERROR: Input file not found: {xml_path}")
        print("Generate a template with:  python run_pystarc.py --template")
        sys.exit(1)
    is_config_root = ET.parse(xml_path).getroot().tag == "config"
    cfg = None
    resumed_work_dir = None
    if resume_mode and is_config_root:
        from pystarc.pipeline.build_inputs import read_config

        raw_cfg = read_config(xml_path)
        raw_bd = raw_cfg.get("bd_work_dir", "bd_sims")
        cand_dir = Path(str(raw_bd))
        if not cand_dir.is_absolute():
            cand_dir = xml_path.resolve().parent / cand_dir
        cand_xml = cand_dir / "input.xml"
        if cand_xml.is_file():
            print(
                f"  [resume] loading existing {cand_xml} without rebuilding inputs",
                flush=True,
            )
            cfg = parse(cand_xml)
            resumed_work_dir = cand_dir
            if raw_cfg.get("n_gpus") not in (None, ""):
                cfg.n_gpus = int(raw_cfg.get("n_gpus") or 1)
    if cfg is None:
        cfg = parse(xml_path)
    if n_gpus_override is not None:
        cfg.n_gpus = n_gpus_override
    # Resolve work_dir relative to the input XML and update cfg directly
    # for rigid and chain runs inside the shard directory.
    work_dir = Path(cfg.work_dir)
    if not work_dir.is_absolute():
        if resumed_work_dir is not None and work_dir == Path("."):
            work_dir = resumed_work_dir
        else:
            work_dir = xml_path.resolve().parent / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.work_dir = work_dir
    if setup_mode:
        print(f"  Setup complete -> {work_dir.resolve()}")
        return
    if cfg.gpu and cfg.chain is None and cfg.n_gpus > 1:
        split_src = (work_dir / "input.xml") if is_config_root else xml_path.resolve()
        _run_multi_gpu(split_src, work_dir.resolve(), cfg.n_gpus, resume=resume_mode)
        return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = work_dir / f"pystarc_{timestamp}.log"
    _gpu_name = _detect_gpu_name(use_cupy=bool(cfg.gpu and cfg.chain is None))
    with open(log_path, "w") as log_f:
        # Write header
        log_f.write("=" * 64 + "\n")
        log_f.write("  PySTARC run log\n")
        log_f.write("=" * 64 + "\n")
        log_f.write(
            f"  Started      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        log_f.write(f"  Input        : {xml_path.resolve()}\n")
        log_f.write(f"  Host         : {os.uname().nodename}\n")
        log_f.write(f"  Platform     : {os.uname().sysname} {os.uname().release}\n")
        log_f.write(f"  GPU          : {_gpu_name}\n")
        log_f.write(f"  Trajectories : {cfg.n_trajectories:,}\n")
        log_f.write(f"  Max steps    : {cfg.max_steps:,}\n")
        log_f.write(f"  Seed         : {cfg.seed}\n")
        log_f.write(f"  Work dir     : {work_dir.resolve()}\n")
        log_f.write("=" * 64 + "\n\n")
        log_f.flush()
        # Redirect stdout such that print() calls go to both terminal and log
        original_stdout = sys.stdout
        sys.stdout = _Tee(original_stdout, log_f)
        _wall_t0 = time.time()
        try:
            if cfg.chain is not None:
                from pystarc.pipeline.chain_pipeline import run_chain

                chain_xml = (work_dir / "input.xml") if is_config_root else xml_path
                out_dir = run_chain(cfg, input_xml_path=chain_xml)
                _wall_elapsed = time.time() - _wall_t0
                rj_path = Path(out_dir) / "results.json"
                rj = json.loads(rj_path.read_text()) if rj_path.exists() else {}
                _sum = rj.get("summary", {})
                _rate = rj.get("rate", {})
                _nr = int(_sum.get("n_reacted", 0))
                _ne = int(_sum.get("n_escaped", 0))
                _nm = int(_sum.get("n_max_steps", 0))
                _ntot = max(_nr + _ne + _nm, 1)
                _prxn = float(_rate.get("P_rxn", _nr / _ntot))
                _chain_gpu_line = (
                    "CPU on main (GPU flexible-chain lives on pystarc_flexible_gpu)"
                    if cfg.gpu
                    else "disabled (<gpu>false</gpu>)"
                )
                _footer = [
                    "",
                    "=" * 64,
                    "  Chain BD run summary",
                    "=" * 64,
                    f"  Host           : {os.uname().nodename}",
                    f"  GPU            : {_chain_gpu_line}",
                    f"  Flexible role  : {cfg.chain.flexible_partner}",
                    f"  Trajectories   : {cfg.n_trajectories:,}",
                    f"  Reacted        : {_nr:,}",
                    f"  Escaped        : {_ne:,}",
                    f"  Max-steps      : {_nm:,}  (counted as not reacted)",
                    f"  P_rxn          : {_prxn:.6f}",
                ]
                if _rate.get("k_on") is not None:
                    _kon = float(_rate["k_on"])
                    _klo = float(_rate["k_on_low"])
                    _khi = float(_rate["k_on_high"])
                    _ci_lvl = float(
                        _rate.get("confidence_level", cfg.confidence_interval)
                    )
                    _ci_lbl = f"{100.0 * _ci_lvl:g}% CI"
                    _footer.append(f"  k_on           : {_kon:.4e} M-1 s-1")
                    _footer.append(f"  {_ci_lbl:<15s}: [{_klo:.4e}, {_khi:.4e}]")
                    if _kon > 0.0:
                        _exp = int(math.floor(math.log10(_kon)))
                        _man = _kon / 10**_exp
                        _man_err = ((_khi - _klo) / 2.0) / 10**_exp
                        _footer.append(
                            f"  k_on (± error) : ({_man:.1f} ± {_man_err:.1f}) x 10^{_exp} M-1 s-1"
                        )
                    _footer.append(
                        f"  k_b            : {float(_rate['k_b']):.4f} A3/ps"
                    )
                _footer += [
                    f"  Total wall time: {_wall_elapsed:.1f} s",
                    f"  Output dir     : {Path(out_dir).resolve()}",
                    f"  Finished       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "=" * 64,
                ]
                for line in _footer:
                    print(line)
                if cfg.convergence_check and _rate.get("k_b") is not None:
                    _nam_omega = (
                        0.0
                        if _rate.get("use_lmz", False)
                        else float(_rate.get("nam_omega", 0.0) or 0.0)
                    )
                    _conv = analyse_convergence(
                        n_reacted=_nr,
                        n_escaped=_ne,
                        n_max_steps=_nm,
                        k_b=float(_rate["k_b"]),
                        tol=cfg.convergence_tol,
                        work_dir=str(out_dir),
                        confidence=cfg.confidence_interval,
                        nam_omega=_nam_omega,
                    )
                    print_convergence(_conv)
                    save_convergence(_conv, work_dir=str(out_dir))
            else:
                # Rigid-body mode (the original PySTARC pipeline).
                result = run(cfg)
                _wall_elapsed = time.time() - _wall_t0
                # Write summary footer
                _footer = [
                    "",
                    "=" * 64,
                    "  Run summary",
                    "=" * 64,
                    f"  Host           : {os.uname().nodename}",
                    f"  GPU            : {_gpu_name}",
                    f"  Trajectories   : {cfg.n_trajectories:,}",
                    f"  Reacted        : {result.n_reacted:,}",
                    f"  Escaped        : {result.n_escaped:,}",
                    f"  Max-steps      : {result.n_max_steps:,}  (counted as not reacted)",
                    f"  P_rxn          : {result.reaction_probability:.6f}",
                ]
                # Get k_on from results.json if available
                rj_path = work_dir / "results.json"
                if rj_path.exists():
                    rj = json.loads(rj_path.read_text())
                    _footer.append(f"  k_on           : {rj['k_on']:.4e} M-1 s-1")
                    _ci = f"{100 * rj.get('confidence_level', cfg.confidence_interval):g}% CI"
                    _footer.append(
                        f"  {_ci:<15s}: [{rj['k_on_low']:.4e}, {rj['k_on_high']:.4e}]"
                    )
                    # Formatted with error: (value ± error) × 10^n
                    _kon = rj["k_on"]
                    _kon_err = (rj["k_on_high"] - rj["k_on_low"]) / 2.0
                    if _kon > 0:
                        _exp = int(math.floor(math.log10(_kon)))
                        _man = _kon / 10**_exp
                        _man_err = _kon_err / 10**_exp
                        _footer.append(
                            f"  k_on (± error) : ({_man:.1f} ± {_man_err:.1f}) x 10^{_exp} M-1 s-1"
                        )
                    _footer.append(f"  k_b            : {rj['k_b']:.4f} A3/ps")
                # The CPU model result has no timing fields.
                if hasattr(result, "elapsed_sec"):
                    _footer.append(f"  BD wall time   : {result.elapsed_sec:.1f} s")
                _footer.append(f"  Total wall time: {_wall_elapsed:.1f} s")
                if hasattr(result, "steps_per_sec"):
                    _footer.append(f"  BD steps/sec   : {result.steps_per_sec:,.0f}")
                _footer += [
                    f"  Finished       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "=" * 64,
                ]
                for line in _footer:
                    print(line)
                # Convergence analysis
                if cfg.convergence_check:
                    _k_b = (
                        rj.get("k_b", 0.0)
                        if rj_path.exists()
                        else getattr(result, "k_db", 0.0)
                    )
                    _conv = analyse_convergence(
                        n_reacted=result.n_reacted,
                        n_escaped=result.n_escaped,
                        n_max_steps=result.n_max_steps,
                        k_b=_k_b,
                        tol=cfg.convergence_tol,
                        work_dir=str(work_dir),
                        confidence=cfg.confidence_interval,
                    )
                    print_convergence(_conv)
                    save_convergence(_conv, work_dir=str(work_dir))
        except Exception as e:
            sys.stdout._log_file.write(f"\nERROR: {e}\n")
            raise
        finally:
            sys.stdout = original_stdout
    print(f"\n  Log saved -> {log_path}")


if __name__ == "__main__":
    main()
