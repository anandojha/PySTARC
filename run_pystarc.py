#!/usr/bin/env python3
"""
PySTARC command to build simulation inputs and calculate association rate constants.

Usage
-----
    python run_pystarc.py config.xml
    python run_pystarc.py config.xml --n-gpus 4
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
import datetime
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
    n_gpus_override = None
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--setup":
            setup_mode = True
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
    return positional, setup_mode, n_gpus_override


def _run_multi_gpu(input_xml_path: Path, work_dir: Path, n_gpus: int):
    from pystarc.multi_GPU.multi_GPU_runs import split_multi_gpu

    rc = split_multi_gpu(str(input_xml_path), n_splits=n_gpus)
    if rc != 0:
        sys.exit(rc)
    visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
    runner = str(Path(__file__).resolve())
    procs = []
    for i in range(1, n_gpus + 1):
        shard_dir = work_dir / f"bd_{i}"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = visible[(i - 1) % len(visible)] if visible else str(i - 1)
        procs.append((i, subprocess.Popen([sys.executable, runner, "input.xml"], cwd=str(shard_dir), env=env)))
    failed = []
    for i, p in procs:
        code = p.wait()
        if code != 0:
            failed.append((i, code))
    if failed:
        sys.exit(f"ERROR: multi-GPU shard failure: {failed}")
    combiner = str(Path(__file__).resolve().parent / "pystarc" / "multi_GPU" / "combine_data.py")
    rc = subprocess.run([sys.executable, combiner, "--bd-sims", str(work_dir)]).returncode
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
    positional, setup_mode, n_gpus_override = _parse_cli(sys.argv[1:])
    if not positional:
        print(__doc__)
        sys.exit(1)
    xml_path = Path(positional[0])
    if not xml_path.exists():
        print(f"ERROR: Input file not found: {xml_path}")
        print("Generate a template with:  python run_pystarc.py --template")
        sys.exit(1)
    is_config_root = ET.parse(xml_path).getroot().tag == "config"
    cfg = parse(xml_path)
    if n_gpus_override is not None:
        cfg.n_gpus = n_gpus_override
    # Set up log file in work_dir
    work_dir = Path(cfg.work_dir)
    if not work_dir.is_absolute():
        work_dir = xml_path.resolve().parent / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    if setup_mode:
        print(f"  Setup complete -> {work_dir.resolve()}")
        return
    if cfg.gpu and cfg.chain is None and cfg.n_gpus > 1:
        split_src = (work_dir / "input.xml") if is_config_root else xml_path.resolve()
        _run_multi_gpu(split_src, work_dir.resolve(), cfg.n_gpus)
        return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = work_dir / f"pystarc_{timestamp}.log"
    # Detect GPU
    _gpu_name = "N/A"
    try:
        _r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if _r.returncode == 0:
            _gpu_name = _r.stdout.strip().split("\n")[0]
    except Exception:
        pass
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
                # Chain BD mode: invoke chain BD orchestrator and print
                # a minimal summary. Stage 2 will expand the chain output
                # writer to produce convergence/k_on metrics matching the
                # rigid-body footer below.
                from pystarc.pipeline.chain_pipeline import run_chain

                run_chain(cfg, input_xml_path=xml_path)
                _wall_elapsed = time.time() - _wall_t0
                _footer = [
                    "",
                    "=" * 64,
                    "  Chain BD run summary",
                    "=" * 64,
                    f"  Host           : {os.uname().nodename}",
                    f"  GPU            : {_gpu_name}",
                    f"  Trajectories   : {cfg.n_trajectories:,}",
                    f"  Total wall time: {_wall_elapsed:.1f} s",
                    f"  Output dir     : {(Path(cfg.work_dir) / 'bd_1').resolve()}",
                    f"  Finished       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "=" * 64,
                ]
                for line in _footer:
                    print(line)
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
                    _k_b = rj.get("k_b", 0.0) if rj_path.exists() else getattr(result, "k_db", 0.0)
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
