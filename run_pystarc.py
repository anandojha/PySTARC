#!/usr/bin/env python3
"""
PySTARC command from PDB to calculate association rate constants.

Usage
-----
    python run_pystarc.py pystarc_input.xml
    # Generate a template input file:
    python run_pystarc.py --template

Requirements
------------
    conda activate PySTARC
    conda install -c conda-forge ambertools apbs -y
    pip install dist/pystarc-1.1.0-py3-none-any.whl
    pip install cupy-cuda12x    # GPU (NVIDIA only, optional)
"""

from pystarc.analysis.convergence import (
    analyse_convergence,
    print_convergence,
    save_convergence,
)
from pystarc.pipeline.input_parser import write_template
from pystarc.pipeline.input_parser import parse
from pystarc.pipeline.pipeline import run
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


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    if sys.argv[1] == "--template":
        out = sys.argv[2] if len(sys.argv) > 2 else "pystarc_input.xml"
        write_template(out)
        print(f"Edit {out} then run:  python run_pystarc.py {out}")
        sys.exit(0)
    xml_path = Path(sys.argv[1])
    if not xml_path.exists():
        print(f"ERROR: Input file not found: {xml_path}")
        print("Generate a template with:  python run_pystarc.py --template")
        sys.exit(1)
    cfg = parse(xml_path)
    # Set up log file in work_dir
    work_dir = Path(cfg.work_dir)
    if not work_dir.is_absolute():
        work_dir = xml_path.parent / work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
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
                f"  Max-steps      : {result.n_max_steps:,}",
                f"  P_rxn          : {result.reaction_probability:.6f}",
            ]
            # Get k_on from results.json if available
            rj_path = work_dir / "results.json"
            if rj_path.exists():
                rj = json.loads(rj_path.read_text())
                _footer.append(f"  k_on           : {rj['k_on']:.4e} M-1 s-1")
                _footer.append(
                    f"  95% CI         : [{rj['k_on_low']:.4e}, {rj['k_on_high']:.4e}]"
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
            _footer += [
                f"  BD wall time   : {result.elapsed_sec:.1f} s",
                f"  Total wall time: {_wall_elapsed:.1f} s",
                f"  BD steps/sec   : {result.steps_per_sec:,.0f}",
                f"  Finished       : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "=" * 64,
            ]
            for line in _footer:
                print(line)
            # Convergence analysis
            if cfg.convergence_check:
                _k_b = rj.get("k_b", 0.0) if rj_path.exists() else 0.0
                _conv = analyse_convergence(
                    n_reacted=result.n_reacted,
                    n_escaped=result.n_escaped,
                    k_b=_k_b,
                    tol=cfg.convergence_tol,
                    work_dir=str(work_dir),
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
