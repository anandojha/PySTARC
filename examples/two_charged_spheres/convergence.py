#!/usr/bin/env python3
"""
Convergence test: charged_spheres - 4 seeds vs exact analytical.
Run from examples/two_charged_spheres && python test_convergence.py
"""
from pystarc.pipeline.pipeline import run as run_pipeline
from pystarc.pipeline.input_parser import parse
from scipy.integrate import quad
import datetime
import math
import time
import sys
import os

class _Tee:
    """Write to both stdout and a log file."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file
    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
    def flush(self):
        self._stream.flush()
        self._log.flush()

# Set up log
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bd_sims")
os.makedirs(log_dir, exist_ok=True)
_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = os.path.join(log_dir, f"convergence_{_ts}.log")
_log_f = open(_log_path, "w")
_orig_stdout = sys.stdout
sys.stdout = _Tee(_orig_stdout, _log_f)

# Exact analytical
eps_s=78.0*0.000142; debye=7.828; D=0.43371; b=10.0; a=2.5; q_esc=20.0
V_fac=1.0*(-1.0)/(4*math.pi*eps_s); CONV=602.214e6
def V(r): return V_fac*math.exp(-r/debye)/r
def ig(s): return (math.exp(V(1/s))/D if s>0 else 1/D)
Ib,_=quad(ig,0,1/b,limit=500); kb=4*math.pi/Ib
Iq,_=quad(ig,0,1/q_esc,limit=500); kq=4*math.pi/Iq; rp=kb/kq
def Si(r): return math.exp(V(r))/(r**2*D)
Sbq,_=quad(Si,b,q_esc,limit=500); Saq,_=quad(Si,a,q_esc,limit=500)
Ps=Sbq/Saq; Pexact=Ps/(1-(1-Ps)*rp); kon_exact=CONV*kb*Pexact

print(f"Exact: P_rxn={Pexact:.6f}, k_on={kon_exact:.4e}")
print()

results = []
for seed in [11111111, 22222222, 33333333, 44444444]:
    cfg = parse("input.xml")
    cfg.seed = seed; cfg.n_trajectories = 10000
    print(f"\n{'='*60}")
    print(f"  Seed={seed}")
    print(f"{'='*60}")
    t0=time.time()
    result = run_pipeline(cfg)
    elapsed=time.time()-t0
    
    p=result.reaction_probability; nr=result.n_reacted; ne=result.n_escaped
    nd=nr+ne; se=math.sqrt(p*(1-p)/nd) if nd>0 and 0<p<1 else 0
    results.append({'seed':seed,'P':p,'se':se,'nr':nr,'ne':ne,'t':elapsed})
    print(f"  -> P_rxn={p:.4f}±{se:.4f} ({nr} reacted, {elapsed:.0f}s)")

print("\n" + "="*70)
print("  Summary: charged_spheres 4-seed test")
print("="*70)
print(f"  Exact P_rxn = {Pexact:.6f}")
print(f"  {'Seed':>12s}  {'P_rxn':>10s}  {'Error':>8s}  {'Reacted':>8s}")
print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}")

pv=[]
for r in results:
    err=abs(r['P']-Pexact)/Pexact*100
    print(f"  {r['seed']:12d}  {r['P']:10.4f}  {err:7.1f}%  {r['nr']:8d}")
    pv.append(r['P'])

pm=sum(pv)/len(pv)
ps=math.sqrt(sum((p-pm)**2 for p in pv)/(len(pv)-1))
print(f"\n  Mean = {pm:.4f} ± {ps/math.sqrt(len(pv)):.4f}")
print(f"  Exact = {Pexact:.4f}")
print(f"  Mean error = {abs(pm-Pexact)/Pexact*100:.2f}%")
if abs(pm-Pexact)/Pexact < 0.02:
    print(f"Pass: within 2% of exact")
else:
    print(f"Mean deviates from exact by >{abs(pm-Pexact)/Pexact*100:.0f}%")

sys.stdout = _orig_stdout
_log_f.close()
print(f"\n  Log saved -> {_log_path}")
