#!/bin/bash
# Run all 8 TTK kinase-inhibitor BD simulations.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYSTARC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$PYSTARC_ROOT/run_pystarc.py"
SYSTEMS=(
    2X9E
    3GFW
    3H9F
    5LJJ
    5N7V
    5N84
    5N93
    5NAD
)

echo "PySTARC TTK kinase-inhibitor simulations"
echo "  PySTARC root: $PYSTARC_ROOT"
echo "  Runner:       $RUNNER"
echo "  Systems:      ${#SYSTEMS[@]}"
echo ""
for system in "${SYSTEMS[@]}"; do
    echo "================================================================"
    echo "  $system"
    echo "================================================================"
    cd "$SCRIPT_DIR/$system"
    # Clean previous outputs (the source PDB is kept; setup reuses it)
    rm -rf bd_sims
    rm -f receptor.pdb receptor.pqr ligand.pdb ligand.pqr input.xml rxns.xml
    rm -f protein.prmtop protein.rst7 ligand.prmtop ligand.rst7
    echo "  Cleaned previous outputs"
    # Setup
    echo "  Running setup.py ..."
    python setup.py
    if [ $? -ne 0 ]; then
        echo "  Error: setup.py failed for $system"
        continue
    fi
    # Run BD
    echo "  Running BD simulation ..."
    python "$RUNNER" input.xml
    if [ $? -ne 0 ]; then
        echo "  Error: BD simulation failed for $system"
        continue
    fi
    echo "  Done: $system"
    echo ""
done
# Compare rates against experiment
echo "================================================================"
echo "  Comparing on-rates against experiment"
echo "================================================================"
cd "$SCRIPT_DIR"
python3 << 'PYEOF'
import json, math, os, datetime
SYSTEMS = ["2X9E", "3GFW", "3H9F", "5LJJ", "5N7V", "5N84", "5N93", "5NAD"]
INHIBITOR = {
    "2X9E": "NMS-P715", "3GFW": "Mps1-IN-1", "3H9F": "Mps1-IN-2",
    "5LJJ": "Reversine", "5N7V": "MPI-0479605", "5N84": "Mps-BAY2b",
    "5N93": "TC-Mps1-12", "5NAD": "BAY-1217389",
}
EXPT_KON = {
    "2X9E": 6.41e5, "3GFW": 3.79e5, "3H9F": 1.19e6, "5LJJ": 2.08e6,
    "5N7V": 1.96e6, "5N84": 2.60e6, "5N93": 2.16e7, "5NAD": 3.79e5,
}

def fmt(k, ke):
    if k <= 0: return "0"
    e = int(math.floor(math.log10(k)))
    m, me = k / 10**e, ke / 10**e
    if me >= 0.1:
        return f"({m:.1f} +/- {me:.1f})e{e}"
    elif me >= 0.01:
        return f"({m:.2f} +/- {me:.2f})e{e}"
    else:
        return f"({m:.1f} +/- {me:.1e})e{e}"

lines = []
def out(s=""):
    print(s)
    lines.append(s)
out(f"TTK kinase-inhibitor complexes: PySTARC k_on vs experiment")
out(f"Collected: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
out()
out(f"  {'System':<8s}  {'Inhibitor':<14s}  {'PySTARC k_on':>22s}  {'Experiment k_on':>18s}  {'Ratio':>7s}  {'P_rxn':>9s}  {'Time':>6s}")
out(f"  {'-'*8}  {'-'*14}  {'-'*22}  {'-'*18}  {'-'*7}  {'-'*9}  {'-'*6}")
pystarc_kons, expt_kons = [], []
for s in SYSTEMS:
    rfile = os.path.join(s, "bd_sims", "results.json")
    if not os.path.exists(rfile):
        out(f"  {s:<8s}  {INHIBITOR[s]:<14s}  {'No results':>22s}")
        continue
    with open(rfile) as f:
        r = json.load(f)
    k = r["k_on"]
    ke = (r["k_on_high"] - r["k_on_low"]) / 2.0
    p = r["P_rxn"]
    wt = r.get("wall_time_sec", 0)
    expt = EXPT_KON[s]
    ratio = k / expt if expt else 0
    flag = ""
    out(f"  {s:<8s}  {INHIBITOR[s]:<14s}  {fmt(k, ke):>22s}  {expt:18.2e}  {ratio:6.1f}x  {p:9.6f}  {wt:5.0f}s{flag}")
    if k > 0:
        pystarc_kons.append(k)
        expt_kons.append(expt)
out()
if len(pystarc_kons) >= 3:
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(pystarc_kons, expt_kons)
        out(f"  Spearman rho = {rho:.3f}  (p = {pval:.4f}, n = {len(pystarc_kons)})")
    except ImportError:
        out("  (scipy not available for rank correlation)")
with open("summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
PYEOF
echo ""
echo "  All 8 receptor-ligand simulations complete."
echo "  Summary saved -> summary.txt"
