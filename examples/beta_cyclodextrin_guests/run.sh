#!/bin/bash
# Run all 7 beta-cyclodextrin host-guest BD simulations.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYSTARC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$PYSTARC_ROOT/run_pystarc.py"
SYSTEMS=(
    BCD_1-propanol
    BCD_1-butanol
    BCD_tertbutanol
    BCD_methyl_butyrate
    BCD_aspirin
    BCD_1-naphthylethanol
    BCD_2-naphthylethanol
)

echo "PySTARC beta-cyclodextrin host-guest simulations"
echo "  PySTARC root: $PYSTARC_ROOT"
echo "  Runner:       $RUNNER"
echo "  Systems:      ${#SYSTEMS[@]}"
echo ""
for system in "${SYSTEMS[@]}"; do
    echo "================================================================"
    echo "  $system"
    echo "================================================================"
    cd "$SCRIPT_DIR/$system"
    # Clean previous outputs
    rm -rf bd_sims
    rm -f receptor.pqr ligand.pqr input.xml rxns.xml _full.rst7 _grid_gen.xml
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
SYSTEMS = [
    "BCD_1-propanol", "BCD_1-butanol", "BCD_tertbutanol",
    "BCD_methyl_butyrate", "BCD_aspirin",
    "BCD_1-naphthylethanol", "BCD_2-naphthylethanol",
]
GUEST = {
    "BCD_1-butanol": "1-butanol", "BCD_1-propanol": "1-propanol",
    "BCD_tertbutanol": "tert-butanol", "BCD_methyl_butyrate": "methyl butyrate",
    "BCD_aspirin": "aspirin", "BCD_1-naphthylethanol": "1-naphthylethanol",
    "BCD_2-naphthylethanol": "2-naphthylethanol",
}
EXPT_KON = {
    "BCD_1-butanol": 2.80e8, "BCD_1-propanol": 5.10e8,
    "BCD_tertbutanol": 3.60e8, "BCD_methyl_butyrate": 3.70e8,
    "BCD_aspirin": 7.21e8, "BCD_1-naphthylethanol": 4.70e8,
    "BCD_2-naphthylethanol": 2.90e8,
}
EXPT_ERR = {
    "BCD_1-butanol": 8.00e7, "BCD_1-propanol": 7.00e7,
    "BCD_tertbutanol": 1.00e7, "BCD_methyl_butyrate": 3.00e7,
    "BCD_aspirin": 4.00e6, "BCD_1-naphthylethanol": 1.90e8,
    "BCD_2-naphthylethanol": 1.60e8,
}

def fmt(k, ke):
    if k <= 0: return "0"
    e = int(math.floor(math.log10(k)))
    m, me = k / 10**e, ke / 10**e
    if me >= 0.1:
        return f"({m:.1f} ± {me:.1f})e{e}"
    elif me >= 0.01:
        return f"({m:.2f} ± {me:.2f})e{e}"
    else:
        return f"({m:.1f} ± {me:.1e})e{e}"

lines = []
def out(s=""):
    print(s)
    lines.append(s)
out(f"Beta-cyclodextrin host-guest complexes: PySTARC k_on vs experiment")
out(f"Collected: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
out()
out(f"  {'Ligand':<22s}  {'PySTARC k_on':>20s}  {'Experiment k_on':>20s}  {'Ratio':>7s}  {'P_rxn':>7s}  {'Time':>5s}")
out(f"  {'-'*22}  {'-'*20}  {'-'*20}  {'-'*7}  {'-'*7}  {'-'*5}")
pystarc_kons, expt_kons = [], []
for s in SYSTEMS:
    rfile = os.path.join(s, "bd_sims", "results.json")
    if not os.path.exists(rfile):
        out(f"  {GUEST[s]:<22s}  {'No results':>20s}")
        continue
    with open(rfile) as f:
        r = json.load(f)
    k = r["k_on"]
    ke = (r["k_on_high"] - r["k_on_low"]) / 2.0
    p = r["P_rxn"]
    wt = r.get("wall_time_sec", 0)
    expt = EXPT_KON[s]
    expt_e = EXPT_ERR[s]
    ratio = k / expt if expt else 0
    out(f"  {GUEST[s]:<22s}  {fmt(k, ke):>20s}  {fmt(expt, expt_e):>20s}  {ratio:6.1f}x  {p:7.4f}  {wt:4.0f}s")
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
echo "  All 7 receptor-ligand simulations complete."
echo "  Summary saved -> summary.txt"
