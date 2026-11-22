# PySTARC Tutorial

This tutorial computes the bimolecular association rate constant $k_{\mathrm{on}}$ of trypsin and benzamidine via rigid-body Brownian dynamics on an NVIDIA GPU. Input files are located in `examples/trypsin_benzamidine`. Sections 3 through 11 document every tag and accepted value across the nine sections of `config.xml`. Every benchmark complex shares this `config.xml` schema, with unused tags blank as `<tag></tag>`. Section 15 describes the CPU flexible-chain workflow in `examples/barnase_barstar`.

## 1. Installation

PySTARC rigid-body GPU execution requires Linux with an NVIDIA GPU and a driver supporting CUDA 12 or newer.

```bash
git clone https://github.com/anandojha/PySTARC.git
cd PySTARC
bash install.sh
conda activate PySTARC
```

`install.sh` builds the `PySTARC` conda environment with AmberTools 24.8, APBS, RDKit, OpenBabel, and CuPy. Any existing `PySTARC` environment is replaced. When no NVIDIA GPU is present or the driver predates CUDA 12, `install.sh` exits without modifying the system.

## 2. Working Directory Setup

```bash
mkdir examples/my_trypsin
cd examples/my_trypsin
cp ../trypsin_benzamidine/{receptor.pqr,ligand.pqr,run.sh,submit.sh} .
python ../../run_pystarc.py --template
```

`run_pystarc.py --template` writes a blank `config.xml` containing the nine sections `structure`, `criterion`, `geometry`, `electrostatics`, `physics`, `simulation`, `compute`, `outputs`, and `chain`. `run.sh` and `submit.sh` locate the repository root from parent directories or from the installed `pystarc` package. Export `PYSTARC_ROOT=/path/to/PySTARC` to target a different checkout.

| File | Role |
|---|---|
| `receptor.pqr` | Trypsin atomic coordinates, partial charges, and radii in the bound pose |
| `ligand.pqr` | Benzamidine atomic coordinates, partial charges, and radii in the bound pose |
| `config.xml` | Input preparation and Brownian dynamics configuration |
| `run.sh` | Single-GPU execution script invoking `run_pystarc.py config.xml` |
| `submit.sh` | SLURM batch script defaulting to 1 node and 4 GPUs, scalable via `sbatch` flags |

## 3. Receptor and Ligand Structures

The `<structure>` section specifies the receptor and ligand structure files in matching formats, the formal ligand charge, and the solute representation.

| Tag | Accepted values | Function |
|---|---|---|
| `receptor` | `.pdb` or `.pqr` path | Prepared receptor structure. `.pdb` inputs are parameterized with `tleap` and `ff14SB`, whereas `.pqr` inputs supply atomic charges and radii directly. |
| `ligand` | `.pdb` or `.pqr` path matching `receptor` | Prepared ligand structure. Small-molecule `.pdb` inputs are parameterized with `antechamber` under AM1-BCC and GAFF2, protein `.pdb` inputs are parameterized with `tleap` for `rigid` or mapped to COFFDROP beads for `chain`, and `.pqr` inputs supply atomic charges and radii directly. |
| `ligand_net_charge` | Integer | Formal net charge of the ligand in elementary charge units. Passed to `antechamber` for small-molecule `.pdb` ligands and checked against total atomic or bead charge for `.pqr` and protein `.pdb` ligands. |
| `model` | `rigid` or `chain` | Rigid-body Brownian dynamics or flexible COFFDROP bead chain |

Mixing `.pdb` and `.pqr` files in one `config.xml` terminates the build with an error. The trypsin-benzamidine benchmark supplies PQR files matching the atomic charges and radii of the BrownDye2 reference.

```xml
<structure>
  <receptor>receptor.pqr</receptor>
  <ligand>ligand.pqr</ligand>
  <ligand_net_charge>1</ligand_net_charge>
  <model>rigid</model>
</structure>
```

`run_pystarc.py` copies `receptor.pqr` and `ligand.pqr` into `bd_sims/receptor.pqr` and `bd_sims/ligand.pqr`, stripping any prior ghost atoms.

### PDB Input Preparation

A new protein-ligand complex starts from prepared `receptor.pdb` and `ligand.pdb` files. To split a raw RCSB PDB complex, assign protonation states, and print `<structure>` and `<criterion>` blocks, run `scripts/prepare_pdb.py` and `scripts/find_reaction_criteria.py` as documented in [`scripts/README.md`](scripts/README.md). For example, `examples/hsp90_inhibitors/HSP90-aminopyridine` configures `<structure>` as below.

```xml
<structure>
  <receptor>receptor.pdb</receptor>
  <ligand>ligand.pdb</ligand>
  <ligand_net_charge>0</ligand_net_charge>
  <model>rigid</model>
</structure>
```

`run_pystarc.py` parameterizes `receptor.pdb` with `tleap` and `ff14SB`. RDKit assigns ligand bond orders from coordinates, explicit hydrogens, and `ligand_net_charge` in `ligand.pdb`, and `antechamber` assigns AM1-BCC partial charges and GAFF2 atom types.

| File | Requirements |
|---|---|
| `receptor.pdb` | Single-conformer `ATOM` records with AMBER residue names (`HID`, `HIE`, or `HIP` for histidine, `CYX` for disulfides, plain `HIS` mapped to `HIE`), with catalytic divalent metals `ZN`, `MG`, `MN`, `FE`, `CU`, `CO`, `NI`, and `CAL` retained after `TER` |
| `ligand.pdb` | Single-conformer `HETATM` or `ATOM` records with explicit hydrogens matching `ligand_net_charge`, or a single protein chain for `model` `chain` |

## 4. Reaction Criterion

The `<criterion>` section defines the encounter state. For rigid-body runs, the build stage writes `bd_sims/rxns.xml` containing receptor-ligand atom pairs and a common reaction distance `pair_cutoff`. A trajectory reacts when at least `n_needed` pairs simultaneously lie below `pair_cutoff`.

| `mode` | Pair definition |
|---|---|
| `contacts` | Explicit receptor and ligand atom pairs specified in `<receptor>` and `<ligand>` |
| `centroid` | Single pair connecting the receptor geometric centroid and the ligand geometric centroid |

### Contact Pairs

In `contacts` mode, the trypsin-benzamidine benchmark specifies six atom pairs with `pair_cutoff` 7.0 Å and `n_needed` 3.

```xml
<criterion>
  <mode>contacts</mode>
  <receptor>
    <target resid="193" atom="O" resname="TRP"/>
    <target resid="192" atom="O" resname="SER"/>
    <target resid="173" atom="O" resname="CYX"/>
    <target resid="205" atom="O" resname="VAL"/>
    <target resid="174" atom="N" resname="GLN"/>
    <target resid="194" atom="N" resname="GLY"/>
  </receptor>
  <ligand>
    <atom name="N2"/>
    <atom name="N2"/>
    <atom name="N1"/>
    <atom name="N2"/>
    <atom name="N1"/>
    <atom name="N2"/>
  </ligand>
  <pair_cutoff>7.0</pair_cutoff>
  <n_needed>3</n_needed>
</criterion>
```

| Receptor atom in `receptor.pqr` | Ligand atom | Bound-pose distance (Å) |
|---|---|---|
| TRP 193 O | N2 | 2.82 |
| SER 192 O | N2 | 2.86 |
| CYX 173 O | N1 | 3.17 |
| VAL 205 O | N2 | 3.34 |
| GLN 174 N | N1 | 4.24 |
| GLY 194 N | N2 | 4.31 |

Each `<target resid="..." atom="..." resname="..."/>` element identifies one receptor atom by residue number, atom name, and residue name. For `.pdb` receptors, residue indices in `receptor.pdb`, including retained catalytic metals such as `ZN`, map automatically to sequential residue indices in `bd_sims/receptor.pqr`. Protonation variants assigned by `tleap` (`HIS`/`HID`/`HIE`/`HIP`, `CYS`/`CYX`/`CYM`, `ASP`/`ASH`, `GLU`/`GLH`, `LYS`/`LYN`) match across group members. Each entry in `<ligand>` corresponds in order to `<receptor>` as `<atom name="..."/>` for single-residue ligands, or `<target resid="..." atom="..." resname="..."/>` for multi-residue protein ligands such as `examples/thrombin_thrombomodulin`.

For rigid-body runs, a single pair requires `n_needed` 1, and an even pair count $2k$ requires `n_needed` $k$.

### Centroid Distance

```xml
<criterion>
  <mode>centroid</mode>
  <receptor></receptor>
  <ligand></ligand>
  <pair_cutoff>8.0</pair_cutoff>
  <n_needed>0</n_needed>
</criterion>
```

In `centroid` mode, the simulation appends a zero-charge, zero-radius ghost atom `GHO` at each molecular centroid. Ghost records are stripped prior to APBS execution to prevent dielectric cavity artifacts in the electrostatic and Born grids. Reaction occurs when the two centroids lie within `pair_cutoff`. `config.xml` specifies `n_needed` 0, and `rxns.xml` writes `1`. Both the $\beta$-cyclodextrin and two-charged-spheres benchmarks select `centroid` mode.

### Criterion Tags

| Tag | Applicable mode | Function |
|---|---|---|
| `mode` | `contacts` or `centroid` | Reaction criterion type |
| `receptor` | `contacts`, blank in `centroid` | Receptor atoms as `<target resid="..." atom="..." resname="..."/>` |
| `ligand` | `contacts`, blank in `centroid` | Ligand atoms as `<atom name="..."/>` or `<target resid="..." atom="..." resname="..."/>` |
| `pair_cutoff` | `contacts` and `centroid` | Reaction cutoff distance per pair (Å) |
| `n_needed` | `contacts` and `centroid` | Simultaneous pairs required inside `pair_cutoff`, set to `0` in `centroid` mode |

### Default Reaction Criterion for a New Complex

For a complex lacking a measured $k_{\mathrm{on}}$, run `scripts/find_reaction_criteria.py` on prepared structures.

```bash
python scripts/find_reaction_criteria.py --receptor receptor.pdb --ligand ligand.pdb --diagnose --suggest-rim
```

- **Symmetric spheres or macrocyclic cavities with `N_rec_heavy < 120`.** `centroid` mode, with `2.5 Å` for analytical charged spheres and `8.0 Å` for $\beta$-cyclodextrin hosts.
- **Solvent-accessible sites with `f_shielded < 0.70` or `N_lig_heavy <= 15` outside a deep-pocket series.** 3-of-6 shortest bound polar contacts among N, O, and S atoms, one per receptor residue, at `7.0 Å`.
- **Deep binding sites with `f_shielded >= 0.70` and `N_lig_heavy > 15`, or series without a bound pose.** 3-of-6 shared solvent-exposed receptor contacts (`Cβ` or exposed sidechain atoms) at the smallest cutoff passing `--check-accessibility` across the series.

Full script options are documented in [`scripts/README.md`](scripts/README.md). Running `python scripts/verify_criteria.py` verifies all 32 rigid-body preparation scripts in `scripts/` against `examples/*/config.xml`.

## 5. Start Sphere and Hydrodynamic Radii

The `<geometry>` section sets the Northrup-Allison-McCammon spherical boundaries and molecular hydrodynamic radii. Every tag in Sections 5 through 10 is required in `config.xml` except the optional tags `r_escape` defaulting to `0`, `debye_length` derived from `ion_concentration` when blank in rigid mode and `7.858 Å` in chain mode, `soft_repulsion` defaulting to `false` in rigid mode and `true` in chain mode, `minimum_core_reaction_dt` defaulting to `0`, `n_gpus` defaulting to `1`, `work_dir` defaulting to `bd_sims`, `full_paths` defaulting to `true`, and `energetics` defaulting to `true`. Rigid-body runs hold every `<chain>` tag blank. Missing required tags halt the build before writing files.

```xml
<geometry>
  <bd_milestone_radius>45.0</bd_milestone_radius>
  <r_escape>0</r_escape>
  <r_hydro_rec>22.3728</r_hydro_rec>
  <r_hydro_lig>4.9867</r_hydro_lig>
</geometry>
```

Each trajectory starts with the ligand centroid at a uniform random point on the $b$-sphere of radius $b$ = `bd_milestone_radius` (Å) centered on the receptor centroid, with a uniform random orientation. `r_escape` sets the escape radius $q$ (Å), where `0` or blank sets $q = 2b$ and any explicit radius must exceed $1.1b$. Trajectories reaching $q$ return to $b$ with the analytical return probability or terminate as escaped.

`r_hydro_rec` and `r_hydro_lig` specify the receptor and ligand hydrodynamic radii $a$ (Å), determining the translational and rotational diffusion coefficients $D_t = k_B T / (6\pi\eta a)$ and $D_r = k_B T / (8\pi\eta a^3)$ at water viscosity $\eta = 0.890\text{ mPa}\cdot\text{s}$ at 25 °C. Setting either radius to `0` triggers the internal Monte Carlo calculation of Hansen (J. Chem. Phys. 121, 9111, 2004) in `pystarc/hydrodynamics/mc_hydro_radius.py` on the PQR coordinates with atomic radii inflated by `srad`, storing a `.cache` file beside each PQR in `bd_sims`. A positive entry overrides the Monte Carlo radius, and the run log prints both values. The trypsin-benzamidine benchmark fixes `22.3728 Å` and `4.9867 Å` to match BrownDye2, whereas setting `0` computes `22.30 Å` and `4.90 Å` from the same PQR files. Intermolecular hydrodynamic coupling is controlled separately by `hydrodynamic_interactions` in Section 7.

## 6. Electrostatics and APBS Grids

The `<electrostatics>` section sets solvent ionic conditions, dielectric constants, and nested APBS grid parameters.

```xml
<electrostatics>
  <debye_length>7.85751</debye_length>
  <ion_concentration>0.15</ion_concentration>
  <ion_radius_pos>0.95</ion_radius_pos>
  <ion_radius_neg>1.81</ion_radius_neg>
  <pdie>4.0</pdie>
  <sdie>78.0</sdie>
  <srad>1.4</srad>
  <apbs_fglen>96</apbs_fglen>
  <apbs_dime>257</apbs_dime>
  <apbs_cglen>0</apbs_cglen>
  <apbs_coarse_dime>0</apbs_coarse_dime>
  <apbs_fine_dime>0</apbs_fine_dime>
</electrostatics>
```

| Tag | Function |
|---|---|
| `debye_length` | Debye screening length $\lambda$ (Å) for exterior electrostatics, the $k_b$ integral, and Born desolvation grids. Set `inf` for zero salt, or blank to derive $\lambda$ from `ion_concentration` in rigid mode (defaults to `7.858 Å` in chain mode). |
| `ion_concentration` | Monovalent salt molarity (M) in APBS, or `0` for zero salt |
| `ion_radius_pos`, `ion_radius_neg` | Cation and anion exclusion radii in APBS (Å) |
| `pdie` | Solute relative permittivity |
| `sdie` | Solvent relative permittivity |
| `srad` | Solvent probe radius for the APBS dielectric surface and hydrodynamic radius calculation (Å) |

Execution terminates when an explicit `debye_length` differs by more than 1% from the value determined by `ion_concentration`, `sdie`, and `temperature` (7.83 Å at 0.15 M, $\varepsilon_s = 78$, and 298.15 K).

### Nested Electrostatic Grids

APBS solves the linearized Poisson-Boltzmann equation on two nested cubic grids centered on each solute, operating on a temporary ghost-free PQR copy with `GHO` records stripped.

| Grid | File in `bd_sims` | Edge length (Å) | Points per axis | Boundary condition |
|---|---|---|---|---|
| Coarse | `receptor0.dx` | `apbs_cglen`, with `0` setting twice the fine edge | `apbs_coarse_dime`, with `0` copying `apbs_dime` | Single-sphere Debye-Hückel (`bcfl sdh`) |
| Fine | `receptor1.dx` | `apbs_fglen`, with `0` deriving edge length | `apbs_fine_dime`, with `0` copying `apbs_dime` | Coarse-grid focusing (`bcfl map`) |

The derived fine edge length is $\max[(L + 2(r_{\max} + \mathrm{srad})) \times 1.05,\ 4(r_{\max} + \mathrm{srad}) \times 1.025]$, where $L$ is the maximum Cartesian span of atomic centers and $r_{\max}$ is the maximum atomic radius. `apbs_cglen` must exceed `apbs_fglen`. `apbs_dime` also controls the resolution of the Born desolvation grids.

During propagation, atomic forces are interpolated from the finest grid containing the atom at least three grid spacings inside every face. Beyond both grids, the receptor field switches to a screened multipole expansion controlled by `multipole_fallback` in Section 7. For trypsin, the fine grid spans `96 Å` with `257` points (`0.375 Å` spacing) and the coarse grid spans `192 Å` (`0.75 Å` spacing). The build prints a warning when $b + l_{\max} \ge \mathrm{apbs\_fglen}/2$. Trypsin triggers this warning (`45 Å + 3.7 Å > 48 Å`), and ligand atoms outside the fine grid interpolate from the coarse grid.

### Multigrid Dimensions

APBS multigrid solvers require $d = c \cdot 2^{l+1} + 1$ grid points per axis with integers $c \ge 1$ and $l \ge 1$. Valid counts are $4k + 1$ up to `65` (`5, 9, 13, 33, 65`) and $32c + 1$ above `65` (`97, 129, 161, 193, 225, 257, 449`). PySTARC validates `apbs_dime`, `apbs_coarse_dime`, and `apbs_fine_dime` before invoking APBS and exits on invalid values without rounding.

## 7. Forces and Excluded Volume

The `<physics>` section configures excluded volume, hydrodynamics, Born desolvation, and exterior electrostatics.

```xml
<physics>
  <soft_repulsion>true</soft_repulsion>
  <overlap_check>false</overlap_check>
  <lj_forces>false</lj_forces>
  <hydrodynamic_interactions>true</hydrodynamic_interactions>
  <desolvation_alpha>1.0</desolvation_alpha>
  <enable_born2_torque>true</enable_born2_torque>
  <multipole_fallback>true</multipole_fallback>
</physics>
```

Boolean tags accept `true`, `false`, `yes`, `no`, `1`, and `0` regardless of case. Sections 7 and 8 describe the GPU engine. Differences in the rigid-body CPU engine are summarized in Section 9.

| Tag | Accepted values | Function |
|---|---|---|
| `soft_repulsion` | Boolean, default `false` in `rigid` and `true` in `chain` | Soft steric repulsion replacing hard overlap rejection in `rigid`, or bead WCA repulsion in `chain` |
| `overlap_check` | Boolean | Rejection of moves overlapping the receptor |
| `lj_forces` | Boolean | Weeks-Chandler-Andersen steric repulsion |
| `hydrodynamic_interactions` | Boolean | Rotne-Prager-Yamakawa $D(r)$ or constant relative diffusion |
| `desolvation_alpha` | Float $\ge 0$ | Prefactor multiplying Born desolvation energy |
| `enable_born2_torque` | Boolean | Desolvation force and torque on the ligand from receptor charges in the ligand Born field |
| `multipole_fallback` | Boolean | Screened multipole field outside the receptor electrostatic grids |

### Excluded Volume Models

The rigid-body soft wall applies $V = k_B T (a/r)^{12}$ with $a = 0.9345 (r_i + r_j)$ to receptor-ligand atom pairs separated by less than $2a$, reading radii from PQR files and omitting zero-radius atoms. The hard overlap check rejects steps bringing any receptor-ligand pair closer than $0.5 (r_i + r_j)$, retaining the prior configuration while advancing simulation time. Weeks-Chandler-Andersen repulsion evaluates $4\varepsilon [(\sigma/r)^{12} - (\sigma/r)^6]$ for $r < 2^{1/6}\sigma$ with $\sigma = r_i + r_j$ and $\varepsilon = 0.1\text{ kcal/mol}$. Both the soft wall and the overlap check decrease the adaptive time step near the receptor surface.

| `soft_repulsion` | `overlap_check` | `lj_forces` | Excluded-volume behavior |
|---|---|---|---|
| `true` | `true` or `false` | `false` | Soft wall |
| `true` | `true` or `false` | `true` | Validation error before propagation |
| `false` | `true` | `false` | Hard overlap rejection |
| `false` | `true` | `true` | Hard overlap rejection with WCA repulsion |
| `false` | `false` | `false` | No steric exclusion |
| `false` | `false` | `true` | Validation error before propagation |

### Hydrodynamics, Desolvation, and Far-Field Electrostatics

When `hydrodynamic_interactions` is `true`, relative translational diffusion $D(r)$ obeys the Rotne-Prager-Yamakawa tensor for spheres of the Section 5 radii, adding the divergence drift $(dD/dr)\Delta t$ at each step and weighting the $k_b$ integral by $D(r)$. When `false`, relative diffusion remains constant at $D_{\mathrm{rec}} + D_{\mathrm{lig}}$.

Born desolvation contributes $\alpha q^2 \phi_{\mathrm{Born}}$ for charge $q$ in the partner Born field $\phi_{\mathrm{Born}}$ with $\alpha$ = `desolvation_alpha`. Setting `1.0` selects the reference desolvation scale, `0` disables desolvation, and negative values fail validation. For $\alpha > 0$, ligand charges feel the receptor Born field. Setting `enable_born2_torque` to `true` adds the reciprocal contribution of receptor charges in the ligand Born field as a net force and torque on the ligand in both GPU and CPU rigid-body engines.

When `multipole_fallback` is `true`, receptor electrostatics outside the grids evaluate a screened monopole, dipole, and quadrupole expansion, and the $k_b$ integral includes the orientation-averaged squared dipole and quadrupole terms for charged ligands. When `false`, the exterior potential retains the net receptor monopole term alone, with zero exterior field for neutral receptors.

The trypsin-benzamidine benchmark turns on soft repulsion, hydrodynamic interactions, bidirectional Born desolvation, and multipole far-field electrostatics. Setting `overlap_check` to `false` has no effect while `soft_repulsion` is `true`.

## 8. Trajectory Propagation and Convergence

The `<simulation>` section sets trajectory count, step cap, adaptive time-step bounds, temperature, seed, convergence reporting, and checkpointing.

```xml
<simulation>
  <n_trajectories>20000000</n_trajectories>
  <max_steps>10000000</max_steps>
  <dt>0.2</dt>
  <minimum_core_dt>0.2</minimum_core_dt>
  <minimum_core_reaction_dt>0.05</minimum_core_reaction_dt>
  <max_dt>0</max_dt>
  <temperature>298.15</temperature>
  <seed>20261107</seed>
  <convergence_check>true</convergence_check>
  <convergence_interval>10</convergence_interval>
  <convergence_tol>0.05</convergence_tol>
  <checkpoint_interval>0</checkpoint_interval>
</simulation>
```

| Tag | Accepted values | Function |
|---|---|---|
| `n_trajectories` | Integer $\ge 1$ | Trajectories launched from the $b$-sphere |
| `max_steps` | Integer $\ge 1$ | Step cap per trajectory |
| `dt` | Float (ps) | Outer time step for flexible-chain runs, ignored in rigid-body runs |
| `minimum_core_dt` | Float $\ge 0$ (ps), `0` for none | Lower bound on the time step |
| `minimum_core_reaction_dt` | Float $\ge 0$ (ps), default `0` | Lower bound on the time step near a reaction surface in state-machine or CPU runs |
| `max_dt` | Float $\ge 0$ (ps), `0` for none | Upper bound on the time step |
| `temperature` | Float (K) | Sets $k_B T$ for APBS, desolvation grids, diffusion coefficients, and forces |
| `seed` | Integer $\ge 0$ | Base random number seed |
| `convergence_check` | Boolean | Writes `bd_sims/convergence.json` on completion |
| `convergence_interval` | Integer (%), `0` disables | Progress interval for printing running $P_{\mathrm{rxn}}$ and $k_{\mathrm{on}}$ |
| `convergence_tol` | Float in $(0, 1)$ | Relative standard error threshold for the `converged` flag |
| `checkpoint_interval` | Integer $\ge 0$, `0` disables | Completed trajectories between checkpoint writes |

The trypsin-benzamidine benchmark launches `20,000,000` trajectories with `max_steps` `10,000,000`. Under the default 4-GPU `submit.sh`, trajectories split across four shards of `5,000,000` with seeds `seed + 11111111 * i` for $i = 1 \dots 4$.

### Adaptive Time Step

Rigid-body runs propagate with an adaptive time step and ignore `dt`. At every step, the GPU engine computes a candidate time step from center separation and net charges, bounded beyond $1.1b$ by distance to the $b$- and $q$-spheres. Soft repulsion or overlap checks reduce the step near the receptor surface. `max_dt` caps the step from above, and `minimum_core_dt` is applied last as a floor over every other rule. In the trypsin-benzamidine run, `minimum_core_dt` is `0.2 ps` with no ceiling, preventing any step from dropping below `0.2 ps`.

`minimum_core_reaction_dt` governs state-machine runs when `state_machine_reactions` is set to `true` in `bd_sims/input.xml` as described in Section 14, reducing the time step near an engaged reaction surface down to `minimum_core_reaction_dt` unless overridden by a larger `minimum_core_dt`. `state_machine_reactions` defaults to `false`, so `0.05` has no effect in the trypsin-benzamidine run.

### Step Limit

Every trajectory terminates by reaction, escape, or reaching `max_steps`. Trajectories terminating at `max_steps` are counted as non-reactive in $P_{\mathrm{rxn}}$. `results.json` records `n_max_steps` together with `k_on_bound_low` (all step-capped trajectories treated as non-reactive) and `k_on_bound_high` (all step-capped trajectories treated as reactive). Running `python -m pystarc.analysis.step_limit bd_sims/trajectories.csv` analyzes a rigid-body or flexible-chain pilot `trajectories.csv` and recommends a `max_steps` value limiting truncation bias in $k_{\mathrm{on}}$ to at most 1%.

### Convergence Analysis

After $N$ trajectories, the relative standard error of $P_{\mathrm{rxn}}$ is $\sqrt{(1 - P_{\mathrm{rxn}}) / (N P_{\mathrm{rxn}})}$. With `convergence_check` set to `true`, PySTARC writes `bd_sims/convergence.json` containing $P_{\mathrm{rxn}}$, $k_{\mathrm{on}}$, standard errors, 95% Wilson intervals, the `converged` flag, and projected trajectory counts for 10%, 5%, and 1% relative standard error. The `converged` flag is `true` when $0 < P_{\mathrm{rxn}} < 1$ and the relative standard error lies below `convergence_tol`.

`convergence_interval` prints running $P_{\mathrm{rxn}}$ and $k_{\mathrm{on}}$ at each percentage multiple of `n_trajectories`, alongside a status line every 10 s once trajectories complete. Convergence flags never terminate execution early. All `n_trajectories` run to completion.

### Checkpointing

When `checkpoint_interval` is positive, PySTARC writes `bd_sims/checkpoint.npz` every `checkpoint_interval` completed trajectories. Resume an interrupted run from the run directory with `python /path/to/PySTARC/run_pystarc.py bd_sims/input.xml`. Both `run.sh` and `submit.sh` erase `bd_sims` before starting and must not be invoked to resume. Resuming requires matching input structures and physical parameters, whereas `gpu_force_batch`, `n_threads`, `convergence_interval`, `<outputs>` switches, and positive `checkpoint_interval` may change. A mismatched checkpoint halts execution with an error. For positive `save_interval`, resumed `paths.npz` and `energetics.npz` contain records written after restart. Checkpoints are removed upon normal completion.

### Temperature and Random Seed

`temperature` scales $k_B T$ across APBS, desolvation grids, diffusion coefficients, and forces while retaining the water viscosity of `0.890 mPa·s` at 25 °C. Temperatures other than `298.15 K` emit a warning and require a consistent `debye_length`. Random streams depend strictly on `seed`, trajectory index, and step counter, remaining invariant to `gpu_force_batch` and checkpoint restarts.

## 9. Compute Resources

The `<compute>` section selects execution hardware, multi-GPU sharding, force batch size, CPU worker count, and output directory.

```xml
<compute>
  <gpu>true</gpu>
  <n_gpus>1</n_gpus>
  <gpu_force_batch>0</gpu_force_batch>
  <n_threads>32</n_threads>
  <work_dir>bd_sims</work_dir>
</compute>
```

| Tag | Accepted values | Function |
|---|---|---|
| `gpu` | Boolean | `true` selects the CuPy GPU engine, `false` selects the CPU engine |
| `n_gpus` | Integer $\ge 0$, default `1` | Parallel GPU shard count, overridden by `--n-gpus N` |
| `gpu_force_batch` | Integer $\ge 0$, `0` for automatic | Trajectories evaluated per GPU force batch |
| `n_threads` | Integer | Worker process count for CPU rigid-body or flexible-chain execution |
| `work_dir` | Directory path, default `bd_sims` | Output folder for prepared inputs, grids, and simulation outputs |

When `gpu` is `true` and `n_gpus` is `1`, all trajectories execute on one GPU via CuPy. When `n_gpus` exceeds `1`, `run_pystarc.py` prepares inputs and APBS grids once in `work_dir`, splits `n_trajectories` across `work_dir/bd_1` through `work_dir/bd_N`, executes shards concurrently across visible GPUs, and pools results into `work_dir/results.json` and `work_dir/convergence.json`. Execution stops prior to propagation if CuPy cannot execute a CUDA kernel.

`gpu_force_batch` bounds GPU memory by partitioning force evaluation across batches. Setting `0` selects $\max(1000,\ 4 \times 1024^3 / (150 \times N_{\mathrm{lig}}))$ trajectories per batch, capped at the concurrent trajectory count. For benzamidine with 18 atoms plus `GHO`, the automatic batch size is `1,507,006`. Batch size does not alter random streams.

### Rigid-Body CPU Engine

Setting `gpu` to `false` on a rigid-body complex executes trajectories across `n_threads` CPU worker processes, bounded by available cores and trajectory count. Values $\le 1$ run serially. Differences relative to the GPU engine of Sections 7 and 8 are summarized below.

| Tag | Rigid-body CPU engine behavior |
|---|---|
| `soft_repulsion` | `true` halts execution prior to propagation |
| `overlap_check` | `true` rejects moves bringing any receptor-ligand atom pair below $0.5 (r_i + r_j)$, redrawing up to five times |
| `lj_forces` | `true` applies Weeks-Chandler-Andersen repulsion at close contact |
| `enable_born2_torque` | Controls bidirectional Born desolvation identically to the GPU engine |
| `multipole_fallback` | Ignored, with no exterior electrostatic field beyond the grids |
| `r_escape` | Superseded when the outer propagator handles excursions beyond $1.1b$ |
| `minimum_core_dt` | Lower time-step bound (ps) when positive |
| `max_dt` | Ignored |
| `minimum_core_reaction_dt` | Time-step floor near the reaction surface (ps), where `0` defaults to `0.05 ps` and quarters when the shortest contact distance falls below 1.5 times its cutoff |
| `seed` | Assigns `seed + i` to trajectory $i$ |
| `convergence_interval`, `checkpoint_interval` | Ignored |
| `save_interval`, `full_paths`, `energetics` | Ignored |

The rigid-body CPU engine writes `bd_sims/results.json`, `bd_sims/trajectories.csv`, and `bd_sims/convergence.json`, omitting `paths.npz` and `energetics.npz`. In the GPU trypsin-benzamidine run, `n_threads` has no effect.

## 10. Trajectory Recording

The `<outputs>` section controls periodic recording of trajectory coordinates and forces or energies.

```xml
<outputs>
  <save_interval>1000</save_interval>
  <full_paths>true</full_paths>
  <energetics>true</energetics>
</outputs>
```

| Tag | Accepted values | Function |
|---|---|---|
| `save_interval` | Integer $\ge 0$ | Steps between recorded frames per running trajectory |
| `full_paths` | Boolean, default `true` | Controls writing `bd_sims/paths.npz` |
| `energetics` | Boolean, default `true` | Controls writing `bd_sims/energetics.npz` |

Every `save_interval` steps, a GPU run appends one frame per running trajectory to `paths.npz` and `energetics.npz`.

| File | Stored columns |
|---|---|
| `paths.npz` | `traj_id`, `step`, `x`, `y`, `z`, `q0`, `q1`, `q2`, `q3` for ligand centroid (Å) and orientation quaternion |
| `energetics.npz` | `traj_id`, `step`, `fx`, `fy`, `fz`, `dt` for net ligand force ($k_B T/\text{Å}$) and time step (ps) in rigid-body GPU runs |

In rigid-body GPU runs, `energetics.npz` stores forces and time steps rather than energies. Flexible-chain runs store `total`, `elec`, `born`, and `steric` energies instead as described in Section 15. Setting `save_interval` to `0` records initial and final poses in `paths.npz` and omits `energetics.npz`. The trypsin-benzamidine benchmark records every `1000` steps.

## 11. Flexible Chain Section

Rigid-body configurations hold every tag inside `<chain>` blank.

```xml
<chain>
  <reaction_pairs_json></reaction_pairs_json>
  <dt_chain></dt_chain>
  <chain_steps_per_outer></chain_steps_per_outer>
  <n_equilibration_steps></n_equilibration_steps>
  <auto_diffusion></auto_diffusion>
  <d_trans></d_trans>
  <d_rot></d_rot>
  <soft_repulsion_eps></soft_repulsion_eps>
  <elastic_network_cutoff></elastic_network_cutoff>
  <elastic_network_k></elastic_network_k>
  <target_grid_dx></target_grid_dx>
  <born_grid_dx></born_grid_dx>
</chain>
```

Populating any `<chain>` tag with `model` set to `rigid` halts the build with an error. Section 15 documents `<chain>` parameters for `model` `chain`.

## 12. Building and Running Simulations

`run_pystarc.py` executes input preparation, APBS electrostatic and Born grid generation, and Brownian dynamics propagation in one command.

```bash
python /path/to/PySTARC/run_pystarc.py config.xml
```

Passing `--setup` stops immediately after input preparation, prior to APBS and propagation.

```bash
python /path/to/PySTARC/run_pystarc.py config.xml --setup
```

During preparation, `run_pystarc.py` writes `receptor.pqr`, `ligand.pqr`, `rxns.xml`, `input.xml`, and `build_record.json` into `work_dir` (`bd_sims` by default). Blank required tags, incompatible model or mode tags, or unrecognized tags stop execution before writing files. Temporary AmberTools intermediates (`protein.pdb`, `protein.prmtop`, `ligand.mol2`, `leap.log`) are created and removed inside `bd_sims` without altering the run directory. For trypsin-benzamidine, preparation prints the summary below.

```
  receptor.pqr: 3220 atoms from receptor.pqr
  ligand.pqr: 18 atoms from ligand.pqr
  6 pairs, n_needed 3
  rmax=28.4 lmax=3.7 b=45 fglen=96  WARNING, b-surface outside the grid
```

`rmax` and `lmax` are maximum atomic distances from each solute centroid (Å). The final diagnostic prints `OK` when $b + l_{\max} < \mathrm{fglen}/2$ and `WARNING` otherwise without aborting the build. PDB parameterization prints additional diagnostics including total charge $Q$.

| File in `bd_sims` | Content |
|---|---|
| `receptor.pqr` | Receptor atomic coordinates, charges, and radii |
| `ligand.pqr` | Ligand atomic coordinates, charges, and radii |
| `rxns.xml` | Reaction definition with atom pairs, cutoffs, and `n_needed` |
| `input.xml` | Generated runtime `<pystarc>` input built from Sections 5 through 11 |
| `build_record.json` | Criterion metadata, atom pairs, and MD5 checksums of the four files above |

### Shell and SLURM Scripts

| Command | GPU allocation | Behavior |
|---|---|---|
| `bash run.sh` | 1 local GPU | Removes `bd_sims` and executes `run_pystarc.py config.xml` |
| `sbatch submit.sh` | 1 node, 4 GPUs | Removes `bd_sims`, runs 4 shards via `srun`, pools into `bd_sims/results.json` |
| `sbatch --nodes=1 --ntasks-per-node=1 submit.sh` | 1 node, 1 GPU | Removes `bd_sims` and executes `run_pystarc.py config.xml --n-gpus 1` |
| `sbatch --nodes=2 --ntasks-per-node=4 submit.sh` | 2 nodes, 8 GPUs | Removes `bd_sims`, runs 8 shards across nodes via `srun`, pools into `bd_sims/results.json` |

Submit `submit.sh` from the complex directory after adapting `#SBATCH` directives (`--partition`, `--mem-per-gpu=64G`, and wall time) to the local cluster. When `conda` is absent on compute nodes, `submit.sh` sources `~/.bashrc` or searches `$CONDA_BASE` and standard conda install paths before activating `PySTARC`. Both `run.sh` and `submit.sh` delete `bd_sims` along with prior outputs and checkpoints.

In multi-GPU runs (`--n-gpus 4` or `<n_gpus>4</n_gpus>`), `run_pystarc.py` prepares inputs and APBS grids once in `bd_sims` and launches shards in `bd_sims/bd_1` through `bd_sims/bd_4`, each running one quarter of `n_trajectories` under its own seed. `pystarc/multi_GPU/combine_data.py` then pools finished shards into `bd_sims/results.json` and `bd_sims/convergence.json`. Shards lacking `results.json` are excluded from pooling.

### Resuming from Runtime Input

To resume from a checkpoint or execute a modified `bd_sims/input.xml` without deleting `bd_sims`, pass `bd_sims/input.xml` directly.

```bash
python /path/to/PySTARC/run_pystarc.py bd_sims/input.xml
```

Execution resumes from `bd_sims/checkpoint.npz` when present and reuses existing grids and hydrodynamic radius caches whenever PQR files and grid parameters remain unchanged. For a multi-GPU run, resume each incomplete shard with `python /path/to/PySTARC/run_pystarc.py bd_sims/bd_i/input.xml` and run `python /path/to/PySTARC/pystarc/multi_GPU/combine_data.py` from the run directory.

## 13. Results and Output Files

Every execution writes outputs into `bd_sims` and logs terminal output to `bd_sims/pystarc_<date>_<time>.log`, ending with the rate summary and convergence report when `convergence_check` is `true`. Runtime failures append the error message to the log. Invalid XML entries halt execution prior to log creation. Multi-GPU runs write per-shard logs in `bd_sims/bd_1` through `bd_sims/bd_N`.

### results.json

`bd_sims/results.json` stores $k_{\mathrm{on}}$, statistical bounds, and trajectory counts. After multi-GPU execution, `bd_sims/results.json` holds the pooled summary.

| Field | Content |
|---|---|
| `k_on` | Association rate constant $6.02214076 \times 10^8\,k_b P_{\mathrm{rxn}}$ ($\text{M}^{-1}\text{s}^{-1}$) |
| `k_on_low`, `k_on_high` | Wilson confidence interval on $k_{\mathrm{on}}$ at `confidence_level` ($\text{M}^{-1}\text{s}^{-1}$) |
| `P_rxn` | Reaction probability `n_reacted / n_trajectories` |
| `P_rxn_low`, `P_rxn_high` | Wilson confidence interval on $P_{\mathrm{rxn}}$ at `confidence_level` |
| `P_rxn_bound_low`, `P_rxn_bound_high` | Lower and upper bounds on $P_{\mathrm{rxn}}$ treating `n_max_steps` trajectories as non-reactive and reactive |
| `k_on_bound_low`, `k_on_bound_high` | Lower and upper bounds on $k_{\mathrm{on}}$ ($\text{M}^{-1}\text{s}^{-1}$) |
| `k_b` | Diffusive encounter rate constant from infinity to the $b$-sphere ($\text{Å}^3/\text{ps}$) |
| `D_rel` | Relative translational diffusion coefficient $D_{\mathrm{rec}} + D_{\mathrm{lig}}$ at infinite separation ($\text{Å}^2/\text{ps}$) |
| `n_trajectories` | Total trajectories launched |
| `n_reacted`, `n_escaped`, `n_max_steps` | Counts of reacted, escaped, and step-capped trajectories |
| `confidence_level` | Value of `confidence_interval` in `input.xml`, default `0.95` |
| `r_start`, `r_escape` | Radii $b$ and $q$ (Å) |
| `log10_k_on` | $\log_{10} k_{\mathrm{on}}$, written when $k_{\mathrm{on}} > 0$ |
| `convergence_tol` | Value of `convergence_tol` from Section 8 |
| `wall_time_sec`, `steps_per_sec`, `total_steps` | Propagation wall time (s), throughput ($\text{steps/s}$), and cumulative BD steps |
| `contact_total_steps` | Step denominator of `contact_frequency.csv` |
| `k_on_units`, `k_b_units`, `D_rel_units` | Physical unit strings |
| `completed_reactions` | Per-reaction firing counts when `state_machine_reactions` is `true` |

The prefactor $6.02214076 \times 10^8$ converts $\text{Å}^3/\text{ps}$ to $\text{M}^{-1}\text{s}^{-1}$. Step-capped trajectories count as non-reactive in $P_{\mathrm{rxn}}$. Lower bounds therefore repeat $P_{\mathrm{rxn}}$ and $k_{\mathrm{on}}$, while upper bounds add `n_max_steps` to `n_reacted`. Wilson intervals evaluate $z = 1.96$ at `confidence_level` `0.95`. Pooled multi-GPU files omit `total_steps` and `contact_total_steps`, reporting the slowest shard duration in `wall_time_sec` and total GPU-seconds across shards in `aggregate_gpu_sec`.

### convergence.json

`run_pystarc.py` writes `bd_sims/convergence.json` when `convergence_check` is `true`, and `combine_data.py` writes the pooled file for multi-GPU runs.

| Field | Content |
|---|---|
| `N`, `n_reacted`, `n_escaped`, `n_max_steps` | Trajectory counts, with `n_max_steps` omitted in the pooled file |
| `P_rxn`, `SE` | Reaction probability and standard error $\sqrt{P_{\mathrm{rxn}}(1 - P_{\mathrm{rxn}})/N}$ |
| `relative_SE`, `relative_SE_pct` | $\mathrm{SE} / P_{\mathrm{rxn}}$ as fraction and percentage |
| `k_on`, `SE_kon` | $k_{\mathrm{on}}$ and its standard error ($\text{M}^{-1}\text{s}^{-1}$) |
| `wilson_CI`, `wilson_CI_P` | Wilson intervals on $k_{\mathrm{on}}$ ($\text{M}^{-1}\text{s}^{-1}$) and $P_{\mathrm{rxn}}$ |
| `converged` | `true` when $0 < P_{\mathrm{rxn}} < 1$ and `relative_SE < convergence_tol` |
| `confidence_level`, `tol`, `tol_pct` | Confidence level and tolerance as fraction and percentage |
| `N_needed` | Projected trajectory counts $(1 - P_{\mathrm{rxn}})/(P_{\mathrm{rxn}} r^2)$ for relative standard errors $r$ of 10%, 5%, and 1% |

### Output Inventory

| File in `bd_sims` | Content |
|---|---|
| `pystarc_<date>_<time>.log` | Terminal log of one `run_pystarc.py` invocation |
| `input.xml`, `rxns.xml`, `build_record.json` | Runtime configuration, reaction pairs, and build checksums |
| `receptor.pqr`, `ligand.pqr` | Centroid-shifted solute PQR files with appended `GHO` centroid atoms |
| `receptor0.dx`, `receptor1.dx` | Coarse and fine receptor electrostatic grids |
| `receptor0_born.dx`, `receptor1_born.dx` | Coarse and fine receptor Born desolvation grids |
| `ligand0.dx`, `ligand1.dx`, `ligand0_born.dx`, `ligand1_born.dx` | Coarse and fine ligand electrostatic and Born desolvation grids |
| `receptor_grids.key`, `ligand_grids.key` | Checksums of ghost-free PQR bytes and grid settings for grid caching |
| `receptor.pqr.r_hydro_*.cache`, `ligand.pqr.r_hydro_*.cache` | Cached Monte Carlo hydrodynamic radii |
| `results.json` | Rate constant, confidence intervals, and trajectory counts |
| `convergence.json` | Statistical precision diagnostics for $P_{\mathrm{rxn}}$ |
| `trajectories.csv` | Outcome, step count, initial pose, and minimum separation per trajectory |
| `encounters.csv` | Ligand centroid and quaternion at reaction |
| `near_misses.csv` | Ligand centroid and quaternion at closest approach for escaped trajectories |
| `bd_time_to_reaction.csv` | Cumulative BD time inside the escape sphere for reacted trajectories |
| `contact_frequency.csv` | Step counts and residence times within `pair_cutoff` per reaction pair |
| `pose_clusters.csv` | Rotation-angle cluster assignment across up to 8 bins, written when reactions exceed 5 |
| `radial_density.csv` | Step counts and residence times per radial shell |
| `angular_map.npz` | Step counts and residence times per angular bin |
| `milestone_flux.csv` | Outward and inward crossings across 11 spherical shells from $b$ to $q$ |
| `transition_matrix.npz` | Transition counts across 50 radial bins |
| `paths.npz`, `energetics.npz` | Periodic trajectory snapshots every `save_interval` steps |
| `checkpoint.npz` | Restart state written when `checkpoint_interval > 0`, deleted upon completion |
| `split.json`, `bd_1` to `bd_N` | Multi-GPU shard metadata and per-shard output directories |

Multi-GPU runs store per-shard `input.xml`, logs, `results.json`, and trajectory files inside `bd_sims/bd_1` through `bd_sims/bd_N`, while storing shared APBS grids, `split.json`, and pooled outputs in `bd_sims`.

## 14. Runtime Input Overrides

`run_pystarc.py config.xml` writes `bd_sims/input.xml` under root tag `<pystarc>`. Low-level `<pystarc>` tags outside `config.xml` can be configured by running `python run_pystarc.py config.xml --setup`, editing `bd_sims/input.xml`, and executing `python run_pystarc.py bd_sims/input.xml`. Top-level options belong directly under `<pystarc>`, and file switches belong inside `<outputs>`.

```xml
  <confidence_interval>0.99</confidence_interval>
  <outputs>
    <save_interval>1000</save_interval>
    <full_paths>true</full_paths>
    <energetics>true</energetics>
    <near_misses_csv>false</near_misses_csv>
  </outputs>
```

| Tag in `<pystarc>` | Accepted values | Function |
|---|---|---|
| `apbs_ligand_dime` | $4k + 1$ up to `65`, $32c + 1$ above `65`, default `129` | Points per axis for ligand APBS and Born grids |
| `brownian_bridge` | Boolean, default `true` | `true` checks reaction crossings along sub-step Brownian bridges as well as endpoints |
| `confidence_interval` | Float in $(0, 1)$, default `0.95` | Confidence level of Wilson intervals |
| `state_machine_reactions` | Boolean, default `false` | `true` evaluates multi-stage `state_before` to `state_after` transitions in `rxns.xml`, `false` evaluates a single combined criterion |
| `relative_viscosity` | `1`, default `1` | Solvent viscosity relative to water at 25 °C |
| `trajectories_csv`, `encounters_csv`, `near_misses_csv`, `radial_density`, `angular_map`, `bd_time_to_reaction`, `contact_frequency`, `milestone_flux`, `transition_matrix`, `pose_clusters` | Boolean, default `true` | `false` omits writing the corresponding output file |
| `results_json` | Boolean, default `true` | `false` omits `results.json`, sets $k_{\mathrm{on}} = 0$ in `convergence.json`, and disables multi-GPU pooling |

Running `python run_pystarc.py --template input.xml` emits a standalone `<pystarc_input>` template for direct low-level engine execution without `config.xml`.

## 15. Flexible Chain Model

The flexible-chain engine propagates a coarse-grained COFFDROP bead chain against a rigid atomistic receptor on CPUs without requiring a GPU. The barnase-barstar benchmark in `examples/barnase_barstar/config.xml` shares the nine-section `<config>` schema with `model` set to `chain` and `<chain>` populated.

```xml
  <structure>
    <receptor>receptor.pdb</receptor>
    <ligand>ligand.pdb</ligand>
    <ligand_net_charge>-5</ligand_net_charge>
    <model>chain</model>
  </structure>
  <criterion>
    <mode>contacts</mode>
    <receptor></receptor>
    <ligand></ligand>
    <pair_cutoff>8.0</pair_cutoff>
    <n_needed>3</n_needed>
  </criterion>
  <chain>
    <reaction_pairs_json>reaction_pairs.json</reaction_pairs_json>
    <dt_chain>0.025</dt_chain>
    <chain_steps_per_outer>8</chain_steps_per_outer>
    <n_equilibration_steps>0</n_equilibration_steps>
    <auto_diffusion>true</auto_diffusion>
    <d_trans>0.015</d_trans>
    <d_rot>0.00005</d_rot>
    <soft_repulsion_eps>1.0</soft_repulsion_eps>
    <elastic_network_cutoff>9.0</elastic_network_cutoff>
    <elastic_network_k>2.0</elastic_network_k>
    <target_grid_dx>apbs_output/barnase1.dx</target_grid_dx>
    <born_grid_dx>apbs_output/barnase0_born.dx</born_grid_dx>
  </chain>
```

On a workstation, lower `n_threads` and `n_trajectories` to fit local CPU cores and run `bash run.sh`, or invoke `python ../../run_pystarc.py config.xml` once `apbs_output` exists or with blank `target_grid_dx` and `born_grid_dx` so APBS runs inside `bd_sims`. On a SLURM cluster, adjust `--partition`, `--constraint`, and `--mem` in `submit.sh` to execute the `10,000`-trajectory multi-node workflow pinned to 16-core NUMA domains.

```bash
cd examples/barnase_barstar
sbatch submit.sh                    # 2 nodes x 8 tasks x 16 CPUs (256 CPUs total, default)
sbatch --nodes=1 submit.sh          # 1 node x 8 tasks x 16 CPUs (128 CPUs total)
sbatch --nodes=10 submit.sh         # 10 nodes x 8 tasks x 16 CPUs (1,280 CPUs total)
```

Alternatively, run the three workflow stages separately with `python make_grids.py`, `./submit_shards.sh`, and `python combine_shards.py`.

During preparation via `run_pystarc.py config.xml --setup`, PySTARC parameterizes `receptor.pdb` with `tleap` and `ff14SB` to write centered `bd_sims/receptor.pqr`, constructs `bd_sims/chain.json` containing 230 COFFDROP beads across 87 barstar residues, copies the 568 receptor-atom and chain-bead pairs of `reaction_pairs.json` into `bd_sims/reaction_pairs.json` (or derives pairs from `<criterion>` when `reaction_pairs_json` is blank), and writes `bd_sims/input.xml` and `bd_sims/build_record.json`.

`make_grids.py` runs APBS on `bd_sims/receptor.pqr` with the `<electrostatics>` settings (`srad` 1.5 Å, `apbs_fglen` 140.0 Å, `apbs_dime` 449, plus `apbs_coarse_dime` and `apbs_fine_dime` when nonzero) and writes `barnase0.dx`, `barnase1.dx`, `barnase0_born.dx`, and `barnase1_born.dx` into `apbs_output`.

`submit.sh` partitions the `10,000` trajectories evenly across 16-CPU NUMA-local SLURM tasks in `shards/shard_01` through `shards/shard_N` with seed `11111111 * i + 1`, executes each shard under `numactl --localalloc`, and invokes `combine_shards.py` upon completion. `submit_shards.sh` instead submits independent single-node 125-CPU jobs. Each shard writes nested `results.json`, `trajectories.csv`, `fpt_distribution.csv`, `energetics.npz` storing `total`, `elec`, `born`, and `steric` energies, `paths.npz`, and a run log inside its shard `bd_sims` directory. `convergence_check` is omitted in chain mode, and `combine_shards.py` reports per-shard counts, pooled $k_{\mathrm{on}} = k_b P_{\mathrm{rxn}}$ with $P_{\mathrm{rxn}} = n_{\mathrm{reacted}} / (n_{\mathrm{reacted}} + n_{\mathrm{escaped}} + n_{\mathrm{max\_steps}})$, the 95% Wilson interval, and upper bounds when $n_{\mathrm{max\_steps}} > 0$.

Each trajectory initializes the chain centroid at radius $b$ = `bd_milestone_radius` around the receptor centroid in a random orientation. Each outer step of duration `dt` translates and rotates the chain as a rigid body while `chain_steps_per_outer` internal steps of duration `dt_chain` update bead coordinates. Beyond $1.1b$, the analytical outer propagator with $q = 20 \times r_{\max}$ returns the chain to the $b$-sphere or terminates the trajectory as escaped, superseding `r_escape`.

| Tag | Section | Accepted values | Function |
|---|---|---|---|
| `receptor`, `ligand`, `model` | `<structure>` | `.pdb` receptor, `.pdb` protein chain, `chain` | Receptor charges and radii, chain sequence and initial bead coordinates |
| `mode`, `receptor`, `ligand`, `pair_cutoff`, `n_needed` | `<criterion>` | `contacts`, optional target entries, distance (Å), count | Pair reaction cutoff and simultaneous pair threshold |
| `reaction_pairs_json` | `<chain>` | JSON path, blank derives pairs from `<criterion>` | Pre-built `[rec_atom_idx, bead_idx, cutoff]` pairs |
| `dt` | `<simulation>` | `chain_steps_per_outer` $\times$ `dt_chain` (ps) | Outer time step, reduced to `0.05 ps` when any reaction pair lies within $1.5 \times \mathrm{pair\_cutoff}$ |
| `dt_chain` | `<chain>` | Float $> 0$ (ps) | Internal bead time step |
| `chain_steps_per_outer` | `<chain>` | Integer $\ge 1$ | Internal steps executed per outer step `dt` |
| `n_equilibration_steps` | `<chain>` | Integer $\ge 0$ | Internal bead equilibration steps in zero receptor field prior to launch |
| `auto_diffusion` | `<chain>` | Boolean | `true` recomputes chain RPY diffusion tensors every 10 outer steps, `false` applies `d_trans` and `d_rot` |
| `d_trans`, `d_rot` | `<chain>` | Float $> 0$ ($\text{Å}^2/\text{ps}$), Float $> 0$ ($\text{rad}^2/\text{ps}$) | Constant chain diffusion coefficients when `auto_diffusion` is `false` |
| `soft_repulsion`, `soft_repulsion_eps` | `<physics>`, `<chain>` | Boolean, Float $\ge 0$ ($k_B T$) | WCA repulsion $4\varepsilon[(\sigma/r)^{12} - (\sigma/r)^6] + \varepsilon$ for $r < 2^{1/6}\sigma$ between beads and receptor atoms and nonbonded bead pairs |
| `elastic_network_cutoff` | `<chain>` | Float $\ge 0$ (Å), blank or `0` disables | Cutoff distance for native-fold backbone $\mathrm{C}_\alpha$ elastic network |
| `elastic_network_k` | `<chain>` | Float $\ge 0$ ($k_B T/\text{Å}^2$) | Harmonic spring constant of the backbone $\mathrm{C}_\alpha$ elastic network |
| `target_grid_dx` | `<chain>` | DX path, blank builds inside `work_dir` | Receptor APBS fine electrostatic potential grid acting on bead charges |
| `born_grid_dx` | `<chain>` | DX path, blank builds inside `work_dir` | Receptor Born desolvation grid acting on bead charges, scaled by `desolvation_alpha` |
| `n_threads` | `<compute>` | Integer $\ge 1$ | CPU worker processes per run or shard |

## 16. Troubleshooting

| Symptom | Resolution |
|---|---|
| `install.sh` exits with no NVIDIA GPU or CUDA below 12 | Run `install.sh` on a Linux GPU node with a CUDA 12+ driver, or install manually on a login node per `README.md` |
| `pip` reports `proprep` or `numpy<2` conflicts | Pin `ambertools=24.8` and `"numpy>=2.0,<2.6"` from `conda-forge` |
| APBS grid-dimension validation error | Set `apbs_dime`, `apbs_coarse_dime`, and `apbs_fine_dime` to $4k + 1$ up to `65` or $32c + 1$ above `65` |
| GPU out-of-memory error | Lower `gpu_force_batch` inside `<compute>`, such as `100000` |
| SLURM rejects partition, constraint, or memory | Adapt `#SBATCH` directives at the top of `submit.sh` to the local cluster |
