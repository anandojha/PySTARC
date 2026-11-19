# PySTARC tutorial

This tutorial computes the association rate constant k<sub>on</sub> of trypsin and benzamidine with rigid body Brownian dynamics on an NVIDIA GPU. The input files are in `examples/trypsin_benzamidine`. Steps 3 to 11 fill the nine sections of `config.xml` one by one and explain every tag with the accepted entries. Every example shares the same `config.xml` template, with tags unused by a complex left blank (`<tag></tag>`). A new complex follows the same steps with prepared receptor and ligand PDB or PQR files (step 3). The flexible chain model is described in step 15 (`examples/barnase_barstar`).

## 1. Install PySTARC

PySTARC runs on a Linux node with an NVIDIA GPU and a driver for CUDA 12 or newer.

```bash
git clone https://github.com/anandojha/PySTARC.git
cd PySTARC
bash install.sh
conda activate PySTARC
```

`install.sh` creates the conda environment PySTARC with AmberTools 24.8, APBS, RDKit, OpenBabel and CuPy. An existing environment named PySTARC is replaced. On a host without an NVIDIA GPU or with an older driver the script stops before any change.

## 2. Create a run folder

```bash
mkdir examples/my_trypsin
cd examples/my_trypsin
cp ../trypsin_benzamidine/{receptor.pqr,ligand.pqr,run.sh,submit.sh} .
python ../../run_pystarc.py --template > config.xml
```

The last command writes a `config.xml` template with the nine sections `structure`, `criterion`, `geometry`, `electrostatics`, `physics`, `simulation`, `compute`, `outputs` and `chain`. `run.sh` and `submit.sh` find PySTARC in a parent folder of the run folder. A run folder outside the PySTARC folder needs `export PYSTARC_ROOT=/path/to/PySTARC`.

| file | content |
|---|---|
| `receptor.pqr` | trypsin coordinates, charges and radii in the bound pose |
| `ligand.pqr` | benzamidine coordinates, charges and radii in the bound pose |
| `config.xml` | settings of the build and the simulation |
| `run.sh` | runs `run_pystarc.py config.xml` on one GPU |
| `submit.sh` | unified SLURM submission script (defaults to 1 node, 4 GPUs, scales to 1 GPU or multiple nodes via `sbatch` flags) |

## 3. Receptor and ligand structures (`structure`)

`<structure>` names two separate solute files (`receptor` and `ligand`) in matching formats (both `.pdb` or both `.pqr`), plus `ligand_net_charge` and `model`.

| tag | accepted entries | effect |
|---|---|---|
| `receptor` | `.pdb` or `.pqr` path | prepared receptor structure; `.pdb` is parameterized with AmberTools (`tleap` and `ff14SB`), `.pqr` supplies existing charges and radii |
| `ligand` | `.pdb` or `.pqr` path (matching `receptor`) | prepared ligand structure; small-molecule `.pdb` is parameterized with `antechamber` (`AM1-BCC` and `GAFF2`), protein `.pdb` is parameterized with `tleap` (`rigid`) or built into COFFDROP beads (`chain`, step 15), and `.pqr` supplies existing charges and radii |
| `ligand_net_charge` | integer | formal net charge of the ligand (in elementary charge units); passed to `antechamber` for a small-molecule `.pdb` and verified against the summed atomic or bead charges of `.pqr` and protein `.pdb` ligands |
| `model` | `rigid` or `chain` | rigid-body Brownian dynamics (`rigid`) or flexible COFFDROP bead chain (`chain`, step 15) |

Mixing `.pdb` and `.pqr` in one `config.xml` stops the build with an error. The trypsin benzamidine example supplies PQR files matching the atomic charges and radii of the BrownDye 2 comparison:

```xml
<structure>
  <receptor>receptor.pqr</receptor>
  <ligand>ligand.pqr</ligand>
  <ligand_net_charge>1</ligand_net_charge>
  <model>rigid</model>
</structure>
```

`run_pystarc.py` copies `receptor.pqr` and `ligand.pqr` into `bd_sims/receptor.pqr` and `bd_sims/ligand.pqr`, removing any ghost atoms left from an earlier run.

### Starting from PDB files

A new protein-ligand complex starts from prepared `receptor.pdb` and `ligand.pdb` files. For example, `examples/hsp90_inhibitors/HSP90-aminopyridine` configures:

```xml
<structure>
  <receptor>receptor.pdb</receptor>
  <ligand>ligand.pdb</ligand>
  <ligand_net_charge>0</ligand_net_charge>
  <model>rigid</model>
</structure>
```

`run_pystarc.py` parameterizes `receptor.pdb` with `tleap` (`ff14SB`). RDKit assigns the ligand bond orders from the coordinates, explicit hydrogens and `ligand_net_charge` in `ligand.pdb`, and `antechamber` assigns `AM1-BCC` partial charges and `GAFF2` atom types.

| file | requirement |
|---|---|
| `receptor.pdb` | `ATOM` records of a single conformer with AMBER residue names (`HID`, `HIE` or `HIP` for histidine and `CYX` for a disulfide; plain `HIS` is read as `HIE`); catalytic divalent metals (`ZN`, `MG`, `MN`, `FE`, `CU`, `CO`, `NI`, `CAL`) retained after `TER` |
| `ligand.pdb` | single-conformer `HETATM`/`ATOM` records with explicit hydrogens matching `ligand_net_charge` (or a single protein chain for `model` `chain`) |

## 4. Reaction criterion (`criterion`)

The reaction criterion defines the bound state. For rigid body runs the build step writes the criterion into `bd_sims/rxns.xml` as receptor and ligand atom pairs with a reaction distance `pair_cutoff`. A trajectory reacts when at least `n_needed` pairs are closer than `pair_cutoff` at one time. `mode` selects one of two criteria.

| `mode` | pairs |
|---|---|
| `contacts` | receptor and ligand atom pairs listed in `<receptor>` and `<ligand>` |
| `centroid` | receptor centroid paired with ligand centroid |

### Contact pairs (`contacts`)

The trypsin benzamidine example specifies six contact pairs with `pair_cutoff` 7.0 Å and `n_needed` 3:

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

| receptor atom in `receptor.pqr` | ligand atom | distance in the bound pose (Å) |
|---|---|---|
| TRP 193 O | N2 | 2.82 |
| SER 192 O | N2 | 2.86 |
| CYX 173 O | N1 | 3.17 |
| VAL 205 O | N2 | 3.34 |
| GLN 174 N | N1 | 4.24 |
| GLY 194 N | N2 | 4.31 |

Each `<target resid="..." atom="..." resname="..."/>` in `<receptor>` names one receptor atom by residue number, atom name and residue name. When `<receptor>` is a `.pdb` file, residue numbers written in `receptor.pdb` (including retained catalytic metals such as `ZN`) are mapped automatically to the residue numbers of `bd_sims/receptor.pqr`. Protonation tautomers assigned by `tleap` (`HIS`/`HID`/`HIE`/`HIP`, `CYS`/`CYX`/`CYM`, `ASP`/`ASH`, `GLU`/`GLH`, `LYS`/`LYN`) match across group members. Each entry in `<ligand>` follows the receptor target order as `<atom name="..."/>` for a single-residue ligand, or `<target resid="..." atom="..." resname="..."/>` for a multi-residue protein ligand (`examples/thrombin_thrombomodulin`).

For rigid body runs, one pair requires `n_needed` 1, and an even number of pairs 2k requires `n_needed` k.

### Distance of the centroids (`centroid`)

```xml
<criterion>
  <mode>centroid</mode>
  <receptor></receptor>
  <ligand></ligand>
  <pair_cutoff>8.0</pair_cutoff>
  <n_needed>0</n_needed>
</criterion>
```

The simulation places a ghost atom GHO without charge or radius at the centroid of each molecule. Zero charge, zero radius ghost records are stripped before APBS runs, avoiding any dielectric cavity perturbation in the electrostatic and Born grids. A trajectory reacts when the two centroids are closer than `pair_cutoff` (`n_needed` is set to `0` in `config.xml`, and `rxns.xml` writes `1`). The beta cyclodextrin and two charged spheres examples configure mode `centroid`.

### Tags of `criterion`

| tag | mode | meaning |
|---|---|---|
| `mode` | `contacts` or `centroid` | reaction criterion type |
| `receptor` | `contacts` (blank in `centroid`) | receptor atoms as `<target resid="..." atom="..." resname="..."/>` |
| `ligand` | `contacts` (blank in `centroid`) | ligand atoms as `<atom name="..."/>` or `<target resid="..." atom="..." resname="..."/>` |
| `pair_cutoff` | `contacts` and `centroid` | reaction distance of every pair (Å) |
| `n_needed` | `contacts` and `centroid` | pairs closer than `pair_cutoff` at one time (`0` in `centroid`, where `rxns.xml` writes `1`) |

## 5. Start sphere and hydrodynamic radii (`geometry`)

Every tag of steps 5 to 10 is required in `config.xml`, except `r_escape` (default `0`), `debye_length` (derived from `ion_concentration` when blank), `soft_repulsion` (default `false`), `minimum_core_reaction_dt` (default `0`), `n_gpus` (default `1`), `work_dir` (default `bd_sims`), `full_paths` (default `true`) and `energetics` (default `true`). Rigid body runs leave the tags of `<chain>` (step 11) blank. The build stops with an error for a missing required tag.

```xml
<geometry>
  <bd_milestone_radius>45.0</bd_milestone_radius>
  <r_escape>0</r_escape>
  <r_hydro_rec>22.3728</r_hydro_rec>
  <r_hydro_lig>4.9867</r_hydro_lig>
</geometry>
```

Each trajectory starts at a random position on the sphere of radius b = `bd_milestone_radius` (Å) around the receptor centroid, with a random orientation of the ligand. `r_escape` sets the escape sphere radius q (Å), with `0` or blank deriving `2b` (custom radii must exceed `1.1b`). A trajectory that reaches the escape sphere goes back to the start sphere with the computed probability of a return from q to b. The remaining trajectories at q escape.

`r_hydro_rec` and `r_hydro_lig` set the hydrodynamic radii a of the receptor and the ligand (Å). The radii fix the diffusion coefficients D<sub>t</sub> = k<sub>B</sub>T/(6πηa) and D<sub>r</sub> = k<sub>B</sub>T/(8πηa³) with the viscosity η = 0.890 mPa s of water at 25 °C. PySTARC computes a by the Monte Carlo method of Hansen (J. Chem. Phys. 121, 9111, 2004) from the PQR file, with every atom radius enlarged by `srad`. The calculation runs inside PySTARC (`pystarc/hydrodynamics/mc_hydro_radius.py`) and needs no external software. A cache file next to each PQR file in `bd_sims` stores the Monte Carlo radius for the next run. An entry of 0 selects the Monte Carlo radius. A positive entry replaces the Monte Carlo radius, and the run log prints both values. The trypsin benzamidine example fixes the radii at 22.3728 Å and 4.9867 Å, computed with BrownDye 2 for a comparison of PySTARC with BrownDye 2. An entry of 0 yields 22.30 Å and 4.90 Å for the same PQR files. Hydrodynamic interactions stay on in both cases. `hydrodynamic_interactions` (step 7) switches the hydrodynamic interactions on and off.

## 6. Salt, dielectrics and electrostatic grids (`electrostatics`)

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

| tag | meaning |
|---|---|
| `debye_length` | Debye length λ (Å) of the screened field outside the grids, of the k<sub>b</sub> integral and of the desolvation grids; `inf` for zero salt, blank to derive from `ion_concentration` |
| `ion_concentration` | concentration of a monovalent salt in every APBS calculation (M), 0 for zero salt |
| `ion_radius_pos`, `ion_radius_neg` | radii of the cation and the anion in APBS (Å) |
| `pdie` | dielectric constant of the solute |
| `sdie` | dielectric constant of the solvent |
| `srad` | solvent probe radius of the APBS surface and of the hydrodynamic radius (Å) |

The run stops when a populated `debye_length` differs by more than 1% from the Debye length of `ion_concentration`, `sdie` and `temperature`, for example 7.83 Å at 0.15 M, 78 and 298.15 K.

### Two grids for the receptor potential

APBS solves the linearized Poisson Boltzmann equation of the receptor on two nested cubic grids centered on the receptor, on a temporary ghost free PQR copy (`GHO` stripped).

| grid | file in `bd_sims` | edge (Å) | points per axis | boundary condition |
|---|---|---|---|---|
| coarse | `receptor0.dx` | `apbs_cglen`, 0 for twice the fine edge | `apbs_coarse_dime`, 0 for `apbs_dime` | Debye Hückel potential of one sphere with the charge, dipole and quadrupole of the receptor (APBS `bcfl sdh`) |
| fine | `receptor1.dx` | `apbs_fglen`, 0 for the derived edge | `apbs_fine_dime`, 0 for `apbs_dime` | potential of the coarse grid (APBS `bcfl map`) |

The derived fine edge is max[(L + 2(r<sub>max</sub> + srad)) × 1.05, 4(r<sub>max</sub> + srad) × 1.025], with L the longest extent of the atom centers and r<sub>max</sub> the largest atom radius. `apbs_cglen` must exceed `apbs_fglen`. `apbs_dime` sets the points per axis of the receptor desolvation grids as well.

During the simulation the force on each ligand atom comes from the finest grid that holds the atom at least three grid spacings inside the faces. Outside both grids the receptor field is a screened multipole expansion (`multipole_fallback`, step 7). For trypsin the fine grid spans 96 Å with 257 points (spacing 0.375 Å), and the coarse grid spans 192 Å (spacing 0.75 Å). The build step prints a warning when b plus the ligand radius reaches past the fine grid (b + ligand radius ≥ `apbs_fglen`/2). Trypsin triggers this warning (45 Å + 3.7 Å > 48 Å), and the ligand atoms outside the fine grid read the coarse grid.

### Grid points per axis

The multigrid solver of APBS needs d = c·2<sup>l+1</sup> + 1 points per axis, with integers c ≥ 1 and l ≥ 1 for l multigrid levels. APBS keeps every d = 4k + 1 up to 65, for example 5, 9, 13, 33 and 65. Above 65 APBS accepts d = 32c + 1, for example 97, 129, 161, 193, 225 and 257. APBS lowers any other d above 65 to the next smaller 32c + 1, for example 101 to 97. PySTARC checks `apbs_dime`, `apbs_coarse_dime` and `apbs_fine_dime` before APBS runs and stops for any other number, without rounding.

## 7. Forces and excluded volume (`physics`)

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

Every flag accepts `true`, `false`, `yes`, `no`, `1` and `0` with any capitalization. Steps 7 and 8 describe the GPU engine. The CPU model of step 9 differs in several tags.

| tag | entries | effect |
|---|---|---|
| `soft_repulsion` | `true` or `false`, default `false` | soft wall in place of the overlap check (`model` `chain` applies bead soft repulsion) |
| `overlap_check` | `true` or `false` | rejection of moves into the receptor |
| `lj_forces` | `true` or `false` | Weeks Chandler Andersen (WCA) repulsion |
| `hydrodynamic_interactions` | `true` or `false` | Rotne Prager Yamakawa D(r) or constant D |
| `desolvation_alpha` | number ≥ 0 | factor on the Born desolvation energy |
| `enable_born2_torque` | `true` or `false` | desolvation of the receptor charges in the ligand Born field |
| `multipole_fallback` | `true` or `false` | receptor multipole field outside the grids |

### Excluded volume

The soft wall adds the energy V = kT (a/r)<sup>12</sup> with a = 0.9345 (r<sub>i</sub> + r<sub>j</sub>) for every receptor atom i and ligand atom j closer than 2a. The radii come from the PQR files. Atoms of radius 0 do not participate. The overlap check rejects a move bringing a ligand atom closer than 0.5 (r<sub>i</sub> + r<sub>j</sub>) to a receptor atom. The trajectory keeps the old pose and the clock still advances. The WCA repulsion is the force of the potential 4ε[(σ/r)<sup>12</sup> − (σ/r)<sup>6</sup>] below r = 2<sup>1/6</sup>σ, with σ = r<sub>i</sub> + r<sub>j</sub> and ε = 0.1 kcal/mol. The soft wall and the overlap check shorten the time step near the receptor (step 8).

| `soft_repulsion` | `overlap_check` | `lj_forces` | excluded volume |
|---|---|---|---|
| `true` | `true` or `false` | `false` | soft wall |
| `true` | `true` or `false` | `true` | error before the first step |
| `false` | `true` | `false` | overlap check |
| `false` | `true` | `true` | overlap check and WCA repulsion |
| `false` | `false` | `false` | none |
| `false` | `false` | `true` | error before the first step |

### Diffusion, desolvation and the far field

With `hydrodynamic_interactions` true the relative diffusion coefficient D(r) of two spheres with the radii of step 5 follows the Rotne Prager Yamakawa tensor. D(r) falls as the molecules approach. Each step adds the drift (dD/dr)Δt and the k<sub>b</sub> integral includes D(r). With `false` the relative diffusion coefficient stays at D<sub>rec</sub> + D<sub>lig</sub>.

The Born desolvation energy of a charge q in the Born field φ<sub>Born</sub> of the other molecule is α q<sup>2</sup> φ<sub>Born</sub> with α = `desolvation_alpha`. The value 1.0 is the base model. The value 0 switches desolvation off. A negative value stops the run at start. With α above 0 the desolvation acts on the ligand charges in the receptor Born field. `enable_born2_torque` true adds the receptor charges in the ligand Born field as a force and a torque on the ligand.

With `multipole_fallback` true the receptor field outside every electrostatic grid comes from a screened expansion of the receptor charges with monopole, dipole and quadrupole terms. The k<sub>b</sub> integral then holds the orientation average of the squared dipole and quadrupole terms for a charged ligand. With `false` the field outside the grids comes from the net receptor charge alone, leaving a neutral receptor without field outside the grids.

The trypsin example enables the soft wall, hydrodynamic interactions, the base desolvation in both directions and the multipole field. `overlap_check` false has no effect with the soft wall on.

## 8. Trajectories, time step and convergence (`simulation`)

```xml
<simulation>
  <n_trajectories>10000000</n_trajectories>
  <max_steps>1000000</max_steps>
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

| tag | entries | effect |
|---|---|---|
| `n_trajectories` | integer ≥ 1 | trajectories started on the b sphere |
| `max_steps` | integer ≥ 1 | step limit of each trajectory |
| `dt` | number (ps) | outer step of the flexible chain model (ignored by rigid body runs) |
| `minimum_core_dt` | number ≥ 0 (ps), 0 for none | floor of the time step |
| `minimum_core_reaction_dt` | number ≥ 0 (ps), default 0 | floor of the time step near a reaction surface in state machine runs |
| `max_dt` | number ≥ 0 (ps), 0 for none | ceiling of the time step |
| `temperature` | number (K) | kT of APBS, the diffusion coefficients and the forces |
| `seed` | integer ≥ 0 | seed of the random numbers |
| `convergence_check` | `true` or `false` | `bd_sims/convergence.json` after the run |
| `convergence_interval` | integer (percent), 0 for none | log lines with the running P<sub>rxn</sub> and k<sub>on</sub> |
| `convergence_tol` | number between 0 and 1 | relative standard error for the converged flag |
| `checkpoint_interval` | integer, 0 for none | finished trajectories between checkpoints |

The trypsin run starts 10,000,000 trajectories with a step limit of 10,000,000. `submit.sh` (4 GPUs by default) splits the run into four shards of 2,500,000 trajectories with the seeds `seed` + 11111111 i for i = 1 to 4.

### Time step

Rigid body runs apply an adaptive time step and ignore `dt`. At every step the GPU engine sets the step of each trajectory from the center distance and the net charges. Beyond 1.1b the distance to the start and escape spheres limits the step as well. The soft wall or the overlap check shortens the step near the receptor surface. `max_dt` then caps the step. `minimum_core_dt` comes last and raises the step to the floor over every other rule. The trypsin run has the floor 0.2 ps and no ceiling. No trypsin step is shorter than 0.2 ps, even at the receptor surface.

`minimum_core_reaction_dt` acts in state machine runs, with `state_machine_reactions` true in `bd_sims/input.xml` (default false, step 14). In a state machine run the step shrinks near a reachable reaction surface without dropping below `minimum_core_reaction_dt`. The trypsin run keeps `state_machine_reactions` false, leaving the trypsin value 0.05 inactive.

### Step limit

Each trajectory runs to a reaction, an escape or `max_steps` steps. A trajectory at the step limit counts as not reacted. `results.json` records the count `n_max_steps` and the bounds `k_on_bound_low` and `k_on_bound_high`. `k_on_bound_low` counts every trajectory at the limit as not reacted and `k_on_bound_high` as reacted. `python -m pystarc.analysis.step_limit bd_sims/trajectories.csv` reads a pilot run and proposes a `max_steps` with a change of k<sub>on</sub> of at most 1% from the trajectories at the limit.

### Convergence

After N trajectories the relative standard error of P<sub>rxn</sub> is √((1 − P<sub>rxn</sub>)/(N P<sub>rxn</sub>)). With `convergence_check` true the run writes `bd_sims/convergence.json` with P<sub>rxn</sub>, k<sub>on</sub>, the standard errors, the 95% Wilson intervals, the converged flag and the N for a relative error of 10, 5 and 1%. The flag is true for a relative standard error below `convergence_tol`. The trypsin value 0.05 flags convergence below a relative standard error of 5%.

`convergence_interval` prints the running P<sub>rxn</sub> and k<sub>on</sub> each time the count of finished trajectories passes a multiple of `convergence_interval` percent of `n_trajectories`. A progress line appears every 10 s once trajectories finish. None of the convergence tags stops a run. The run propagates every trajectory to a reaction, an escape or the step limit.

### Checkpoints

With `checkpoint_interval` above 0 the run writes `bd_sims/checkpoint.npz` each time `checkpoint_interval` more trajectories have finished. A stopped run resumes with `python /path/to/PySTARC/run_pystarc.py bd_sims/input.xml` in the run folder. `run.sh` and the SLURM scripts delete `bd_sims` and start over. The resumed run needs the same input files and the same simulation settings; `gpu_force_batch`, `n_threads`, `convergence_interval`, the `outputs` tags and a positive `checkpoint_interval` may change. A checkpoint of another run stops the run with an error. With `save_interval` above 0 the files `paths.npz` and `energetics.npz` of a resumed run hold the records written after the resume. A finished run deletes the checkpoint. The trypsin run writes no checkpoint.

### Temperature and seed

`temperature` sets kT in APBS, the desolvation grids, the diffusion coefficients and the forces. The viscosity stays at 0.890 mPa s of water at 25 °C. The run prints a warning for a temperature other than 298.15 K. A new `temperature` needs a matching `debye_length` (step 6).

The random numbers of a trajectory depend on `seed`, the trajectory number and the step count. `gpu_force_batch` and a resume leave the random numbers unchanged.

## 9. Hardware (`compute`)

```xml
<compute>
  <gpu>true</gpu>
  <n_gpus>1</n_gpus>
  <gpu_force_batch>0</gpu_force_batch>
  <n_threads>32</n_threads>
  <work_dir>bd_sims</work_dir>
</compute>
```

| tag | entries | effect |
|---|---|---|
| `gpu` | `true` or `false` | GPU engine for `true`, CPU model for `false` |
| `n_gpus` | integer ≥ 0, default 1 | GPU workers for parallel trajectory splitting (`--n-gpus N` overrides this tag) |
| `gpu_force_batch` | integer ≥ 0, 0 for automatic | trajectories per GPU force evaluation |
| `n_threads` | integer | worker processes of the CPU model or flexible chain shard |
| `work_dir` | folder path, default `bd_sims` | output folder for generated inputs, grids and trajectory data |

With `gpu` true and `n_gpus` 1 the GPU engine runs every trajectory on one GPU through CuPy. With `n_gpus` above 1 (or `--n-gpus N` on the command line), `run_pystarc.py` generates the inputs and APBS grids once in `work_dir`, splits `n_trajectories` across `work_dir/bd_1` to `work_dir/bd_N`, runs the shards in parallel across the visible GPUs, and pools the counts into `work_dir/results.json` and `work_dir/convergence.json`. The run stops before the first step when CuPy cannot execute a kernel.

`gpu_force_batch` splits the force evaluation of the running trajectories into batches and bounds the GPU memory. The value 0 picks max(1000, 4 × 1024<sup>3</sup>/(150 × ligand atoms)) trajectories per batch, capped at the count of running trajectories. The 18 ligand atoms and GHO in `bd_sims/ligand.pqr` set a batch of 1,507,006 trajectories. The batch size has no effect on the random numbers.

### CPU model

With `gpu` false on a rigid body system, the CPU model runs the trajectories in `n_threads` worker processes, at most one per CPU core and one per trajectory. A value of 1 or less runs one process. The CPU model differs from the GPU engine of steps 7 and 8 in the tags below.

| tag | CPU model |
|---|---|
| `soft_repulsion` | `true` stops a rigid body CPU run before the first step |
| `overlap_check`, `lj_forces` | no effect, a move with a ligand atom closer than r<sub>i</sub> + r<sub>j</sub> to a receptor atom is drawn again up to five times and the old pose stays after five failures |
| `enable_born2_torque` | no effect, both desolvation directions in every run |
| `multipole_fallback` | no effect, no receptor field outside the grids |
| `minimum_core_dt`, `max_dt` | no effect |
| `minimum_core_reaction_dt` | floor of the time step (0.05 ps for 0), a quarter of the floor at a center distance below 1.5 times the smallest reaction distance |
| `seed` | seed `seed` + i for trajectory i |
| `convergence_interval`, `checkpoint_interval` | no effect |
| `save_interval`, `full_paths`, `energetics` | no effect |

The rigid body CPU model prints P<sub>rxn</sub> and k<sub>on</sub> in the log and writes `bd_sims/convergence.json`, without writing `results.json`, `trajectories.csv`, `paths.npz` or `energetics.npz`.

The trypsin example runs on the GPU. `n_threads` 32 has no effect on the trypsin run.

## 10. Saved paths (`outputs`)

```xml
<outputs>
  <save_interval>100</save_interval>
  <full_paths>true</full_paths>
  <energetics>true</energetics>
</outputs>
```

| tag | entries | effect |
|---|---|---|
| `save_interval` | integer ≥ 0 | steps between records of every running trajectory |
| `full_paths` | `true` or `false`, default `true` | `bd_sims/paths.npz` |
| `energetics` | `true` or `false`, default `true` | `bd_sims/energetics.npz` |

Every `save_interval` steps the GPU run adds one record per running trajectory to `paths.npz` and `energetics.npz`.

| file | columns |
|---|---|
| `paths.npz` | traj_id, step, x, y, z, q0, q1, q2, q3, with the ligand center (Å) and the orientation quaternion |
| `energetics.npz` | traj_id, step, fx, fy, fz, dt, with the force on the ligand (kT/Å) and the time step (ps) |

`energetics.npz` holds no energies. With `save_interval` 0 the file `paths.npz` holds the start pose and the last pose of every trajectory, and no `energetics.npz` is written. The trypsin run records every 100 steps.

## 11. Flexible chain settings (`chain`)

Rigid body examples keep every tag inside `<chain>` blank:

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
  <target_grid_dx></target_grid_dx>
  <born_grid_dx></born_grid_dx>
</chain>
```

Populating `<chain>` tags when `model` is `rigid` stops the build with an error. Step 15 describes `<chain>` for `model` `chain`.

## 12. Build and run the simulation

`run_pystarc.py` executes input preparation, APBS and Born grid generation, and Brownian dynamics in one command:

```bash
python /path/to/PySTARC/run_pystarc.py config.xml
```

Passing `--setup` runs the input preparation step alone and stops before APBS and Brownian dynamics:

```bash
python /path/to/PySTARC/run_pystarc.py config.xml --setup
```

During the build step `run_pystarc.py` reads `config.xml` and writes five files into `work_dir` (`bd_sims` by default). A blank required tag, an incompatible tag for the chosen `model` or `mode`, or an unknown tag stops the build before any file is written. The build prepares `receptor.pqr` and `ligand.pqr` (step 3), picks the reaction pairs (step 4), and writes `input.xml` and `build_record.json` inside `bd_sims`. Intermediate files from AmberTools (`protein.pdb`, `protein.prmtop`, `ligand.mol2`, `leap.log`) are created and cleaned up inside `bd_sims`, leaving the run folder untouched. For the trypsin benzamidine example the build step prints four lines:

```
  receptor.pqr: 3220 atoms from receptor.pqr
  ligand.pqr: 18 atoms from ligand.pqr
  6 pairs, n_needed 3
  rmax=28.4 lmax=3.7 b=45 fglen=96  WARNING, b-surface outside the grid
```

| line | content |
|---|---|
| `receptor.pqr`, `ligand.pqr` | atom count of each PQR file (plus net charge Q when parameterized from `.pdb`) |
| `6 pairs, n_needed 3` | pairs in `rxns.xml` and `n_needed` |
| `rmax`, `lmax` | largest distance of a receptor atom and of a ligand atom from the centroid of the molecule (Å) |
| `b`, `fglen` | `bd_milestone_radius` and the edge of the fine grid (Å) |

The last line ends with OK for b + lmax < fglen/2 and with WARNING for b + lmax ≥ fglen/2 (step 6). The warning does not stop the build. For `apbs_fglen` 0 the build derives the fine edge with the formula of step 6. PDB parameterization prints additional lines.

| file in `bd_sims` | content |
|---|---|
| `receptor.pqr` | coordinates, charges and radii of the receptor atoms |
| `ligand.pqr` | coordinates, charges and radii of the ligand atoms |
| `rxns.xml` | one reaction with the atom pairs, the reaction distances and `n_needed` |
| `input.xml` | runtime `<pystarc>` settings generated from steps 5 to 11 |
| `build_record.json` | criterion, pairs and MD5 checksums of the four files above |

### Helper and SLURM scripts

| command | GPUs | actions |
|---|---|---|
| `bash run.sh` | one GPU of the current node | deletes `bd_sims`, runs `run_pystarc.py config.xml` |
| `sbatch submit.sh` | 1 node, 4 GPUs (default) | deletes `bd_sims`, splits into 4 shards via `srun`, pools into `bd_sims/results.json` |
| `sbatch --nodes=1 --ntasks-per-node=1 submit.sh` | 1 node, 1 GPU | deletes `bd_sims`, runs `run_pystarc.py config.xml --n-gpus 1` |
| `sbatch --nodes=2 --ntasks-per-node=4 submit.sh` | 2 nodes, 4 GPUs per node (8 total) | deletes `bd_sims`, splits into 8 shards across nodes via `srun`, pools into `bd_sims/results.json` |

Submit `submit.sh` from the run folder. The `#SBATCH` lines set the job name, partition, node/GPU count, memory, and time limit of the local cluster. Both `run.sh` and `submit.sh` delete `bd_sims` with every earlier result and checkpoint.

In a four GPU run (`--n-gpus 4` or `<n_gpus>4</n_gpus>`), `run_pystarc.py` builds the inputs and grids once in `bd_sims` and starts four shards in `bd_sims/bd_1` to `bd_sims/bd_4`. Each shard runs a quarter of `n_trajectories` with a seed of its own (step 8). `pystarc/multi_GPU/combine_data.py` then pools the counts of the finished shards into `bd_sims/results.json` and `bd_sims/convergence.json`. A shard without `results.json` stays out of the pool.

### Resuming or running from `bd_sims/input.xml`

A run from a checkpoint or with a hand edited `bd_sims/input.xml` invokes `run_pystarc.py` directly on `bd_sims/input.xml` without deleting `bd_sims`:

```bash
python /path/to/PySTARC/run_pystarc.py bd_sims/input.xml
```

The run resumes from `bd_sims/checkpoint.npz` when present (step 8) and reuses the grids and hydrodynamic radii of `bd_sims` for unchanged settings and PQR files. A four GPU run resumes with `python /path/to/PySTARC/run_pystarc.py bd_sims/bd_i/input.xml` in each unfinished shard, followed by `python /path/to/PySTARC/pystarc/multi_GPU/combine_data.py` in the run folder.

## 13. Results and output files

`run_pystarc.py` writes every output into `bd_sims`. Each start writes a log `bd_sims/pystarc_<date>_<time>.log` with the printout of the run. The log ends with the run summary and, with `convergence_check` `true`, the convergence analysis. An error during the run ends the log with the error message. An invalid entry of `config.xml` or `input.xml` stops the run before the log starts. A four GPU run keeps the per shard logs in `bd_sims/bd_1` to `bd_sims/bd_4`. The expected results of the trypsin benzamidine example are in `examples/trypsin_benzamidine/README.md`.

### results.json

`bd_sims/results.json` holds k<sub>on</sub> and the counts of the run. After a multi GPU run `bd_sims/results.json` holds the pool of the shards.

| field | content |
|---|---|
| `k_on` | association rate constant 6.022×10<sup>8</sup> k<sub>b</sub> P<sub>rxn</sub> (M⁻¹ s⁻¹) |
| `k_on_low`, `k_on_high` | Wilson interval of k<sub>on</sub> at `confidence_level` (M⁻¹ s⁻¹) |
| `P_rxn` | reaction probability, `n_reacted` over `n_trajectories` |
| `P_rxn_low`, `P_rxn_high` | Wilson interval of P<sub>rxn</sub> at `confidence_level` |
| `P_rxn_bound_low`, `P_rxn_bound_high` | P<sub>rxn</sub> with the trajectories at `max_steps` counted as not reacted and as reacted |
| `k_on_bound_low`, `k_on_bound_high` | k<sub>on</sub> with the trajectories at `max_steps` counted as not reacted and as reacted (M⁻¹ s⁻¹) |
| `k_b` | rate constant of diffusion from infinite separation to the start sphere (Å³/ps) |
| `D_rel` | relative translational diffusion coefficient D<sub>rec</sub> + D<sub>lig</sub> at infinite separation (Å²/ps) |
| `n_trajectories` | trajectories of the run |
| `n_reacted`, `n_escaped`, `n_max_steps` | trajectories ended by reaction, by escape and at `max_steps` |
| `confidence_level` | `confidence_interval` of `input.xml`, default 0.95 (step 14) |
| `r_start`, `r_escape` | radius b of the start sphere and radius of the escape sphere (Å) |
| `log10_k_on` | base 10 logarithm of `k_on`, written for `k_on` above 0 |
| `convergence_tol` | `convergence_tol` of step 8 |
| `wall_time_sec`, `steps_per_sec`, `total_steps` | wall time of the BD simulation (s), BD steps per second and BD steps of all trajectories |
| `contact_total_steps` | steps counted for `contact_frequency.csv` |
| `k_on_units`, `k_b_units`, `D_rel_units` | unit labels |
| `completed_reactions` | firing count of each reaction with `state_machine_reactions` `true` |

The factor 6.022×10<sup>8</sup> converts Å³/ps into M⁻¹ s⁻¹. A trajectory at `max_steps` counts as not reacted in P<sub>rxn</sub>. The low bounds therefore repeat P<sub>rxn</sub> and k<sub>on</sub>. The high bounds count every trajectory at `max_steps` as reacted (step 8). The Wilson intervals evaluate z = 1.96 at the confidence level 0.95. The pooled file of a multi GPU run has no `total_steps` and no `contact_total_steps`; `wall_time_sec` reports the slowest shard and `aggregate_gpu_sec` reports the summed wall time across shards (s).

### convergence.json

`run_pystarc.py` writes `bd_sims/convergence.json` with `convergence_check` `true` (step 8). `combine_data.py` writes the pooled `convergence.json` of a multi GPU run.

| field | content |
|---|---|
| `N`, `n_reacted`, `n_escaped`, `n_max_steps` | trajectory counts, no `n_max_steps` in the pooled file |
| `P_rxn`, `SE` | reaction probability and standard error √(P<sub>rxn</sub>(1 − P<sub>rxn</sub>)/N) |
| `relative_SE`, `relative_SE_pct` | SE/P<sub>rxn</sub> as a fraction and in percent |
| `k_on`, `SE_kon` | k<sub>on</sub> and standard error of k<sub>on</sub> (M⁻¹ s⁻¹) |
| `wilson_CI`, `wilson_CI_P` | Wilson interval of k<sub>on</sub> (M⁻¹ s⁻¹) and of P<sub>rxn</sub> |
| `converged` | `true` for 0 < P<sub>rxn</sub> < 1 and `relative_SE` below `convergence_tol` |
| `confidence_level`, `tol`, `tol_pct` | confidence level and `convergence_tol` as a fraction and in percent |
| `N_needed` | trajectories (1 − P<sub>rxn</sub>)/(P<sub>rxn</sub> r²) for a relative SE r of 10%, 5% and 1% |

### Files in bd_sims

| file | content |
|---|---|
| `pystarc_<date>_<time>.log` | printout of one start of `run_pystarc.py` |
| `input.xml`, `rxns.xml`, `build_record.json` | generated runtime input, reaction criterion and build checksums |
| `receptor.pqr`, `ligand.pqr` | copies centered on the atom centroid, with a GHO atom at the centroid |
| `receptor0.dx`, `receptor1.dx` | coarse and fine electrostatic grids of the receptor (step 6) |
| `receptor0_born.dx`, `receptor1_born.dx` | coarse and fine desolvation grids of the receptor |
| `ligand0.dx`, `ligand1.dx`, `ligand0_born.dx`, `ligand1_born.dx` | electrostatic and desolvation grids of the ligand |
| `receptor_grids.key`, `ligand_grids.key` | fingerprints of the ghost free PQR bytes and grid settings for grid reuse |
| `receptor.pqr.r_hydro_*.cache`, `ligand.pqr.r_hydro_*.cache` | Monte Carlo hydrodynamic radius of each molecule (step 5) |
| `results.json` | rate constant and counts |
| `convergence.json` | statistics of P<sub>rxn</sub> |
| `trajectories.csv` | outcome, step count, start pose and closest approach of each trajectory |
| `encounters.csv` | ligand position and orientation at each reaction |
| `near_misses.csv` | ligand position and orientation at the closest approach of each escaped trajectory |
| `bd_time_to_reaction.csv` | BD time inside the escape sphere of each reacted trajectory |
| `contact_frequency.csv` | steps and time of each reaction pair inside the pair cutoff |
| `pose_clusters.csv` | rotation angle of each reaction pose in up to 8 bins, written for more than 5 reactions |
| `radial_density.csv` | steps and time per radial shell |
| `angular_map.npz` | steps and time per angular bin |
| `milestone_flux.csv` | outward and inward crossings of 11 shells from b to the escape sphere |
| `transition_matrix.npz` | moves between 50 radial bins |
| `paths.npz`, `energetics.npz` | records every `save_interval` steps (step 10) |
| `checkpoint.npz` | state of a run with `checkpoint_interval` above 0, deleted at the end |
| `split.json`, `bd_1` to `bd_N` | split of a multi GPU run |

A multi GPU run writes each shard into `bd_sims/bd_1` to `bd_sims/bd_N` with a separate `input.xml`, log, `results.json` and output files. `bd_sims` holds the shared grids, `split.json` and the pooled `results.json`, `convergence.json` and output files.

## 14. Runtime `bd_sims/input.xml` overrides

`run_pystarc.py config.xml` generates `bd_sims/input.xml` (`<pystarc>`) from `config.xml`. Low level `<pystarc>` tags not exposed in `config.xml` can be set by running `python run_pystarc.py config.xml --setup`, editing `bd_sims/input.xml`, and starting `python run_pystarc.py bd_sims/input.xml`. Top level tags sit inside `<pystarc>` and per file switches sit inside `<outputs>`:

```xml
  <confidence_interval>0.99</confidence_interval>
  <outputs>
    <save_interval>100</save_interval>
    <full_paths>true</full_paths>
    <energetics>true</energetics>
    <near_misses_csv>false</near_misses_csv>
  </outputs>
```

| tag in `<pystarc>` | entries | effect |
|---|---|---|
| `apbs_ligand_dime` | 4k + 1 with k ≥ 1, default 129 | grid points per axis of the ligand grids |
| `brownian_bridge` | `true` or `false`, default `true` | `true` evaluates reactions along sub step bridges as well as step endpoints |
| `confidence_interval` | number between 0 and 1, default 0.95 | confidence level of the Wilson intervals |
| `state_machine_reactions` | `true` or `false`, default `false` | `true` fires each one pair reaction of `rxns.xml` from `state_before` to `state_after`, `false` reads one reaction with every pair |
| `relative_viscosity` | `1`, default `1` | viscosity relative to water at 25 °C |
| `trajectories_csv`, `encounters_csv`, `near_misses_csv`, `radial_density`, `angular_map`, `bd_time_to_reaction`, `contact_frequency`, `milestone_flux`, `transition_matrix`, `pose_clusters` | `true` or `false`, default `true` | `false` skips writing that output file in `bd_sims` |
| `results_json` | `true` or `false`, default `true` | `false` skips `results.json`, sets k<sub>on</sub> to 0 in `convergence.json` and skips multi GPU pooling |

`run_pystarc.py --template input.xml` prints a standalone `<pystarc>` template for low level engine runs without `config.xml`.

## 15. Flexible chain model (barnase and barstar)

The flexible chain model treats the ligand as a chain of coarse grained COFFDROP beads with internal motion and the receptor as a rigid body. The chain model runs on CPUs and needs no GPU. `examples/barnase_barstar/config.xml` follows the same nine section `<config>` schema as every rigid body example, with `model` set to `chain` and `<chain>` populated:

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
    <target_grid_dx>apbs_output/barnase1.dx</target_grid_dx>
    <born_grid_dx>apbs_output/barnase0_born.dx</born_grid_dx>
  </chain>
```

On a single workstation, `bash run.sh` (or `python ../../run_pystarc.py config.xml` once `apbs_output` exists, or with `target_grid_dx` and `born_grid_dx` left blank so APBS runs inside `bd_sims`) builds the chain inputs and runs the simulation. On a SLURM cluster, three commands run the 25 shard production workflow in order:

```bash
cd examples/barnase_barstar
python make_grids.py
./submit_shards.sh
python combine_shards.py
```

During the build step (`python ../../run_pystarc.py config.xml --setup`, called inside `submit_shards.sh`), `run_pystarc.py` parameterizes `receptor.pdb` with `tleap` (`ff14SB`) and writes a centered receptor PQR to `bd_sims/receptor.pqr`, builds `ligand.pdb` as 230 COFFDROP beads across 87 residues in `bd_sims/chain.json`, copies the 568 receptor atom and bead pairs of `reaction_pairs.json` into `bd_sims/reaction_pairs.json` (or builds `bd_sims/reaction_pairs.json` from `<criterion>` when `reaction_pairs_json` is blank), and writes `bd_sims/input.xml` and `bd_sims/build_record.json`.

`make_grids.py` builds `bd_sims/receptor.pqr` from `config.xml`, runs APBS with the `<electrostatics>` settings (`srad` 1.5 Å, `apbs_fglen` 140.0 Å, `apbs_dime` 449), and writes `barnase0.dx`, `barnase1.dx`, `barnase0_born.dx` and `barnase1_born.dx` into `apbs_output`. `examples/barnase_barstar/README.md` submits `make_grids.py` as a SLURM job with 128 GB of memory.

`submit_shards.sh` creates `shards/shard_01` to `shards/shard_25` (`NSHARD=25` by default), links `apbs_output`, copies `bd_sims/input.xml` into each shard folder with seed `11111111 * i + 1`, and submits one 96 CPU SLURM job per shard running `python $RUNNER input.xml`. Each shard writes `results.json`, `trajectories.csv`, saved paths and a log into `bd_sims` of that shard.

`combine_shards.py` reads `results.json` from every finished shard, prints the counts per shard, and prints the pooled k<sub>on</sub> = k<sub>b</sub> P<sub>rxn</sub> with a Wilson 95% interval, where P<sub>rxn</sub> = n<sub>reacted</sub>/(n<sub>reacted</sub> + n<sub>escaped</sub>) summed across shards.

Each trajectory starts with the chain centroid on the sphere of radius b = `bd_milestone_radius` around the receptor centroid in a random orientation. Each outer step of `dt` moves the chain as a rigid body, while `chain_steps_per_outer` internal steps of `dt_chain` move the beads within the chain. Beyond 1.1 b an outer propagator returns the chain to the b sphere or ends the trajectory as escaped.

| tag in `<chain>` (and companion sections) | entries | effect |
|---|---|---|
| `receptor`, `ligand`, `model` (`<structure>`) | `.pdb` receptor and `.pdb` protein chain, `chain` | receptor charges and radii, chain sequence and bead positions |
| `mode`, `receptor`, `ligand`, `pair_cutoff`, `n_needed` (`<criterion>`) | `contacts`, optional target lists, distance (Å) and count | reaction distance per pair and simultaneous pairs needed |
| `reaction_pairs_json` | JSON path, blank to build from `<criterion>` | pre-built `[rec_atom_idx, bead_idx, cutoff]` pairs |
| `dt` (`<simulation>`) | `chain_steps_per_outer` × `dt_chain` (ps) | outer step, shortened to 0.05 ps when a pair is closer than 1.5 × `pair_cutoff` |
| `dt_chain` | time > 0 (ps) | internal step of the beads |
| `chain_steps_per_outer` | integer ≥ 1 | internal steps per outer step of `dt` |
| `n_equilibration_steps` | integer ≥ 0 | internal steps without receptor forces before each trajectory, 0 for none |
| `auto_diffusion` | `true` or `false` | `true` rebuilds chain diffusion tensors from bead geometry every 10 outer steps, `false` reads `d_trans` and `d_rot` |
| `d_trans`, `d_rot` | > 0 (Å<sup>2</sup>/ps), > 0 (rad<sup>2</sup>/ps) | chain diffusion coefficients when `auto_diffusion` is `false` |
| `soft_repulsion` (`<physics>`), `soft_repulsion_eps` | `true` or `false`, energy ≥ 0 (kT) | repulsion 4ε[(σ/r)<sup>12</sup> − (σ/r)<sup>6</sup>] + ε for r < 2<sup>1/6</sup>σ between beads and receptor atoms and nonbonded bead pairs |
| `target_grid_dx` | DX path, blank to build in `work_dir` | receptor APBS fine potential grid acting on bead charges |
| `born_grid_dx` | DX path, blank to build in `work_dir` | receptor Born desolvation grid acting on bead charges, scaled by `desolvation_alpha` |
| `n_threads` (`<compute>`) | integer ≥ 1 | worker processes per run or shard |
