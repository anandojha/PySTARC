"""
Command-line interface for PySTARC.
"""

from __future__ import annotations
from pystarc.simulation.chain_simulator import ChainBDParameters, ChainBDSimulator
from pystarc.simulation.nam_simulator import NAMSimulator, NAMParameters
from pystarc.pipeline.chain_output_writer import write_chain_results
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
from pystarc.structures.chain_io import load_chain_from_json
from pystarc.xml_io.simulation_io import parse_reaction_xml
from pystarc.forces.electrostatic.grid_force import DXGrid
from pystarc.simulation.nam_simulator import zero_force
from pystarc.structures.pqr_io import parse_pqr
from pystarc.aux.aux_tools import bounding_box
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import click
import sys


@click.group()
@click.version_option(package_name="pystarc")
def cli():
    """PySTARC - Python Simulation Toolkit for Association Rate Constants"""
    pass


# Run a NAM (Northrup-Allison-McCammon) Brownian dynamics simulation.
@cli.command("nam_simulation")
@click.option("--mol1", required=True, help="PQR file for molecule 1")
@click.option("--mol2", required=True, help="PQR file for molecule 2")
@click.option("--rxn", required=True, help="Reaction XML file")
@click.option("--n", default=1000, show_default=True, help="Number of trajectories")
@click.option("--dt", default=0.2, show_default=True, help="Time step (ps)")
@click.option("--r-start", default=100.0, show_default=True, help="Start radius (Å)")
@click.option("--dx", multiple=True, help="APBS .dx grid file(s)")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--verbose", is_flag=True, help="Print progress")
@click.option("--output", default="results.xml", help="Output XML file")
def nam_simulation(mol1, mol2, rxn, n, dt, r_start, dx, seed, verbose, output):
    """Run a NAM Brownian dynamics simulation."""
    click.echo(f"Loading molecules …")
    m1 = parse_pqr(mol1)
    m2 = parse_pqr(mol2)
    click.echo(f"  mol1: {m1}")
    click.echo(f"  mol2: {m2}")
    pathways = parse_reaction_xml(rxn)
    click.echo(f"  reactions: {pathways}")
    # Build the mobility tensor from the bounding radii of the two molecules.
    r1 = m1.bounding_radius()
    r2 = m2.bounding_radius()
    mobility = MobilityTensor.from_radii(r1, r2)
    click.echo(f"  mobility: {mobility}")
    # Load the APBS .dx electrostatic grids, if any were supplied.
    grids = []
    for dx_file in dx:
        g = DXGrid.from_file(dx_file)
        grids.append(g)
        click.echo(f"  loaded grid: {g}")
    # Build the force function the simulation will call at each step. If grids
    # are present, the force and energy come from the screened-Coulomb
    # interaction of molecule 2's charges with the target's grid potential.
    # Otherwise the molecules feel no external force and move by pure diffusion.
    if grids:

        def force_fn(mol_1, mol_2):
            force = np.zeros(3)
            torque = np.zeros(3)
            energy = 0.0
            for grid in grids:
                for atom in mol_2.atoms:
                    if abs(atom.charge) < 1e-9:
                        continue
                    f = grid.force_on_charge(atom.position, atom.charge)
                    force += f
                    energy += grid.interpolate(atom.position) * atom.charge
                    # The torque is the cross product r × f, where r is the
                    # atom position relative to the molecule centroid and f is
                    # the force on the atom.
                    r = atom.position - mol_2.centroid()
                    torque += np.cross(r, f)
            return force, torque, energy

    else:
        force_fn = zero_force
    params = NAMParameters(
        n_trajectories=n,
        dt=dt,
        r_start=r_start,
        seed=seed,
        verbose=verbose,
    )
    sim = NAMSimulator(m1, m2, mobility, pathways, params, force_fn)
    click.echo(f"\nRunning {n} trajectories …")
    result = sim.run()
    click.echo(f"\n{'-'*50}")
    click.echo(f"Results:")
    click.echo(f"  Reacted : {result.n_reacted}")
    click.echo(f"  Escaped : {result.n_escaped}")
    click.echo(f"  P(rxn)  : {result.reaction_probability:.4f}")
    D_rel = mobility.relative_translational_diffusion()
    k = result.rate_constant(D_rel)
    click.echo(f"  k_assoc : {k:.3e} M⁻¹s⁻¹")
    click.echo(f"{'-'*50}")


# Print the bounding box of a PQR molecule.
@cli.command("bounding_box")
@click.argument("pqr_file")
@click.option("--padding", default=5.0, show_default=True, help="Padding in Å")
def bounding_box_cmd(pqr_file, padding):
    """Print bounding box of a PQR molecule."""
    mol = parse_pqr(pqr_file)
    bb = bounding_box(mol, padding)
    click.echo(f"Bounding box for {pqr_file}:")
    click.echo(f"  x: [{bb.xmin:.3f}, {bb.xmax:.3f}]")
    click.echo(f"  y: [{bb.ymin:.3f}, {bb.ymax:.3f}]")
    click.echo(f"  z: [{bb.zmin:.3f}, {bb.zmax:.3f}]")
    click.echo(f"  center: {bb.center}")
    click.echo(f"  size:   {bb.size}")


# Convert a PQR file to the PySTARC molecule XML format.
@cli.command("pqr_to_xml")
@click.argument("pqr_file")
@click.option("--output", "-o", default=None, help="Output XML file")
def pqr_to_xml(pqr_file, output):
    """Convert a PQR file to PySTARC molecule XML format."""
    mol = parse_pqr(pqr_file)
    root = ET.Element("molecule", name=mol.name)
    for a in mol.atoms:
        ET.SubElement(
            root,
            "atom",
            index=str(a.index),
            name=a.name,
            resname=a.residue_name,
            resid=str(a.residue_index),
            x=f"{a.x:.4f}",
            y=f"{a.y:.4f}",
            z=f"{a.z:.4f}",
            charge=f"{a.charge:.4f}",
            radius=f"{a.radius:.4f}",
        )
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    out = output or (Path(pqr_file).stem + ".xml")
    tree.write(out, encoding="unicode", xml_declaration=True)
    click.echo(f"Written: {out}  ({len(mol.atoms)} atoms)")


def main():
    cli()


# Run a flexible-chain Brownian dynamics simulation against a rigid target.
@cli.command("chain_simulation")
@click.option(
    "--chain",
    "chain_file",
    required=True,
    help="JSON file with chain topology and initial body positions",
)
@click.option("--target", required=True, help="PQR file for the rigid target molecule")
@click.option("--rxn", required=True, help="Reaction XML file")
@click.option(
    "--dx", multiple=True, help="APBS .dx grid file(s) for the target's PB potential"
)
@click.option("--n", default=1000, show_default=True, help="Number of trajectories")
@click.option(
    "--dt", default=0.2, show_default=True, help="Outer rigid-body timestep (ps)"
)
@click.option(
    "--dt-chain",
    default=0.05,
    show_default=True,
    help="Inner internal-coordinate timestep (ps)",
)
@click.option(
    "--chain-steps-per-outer",
    default=4,
    show_default=True,
    help="Number of inner steps per outer step",
)
@click.option(
    "--r-start",
    default=100.0,
    show_default=True,
    help="b-sphere starting radius (Angstroms)",
)
@click.option(
    "--r-escape", default=0.0, help="Escape radius (A); if 0, set to r_start * 1.1"
)
@click.option(
    "--d-trans",
    "d_trans",
    default=None,
    type=float,
    help="Chain translational diffusion coefficient (A^2/ps). "
    "Required unless --auto-diffusion is set.",
)
@click.option(
    "--d-rot",
    "d_rot",
    default=None,
    type=float,
    help="Chain rotational diffusion coefficient (rad^2/ps). "
    "Required unless --auto-diffusion is set.",
)
@click.option(
    "--max-steps",
    default=1_000_000,
    show_default=True,
    help="Hard cap on outer steps per trajectory",
)
@click.option(
    "--threads",
    "n_threads",
    default=1,
    show_default=True,
    help="Number of parallel worker processes (1 = serial)",
)
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--verbose", is_flag=True, help="Print progress")
@click.option(
    "--output-dir",
    "-o",
    "output_dir",
    default="chain_bd_results",
    show_default=True,
    help="Directory to write results.json and trajectories.csv. " "Created if missing.",
)
@click.option(
    "--auto-diffusion",
    "auto_diffusion",
    is_flag=True,
    default=False,
    help="Compute the chain's translational and rotational diffusion "
    "tensors automatically from its body geometry using full "
    "Rotne-Prager hydrodynamics. When set, --d-trans and "
    "--d-rot must NOT be supplied.",
)
def chain_simulation(
    chain_file,
    target,
    rxn,
    dx,
    n,
    dt,
    dt_chain,
    chain_steps_per_outer,
    r_start,
    r_escape,
    d_trans,
    d_rot,
    max_steps,
    n_threads,
    seed,
    verbose,
    output_dir,
    auto_diffusion,
):
    """Run a chain Brownian dynamics simulation against a rigid target.

    The flexible chain is loaded from a JSON file giving its atoms, bonds,
    angles, torsions, and initial body-frame positions, and the target is
    loaded from a PQR file. The target's electrostatic environment may be
    supplied through one or more APBS .dx grids. Without them the chain feels
    only its internal bonded forces and moves by pure diffusion.
    """
    # Validate the diffusion-coefficient mode before doing any work, so the
    # user sees the error immediately rather than after reading the inputs.
    if auto_diffusion:
        if d_trans is not None or d_rot is not None:
            raise click.UsageError(
                "--auto-diffusion cannot be combined with --d-trans or --d-rot. "
                "Either set --auto-diffusion (compute D from chain geometry) "
                "or pass --d-trans and --d-rot explicitly."
            )
    else:
        if d_trans is None or d_rot is None:
            raise click.UsageError(
                "Both --d-trans and --d-rot are required (unless --auto-diffusion "
                "is set, which computes the diffusion tensors from chain geometry)."
            )

    click.echo("Loading inputs ...")
    chain_template, body_positions = load_chain_from_json(chain_file)
    click.echo(
        f"  chain   : {chain_template.name} "
        f"({len(chain_template.atoms)} atoms, "
        f"{len(chain_template.bonds)} bonds, "
        f"{len(chain_template.angles)} angles, "
        f"{len(chain_template.torsions)} torsions)"
    )

    target_mol = parse_pqr(target)
    click.echo(f"  target  : {target_mol}")

    pathways = parse_reaction_xml(rxn)
    click.echo(f"  reactions: {pathways}")

    grids = []
    for dx_file in dx:
        g = DXGrid.from_file(dx_file)
        grids.append(g)
        click.echo(f"  loaded grid: {g}")
    if len(grids) > 1:
        click.echo("  warning: chain_simulation currently uses only the first DX grid")
    target_grid = grids[0] if grids else None

    params = ChainBDParameters(
        n_trajectories=n,
        dt=dt,
        dt_chain=dt_chain,
        chain_steps_per_outer=chain_steps_per_outer,
        max_steps=max_steps,
        r_start=r_start,
        r_escape=r_escape,
        seed=seed,
        n_threads=n_threads,
        verbose=verbose,
    )
    if auto_diffusion:
        click.echo(
            "  hydrodynamics: full Rotne-Prager (auto-computing D from "
            "chain geometry)"
        )
        sim = ChainBDSimulator(
            target=target_mol,
            chain_template=chain_template,
            chain_init_body_positions=body_positions,
            params=params,
            pathway_set=pathways,
            target_grid=target_grid,
            auto_diffusion=True,
        )
        click.echo(f"  D_trans diag (A^2/ps): {np.diag(sim.D_trans)}")
        click.echo(f"  D_rot   diag (rad^2/ps): {np.diag(sim.D_rot)}")
    else:
        click.echo(
            f"  hydrodynamics: scalar isotropic " f"(D_trans={d_trans}, D_rot={d_rot})"
        )
        sim = ChainBDSimulator(
            target=target_mol,
            chain_template=chain_template,
            chain_init_body_positions=body_positions,
            params=params,
            pathway_set=pathways,
            D_trans=d_trans,
            D_rot=d_rot,
            target_grid=target_grid,
        )

    click.echo(f"\nRunning {n} trajectories ...")
    import time as _time

    _t0 = _time.time()
    results = sim.run()
    _wall = _time.time() - _t0
    click.echo(f"  done: {len(results)} trajectories in {_wall:.2f}s")
    # Write the summary results and the per-trajectory CSV.
    output_path = Path(output_dir)
    written = write_chain_results(
        output_path,
        sim,
        results,
        wall_time_sec=_wall,
    )
    click.echo(f"\nOutput files in {output_path}/:")
    for name, _ in written:
        click.echo(f"  {name}")

    click.echo(f"\n{'-' * 50}")
    click.echo("Results:")
    click.echo(f"  Reacted : {sim.n_reacted}")
    click.echo(f"  Escaped : {sim.n_escaped}")
    if (sim.n_reacted + sim.n_escaped) > 0:
        p_rxn = sim.n_reacted / (sim.n_reacted + sim.n_escaped)
        click.echo(f"  P(rxn)  : {p_rxn:.4f}")
    click.echo(f"{'-' * 50}")


if __name__ == "__main__":
    cli()
