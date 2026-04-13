"""
Command-line interface for PySTARC.
"""

from __future__ import annotations
from pystarc.simulation.nam_simulator import NAMSimulator, NAMParameters
from pystarc.hydrodynamics.rotne_prager import MobilityTensor
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


# nam_simulation
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
    # Mobility from bounding radii
    r1 = m1.bounding_radius()
    r2 = m2.bounding_radius()
    mobility = MobilityTensor.from_radii(r1, r2)
    click.echo(f"  mobility: {mobility}")
    # Load DX grids if provided
    grids = []
    for dx_file in dx:
        g = DXGrid.from_file(dx_file)
        grids.append(g)
        click.echo(f"  loaded grid: {g}")
    # Build force function
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
                    # torque = r × f
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


# bounding_box
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


# pqr_to_xml
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


if __name__ == "__main__":
    main()
