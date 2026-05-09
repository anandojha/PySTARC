"""COFFDROP chain BD: minimal example showing end-to-end usage.

Builds a small peptide, exercises the full force pipeline, and runs
a short deterministic Euler integration to demonstrate stability.
This is intended as a reference for new users.

Run:
    python examples/coffdrop_chain_intro.py
"""

import numpy as np
from pystarc.simulation.coffdrop_chain import (
    chain_from_sequence,
    ChainState,
    compute_chain_forces,
)


def main():
    print("=" * 60)
    print("PySTARC: COFFDROP chain BD intro example")
    print("=" * 60)

    # 1. Build a chain from a sequence string.
    #    Single-letter codes ARWGL = ALA-ARG-TRP-GLY-LEU
    sequence = "ARWGL"
    print(f"\nBuilding chain from sequence: '{sequence}'")
    chain = chain_from_sequence(sequence)

    print(f"\nChain topology:")
    print(f"  Name:       {chain.name}")
    print(f"  n_atoms:    {chain.n_atoms}")
    print(f"  bonds:      {len(chain.bonds)}")
    print(f"  angles:     {len(chain.angles)}")
    print(f"  torsions:   {len(chain.torsions)}")
    print(f"  pair_lookups: {len(chain.pair_lookups)}")

    print(f"\nAtoms:")
    for i, a in enumerate(chain.atoms):
        print(f"  [{i:3d}] {a.resname:12s} charge={a.charge:+.2f} radius={a.radius:.2f}")

    # 2. Place atoms at a starting geometry. The helper handles the
    #    geometry construction (CAs along x-axis at proper spacing,
    #    sidechains projected with non-degenerate tilts) so users
    #    don't need to roll their own placement.
    print(f"\nPlacing atoms at starting geometry...")
    from pystarc.simulation.coffdrop_chain import place_relaxed_geometry
    positions = place_relaxed_geometry(chain)

    # 3. Compute forces using full COFFDROP physics (bonds + angles + torsions
    #    + non-bonded pairs).
    state = ChainState.from_template(chain, positions)
    compute_chain_forces(state)
    max_F = float(np.max(np.abs(state.forces)))
    sum_F = float(np.linalg.norm(state.forces.sum(axis=0)))
    print(f"\nInitial forces:")
    print(f"  max |F|: {max_F:.3f} kBT/A")
    print(f"  sum |F|: {sum_F:.2e} (Newton's 3rd law check; should be ~0)")

    # 4. Run a short deterministic integration to demonstrate stability.
    print(f"\nRunning 50 deterministic Euler steps (dt=1e-5, damping=0.5)...")
    dt = 1e-5
    damping = 0.5
    forces_history = [max_F]
    for step in range(50):
        compute_chain_forces(state)
        state.positions += damping * state.forces * dt
        forces_history.append(float(np.max(np.abs(state.forces))))

    print(f"  Initial max |F|: {forces_history[0]:.3f} kBT/A")
    print(f"  Final   max |F|: {forces_history[-1]:.3f} kBT/A")
    print(f"  All positions finite: {np.all(np.isfinite(state.positions))}")

    print()
    print("=" * 60)
    print("Done. To use this in your own simulation, replace the above")
    print("Euler integration with ChainBDSimulator from chain_simulator.py")
    print("for proper Brownian dynamics with thermal noise.")
    print("=" * 60)


if __name__ == "__main__":
    main()
