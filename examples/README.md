# Examples

<table width="100%">
<thead><tr><th align="left">Complex</th><th align="left">Description</th><th align="left" nowrap>Category</th></tr></thead>
<tbody>
<tr><td><code>two_charged_spheres/</code></td><td>Two oppositely charged spheres</td><td nowrap>Analytical validation</td></tr>
<tr><td><code>trypsin_benzamidine/</code></td><td>Trypsin and benzamidine</td><td nowrap>Protein-ligand complex</td></tr>
<tr><td><code>beta_cyclodextrin_guests/</code></td><td>β-cyclodextrin host-guest complexes</td><td nowrap>Host-guest complexes</td></tr>
<tr><td><code>thrombin_thrombomodulin/</code></td><td>Thrombin and thrombomodulin</td><td nowrap>Protein-protein complex</td></tr>
<tr><td><code>p38_mapk_sb203580/</code></td><td>p38 MAPK with SB203580 inhibitor</td><td nowrap>Protein-ligand complex</td></tr>
<tr><td><code>carbonic_anhydrase_inhibitors/</code></td><td>Sulfonamide inhibitors across 3 isozymes</td><td nowrap>Protein-ligand complex</td></tr>
<tr><td><code>hsp90_inhibitors/</code></td><td>Heat shock protein 90 inhibitors</td><td nowrap>Protein-ligand complex</td></tr>
<tr><td><code>ttk_inhibitors/</code></td><td>Threonine tyrosine kinase inhibitors</td><td nowrap>Protein-ligand complex</td></tr>
<tr><td><code>barnase_barstar/</code></td><td>Barnase-barstar protein complex</td><td nowrap>Protein-protein complex</td></tr>
</tbody>
</table>

Each directory contains a README file summarizing its input files, execution commands, and output files.

## Running a copy outside the repository
The scripts locate PySTARC by searching parent folders. For a copy stored elsewhere, export the repository path once:
```
export PYSTARC_ROOT=~/PySTARC
```
