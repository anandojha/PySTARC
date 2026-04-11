"""
PySTARC output system tests

Tests for:
- OutputConfig dataclass and defaults
- XML parsing of <outputs> block
- output_writer file generation and content validation
- results.json schematic
- CSV file structure
- NPZ file structure
- Edge cases (zero reactions, all escaped, etc.)
"""

from pystarc.simulation.gpu_batch_simulator import GPUBatchResult
from pystarc.pipeline.input_parser import PySTARCConfig
from pystarc.pipeline.input_parser import OutputConfig
from pystarc.pipeline.output_writer import write_all
from pystarc.pipeline.input_parser import parse
from dataclasses import fields
from pathlib import Path
import numpy as np
import pytest
import json
import math
import csv

# OutputConfig tests
class TestOutputConfig:
    """Test the OutputConfig dataclass."""

    def test_all_defaults_true(self):
        oc = OutputConfig()
        for f in fields(oc):
            if f.type is bool:
                assert getattr(oc, f.name) is True, f"{f.name} should default True"

    def test_save_interval_default(self):
        oc = OutputConfig()
        assert oc.save_interval == 10

    def test_custom_save_interval(self):
        oc = OutputConfig(save_interval=100)
        assert oc.save_interval == 100

    def test_disable_heavy_outputs(self):
        oc = OutputConfig(full_paths=False, energetics=False)
        assert oc.full_paths is False
        assert oc.energetics is False
        assert oc.results_json is True

    def test_field_count(self):
        oc = OutputConfig()
        # 14 bool flags + 1 int save_interval = 15 fields
        assert len(fields(oc)) == 15

    def test_pystarc_config_has_outputs(self):
        cfg = PySTARCConfig()
        assert cfg.outputs is not None
        assert cfg.outputs.results_json is True
        assert cfg.outputs.save_interval == 10

# XML parsing tests
class TestOutputXMLParsing:
    """Test parsing <outputs> block from XML."""

    def _write_xml(self, tmp_path, outputs_block=""):
        xml = f"""<?xml version="1.0" ?>
<pystarc>
  <receptor_pqr>r.pqr</receptor_pqr>
  <ligand_pqr>l.pqr</ligand_pqr>
  <n_trajectories>100</n_trajectories>
  {outputs_block}
</pystarc>"""
        p = tmp_path / "test.xml"
        p.write_text(xml)
        return p

    def test_no_outputs_block_uses_defaults(self, tmp_path):
        p = self._write_xml(tmp_path)
        cfg = parse(p)
        assert cfg.outputs.results_json is True
        assert cfg.outputs.full_paths is True
        assert cfg.outputs.save_interval == 10

    def test_disable_paths(self, tmp_path):
        p = self._write_xml(tmp_path, """
  <outputs>
    <full_paths>false</full_paths>
  </outputs>""")
        cfg = parse(p)
        assert cfg.outputs.full_paths is False
        assert cfg.outputs.results_json is True  # other defaults unchanged

    def test_custom_save_interval(self, tmp_path):
        p = self._write_xml(tmp_path, """
  <outputs>
    <save_interval>50</save_interval>
  </outputs>""")
        cfg = parse(p)
        assert cfg.outputs.save_interval == 50

    def test_disable_multiple(self, tmp_path):
        p = self._write_xml(tmp_path, """
  <outputs>
    <full_paths>false</full_paths>
    <energetics>false</energetics>
    <transition_matrix>false</transition_matrix>
    <save_interval>1</save_interval>
  </outputs>""")
        cfg = parse(p)
        assert cfg.outputs.full_paths is False
        assert cfg.outputs.energetics is False
        assert cfg.outputs.transition_matrix is False
        assert cfg.outputs.save_interval == 1
        assert cfg.outputs.trajectories_csv is True

    def test_yes_true_1_all_work(self, tmp_path):
        for val in ['true', 'True', 'TRUE', 'yes', 'Yes', '1']:
            p = self._write_xml(tmp_path, f"""
  <outputs>
    <full_paths>{val}</full_paths>
  </outputs>""")
            cfg = parse(p)
            assert cfg.outputs.full_paths is True, f"'{val}' should parse as True"

    def test_false_no_0_all_work(self, tmp_path):
        for val in ['false', 'False', 'FALSE', 'no', 'No', '0']:
            p = self._write_xml(tmp_path, f"""
  <outputs>
    <full_paths>{val}</full_paths>
  </outputs>""")
            cfg = parse(p)
            assert cfg.outputs.full_paths is False, f"'{val}' should parse as False"

# Output writer tests
def _make_dummy_data(N=100, n_react=45, n_escape=55, n_pairs=3):
    """Create realistic dummy simulation data."""
    outcome = np.array([1]*n_react + [2]*n_escape)
    return {
        'outcome':           outcome,
        'n_steps':           np.random.randint(100, 1000, N),
        'start_pos':         np.random.randn(N, 3) * 10,
        'start_q':           np.random.randn(N, 4),
        'min_dist':          np.random.uniform(2, 20, N),
        'step_at_min':       np.random.randint(0, 500, N),
        'total_time_ps':     np.random.uniform(10, 1000, N),
        'n_returns':         np.random.randint(0, 5, N),
        'bb_triggered':      np.random.randint(0, 2, N),
        'encounter_pos':     np.random.randn(n_react, 3),
        'encounter_q':       np.random.randn(n_react, 4),
        'encounter_traj':    np.arange(n_react, dtype=np.int64),
        'encounter_step':    np.random.randint(100, 500, n_react).astype(np.int64),
        'encounter_n_pairs': np.full(n_react, n_pairs, dtype=np.int64),
        'near_miss_pos':     np.random.randn(n_escape, 3),
        'near_miss_q':       np.random.randn(n_escape, 4),
        'near_miss_traj':    np.arange(n_react, N, dtype=np.int64),
        'near_miss_dist':    np.random.uniform(3, 15, n_escape),
        'path_steps':        [np.random.randn(50, 8) for _ in range(5)],
        'energy_steps':      [np.random.randn(50, 6) for _ in range(5)],
        'radial_bins':       np.linspace(0, 24, 201),
        'radial_counts':     np.random.randint(0, 100, 200),
        'angular_theta':     np.linspace(0, np.pi, 36),
        'angular_phi':       np.linspace(0, 2*np.pi, 72),
        'angular_counts':    np.random.randint(0, 50, (36, 72)),
        'milestone_radii':   np.linspace(10, 20, 11),
        'milestone_flux_out': np.random.randint(0, 500, 11),
        'milestone_flux_in': np.random.randint(0, 500, 11),
        'contact_pair_counts': np.random.randint(0, 1000, n_pairs),
        'contact_total_steps': 50000,
        'trans_bins':        np.linspace(0, 24, 51),
        'trans_matrix':      np.random.randint(0, 100, (50, 50)),
    }

def _make_result(n_react=45, n_escape=55):
    return GPUBatchResult(
        n_trajectories=n_react+n_escape, n_reacted=n_react, n_escaped=n_escape,
        n_max_steps=0, reaction_counts={'stage_0': n_react},
        r_start=10.0, r_escape=20.0, dt=0.2,
        elapsed_sec=5.0, steps_per_sec=100000
    )

class TestResultsJSON:
    """Test results.json output."""

    def test_file_created(self, tmp_path):
        result = _make_result()
        data = _make_dummy_data()
        write_all(tmp_path, result, data, OutputConfig(), k_b=57.47, D_rel=0.434)
        assert (tmp_path / 'results.json').exists()

    def test_json_parseable(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        data = json.loads((tmp_path / 'results.json').read_text())
        assert isinstance(data, dict)

    def test_required_fields(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        data = json.loads((tmp_path / 'results.json').read_text())
        required = ['k_on', 'k_on_low', 'k_on_high', 'P_rxn', 'P_rxn_low',
                     'P_rxn_high', 'k_b', 'D_rel', 'n_trajectories', 'n_reacted',
                     'n_escaped', 'wall_time_sec', 'steps_per_sec']
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_k_on_positive(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        data = json.loads((tmp_path / 'results.json').read_text())
        assert data['k_on'] > 0
        assert data['k_on_low'] <= data['k_on'] <= data['k_on_high']

    def test_log10_present(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        data = json.loads((tmp_path / 'results.json').read_text())
        assert 'log10_k_on' in data
        assert abs(data['log10_k_on'] - math.log10(data['k_on'])) < 1e-6

    def test_disabled(self, tmp_path):
        oc = OutputConfig(results_json=False)
        write_all(tmp_path, _make_result(), _make_dummy_data(), oc,
                  k_b=57.47, D_rel=0.434)
        assert not (tmp_path / 'results.json').exists()

class TestTrajectoriesCSV:
    """Test trajectories.csv output."""

    def test_file_created(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        assert (tmp_path / 'trajectories.csv').exists()

    def test_correct_row_count(self, tmp_path):
        write_all(tmp_path, _make_result(45, 55), _make_dummy_data(100, 45, 55),
                  OutputConfig(), k_b=57.47, D_rel=0.434)
        with open(tmp_path / 'trajectories.csv') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 100

    def test_outcome_values(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        with open(tmp_path / 'trajectories.csv') as f:
            rows = list(csv.DictReader(f))
        outcomes = {r['outcome'] for r in rows}
        assert outcomes <= {'reacted', 'escaped', 'max_steps', 'running'}
        reacted = sum(1 for r in rows if r['outcome'] == 'reacted')
        assert reacted == 45

    def test_columns_present(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        with open(tmp_path / 'trajectories.csv') as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        expected = ['traj_id', 'outcome', 'n_steps', 'start_x', 'start_y', 'start_z',
                    'start_q0', 'start_q1', 'start_q2', 'start_q3',
                    'min_distance', 'step_at_min', 'total_time_ps',
                    'n_returns', 'bb_triggered']
        for c in expected:
            assert c in cols, f"Missing column: {c}"

class TestEncountersCSV:
    """Test encounters.csv output."""

    def test_file_created_when_reactions(self, tmp_path):
        write_all(tmp_path, _make_result(10, 90), _make_dummy_data(100, 10, 90),
                  OutputConfig(), k_b=57.47, D_rel=0.434)
        assert (tmp_path / 'encounters.csv').exists()

    def test_row_count_matches_reactions(self, tmp_path):
        write_all(tmp_path, _make_result(20, 80), _make_dummy_data(100, 20, 80),
                  OutputConfig(), k_b=57.47, D_rel=0.434)
        with open(tmp_path / 'encounters.csv') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 20

class TestNearMissesCSV:
    """Test near_misses.csv output."""

    def test_row_count_matches_escaped(self, tmp_path):
        write_all(tmp_path, _make_result(30, 70), _make_dummy_data(100, 30, 70),
                  OutputConfig(), k_b=57.47, D_rel=0.434)
        with open(tmp_path / 'near_misses.csv') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 70

class TestPathsNPZ:
    """Test paths.npz output."""

    def test_file_created(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        assert (tmp_path / 'paths.npz').exists()

    def test_shape_correct(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        d = np.load(tmp_path / 'paths.npz')
        assert d['data'].shape[1] == 8  # traj_id, step, x, y, z, q0, q1, q2

    def test_disabled(self, tmp_path):
        oc = OutputConfig(full_paths=False)
        write_all(tmp_path, _make_result(), _make_dummy_data(), oc,
                  k_b=57.47, D_rel=0.434)
        assert not (tmp_path / 'paths.npz').exists()


class TestRadialDensity:
    """Test radial_density.csv output."""

    def test_columns(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        with open(tmp_path / 'radial_density.csv') as f:
            cols = csv.DictReader(f).fieldnames
        assert 'r_center' in cols
        assert 'density' in cols

    def test_density_nonnegative(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        with open(tmp_path / 'radial_density.csv') as f:
            for row in csv.DictReader(f):
                assert float(row['density']) >= 0


class TestMilestoneFlux:
    """Test milestone_flux.csv output."""

    def test_columns(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        with open(tmp_path / 'milestone_flux.csv') as f:
            cols = csv.DictReader(f).fieldnames
        expected = ['radius', 'flux_outward', 'flux_inward', 'net_flux']
        for c in expected:
            assert c in cols

    def test_row_count(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        with open(tmp_path / 'milestone_flux.csv') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 11  # 11 milestone radii

class TestTransitionMatrix:
    """Test transition_matrix.npz output."""

    def test_square(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        d = np.load(tmp_path / 'transition_matrix.npz')
        assert d['counts'].shape[0] == d['counts'].shape[1]
        assert d['counts'].shape[0] == 50

class TestPCommit:
    """Test p_commit.npz output."""

    def test_values_in_01(self, tmp_path):
        write_all(tmp_path, _make_result(), _make_dummy_data(), OutputConfig(),
                  k_b=57.47, D_rel=0.434)
        d = np.load(tmp_path / 'p_commit.npz')
        assert np.all(d['p_commit'] >= 0)
        assert np.all(d['p_commit'] <= 1)

class TestEdgeCases:
    """Test edge cases."""

    def test_zero_reactions(self, tmp_path):
        data = _make_dummy_data(100, 0, 100)
        result = _make_result(0, 100)
        write_all(tmp_path, result, data, OutputConfig(), k_b=57.47, D_rel=0.434)
        rj = json.loads((tmp_path / 'results.json').read_text())
        assert rj['P_rxn'] == 0.0
        assert rj['k_on'] == 0.0
        # encounters.csv should not be created
        assert not (tmp_path / 'encounters.csv').exists()

    def test_all_reacted(self, tmp_path):
        data = _make_dummy_data(50, 50, 0)
        result = _make_result(50, 0)
        write_all(tmp_path, result, data, OutputConfig(), k_b=57.47, D_rel=0.434)
        rj = json.loads((tmp_path / 'results.json').read_text())
        assert rj['P_rxn'] == 1.0
        # near_misses.csv should have 0 rows (no escapes)

    def test_all_disabled(self, tmp_path):
        oc = OutputConfig(
            results_json=False, trajectories_csv=False, encounters_csv=False,
            near_misses_csv=False, full_paths=False, radial_density=False,
            angular_map=False, fpt_distribution=False, contact_frequency=False,
            milestone_flux=False, p_commit=False, transition_matrix=False,
            energetics=False, pose_clusters=False,
        )
        written = write_all(tmp_path, _make_result(), _make_dummy_data(), oc,
                            k_b=57.47, D_rel=0.434)
        assert len(written) == 0

    def test_file_count_all_enabled(self, tmp_path):
        written = write_all(tmp_path, _make_result(), _make_dummy_data(),
                            OutputConfig(), k_b=57.47, D_rel=0.434)
        assert len(written) == 14
