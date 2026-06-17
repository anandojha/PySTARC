"""Regression tests for the multi-GPU run splitter element handling.

The splitter rewrites n_trajectories, max_steps, seed, and work_dir in a copy
of the input XML before writing one file per GPU run. Those tags are optional
in a PySTARC input file (the parser supplies defaults when they are absent), so
the splitter must be able to set them even when the source XML omits them.
"""

import xml.etree.ElementTree as ET

from pystarc.multi_GPU.multi_GPU_runs import _set_or_create


def test_set_or_create_creates_missing_tag():
    """A tag absent from the XML is created and given the requested text."""
    root = ET.fromstring("<simulation><receptor_pqr>r.pqr</receptor_pqr></simulation>")
    assert root.find("n_trajectories") is None

    el = _set_or_create(root, "n_trajectories", "1")

    assert el is root.find("n_trajectories")
    assert root.findtext("n_trajectories") == "1"


def test_set_or_create_updates_existing_tag():
    """A tag already present has its text overwritten in place."""
    root = ET.fromstring("<simulation><seed>1523</seed></simulation>")
    before = list(root)

    el = _set_or_create(root, "seed", "99")

    assert el is root.find("seed")
    assert root.findtext("seed") == "99"
    # No duplicate element is appended when the tag already exists.
    assert len(list(root)) == len(before)


def test_set_or_create_handles_all_optional_tags():
    """All four optional-with-default tags can be set on an XML that omits them."""
    root = ET.fromstring("<simulation><receptor_pqr>r.pqr</receptor_pqr></simulation>")

    _set_or_create(root, "n_trajectories", "25000")
    _set_or_create(root, "max_steps", "1")
    _set_or_create(root, "seed", "11111112")
    _set_or_create(root, "work_dir", ".")

    assert root.findtext("n_trajectories") == "25000"
    assert root.findtext("max_steps") == "1"
    assert root.findtext("seed") == "11111112"
    assert root.findtext("work_dir") == "."


def test_naive_find_text_assignment_would_crash():
    """The previous direct find(tag).text pattern raises on a missing tag,
    which is the failure mode the get-or-create helper removes."""
    root = ET.fromstring("<simulation></simulation>")

    try:
        root.find("n_trajectories").text = "1"
    except AttributeError:
        pass
    else:
        raise AssertionError("expected AttributeError on missing tag")

    # The helper sets the same tag without raising.
    _set_or_create(root, "n_trajectories", "1")
    assert root.findtext("n_trajectories") == "1"
