"""Test that all the notebooks can be executed."""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pytest import mark


def available_notebooks():
    """Return a list of all available notebooks."""
    base_path = Path(__file__).parents[2] / "notebooks"
    return list(base_path.rglob("*.ipynb"))


@mark.notebook
@mark.parametrize("notebook", available_notebooks())
def test_notebooks(notebook):
    """Test notebook execution.

    For a given notebook, try to execute it. If an error is raised, it fails.

    Args:
    ----
        notebook (str): The path to the notebook.
    """
    with notebook.open() as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(notebook.parent)}})
