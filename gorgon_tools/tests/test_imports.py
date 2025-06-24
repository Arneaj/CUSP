"""Test that all modules can be imported."""
import os
from pathlib import Path

import pytest


def include(dirpath, filename):
    """Rules to include a file/path.

    Returns True if the file/path is included, False otherwise.

    Args:
    ----
        dirpath (str): The path to the directory.
        filename (str): The name of the file.

    Returns:
    -------
        bool: True if the file/path is included, False otherwise.
    """
    if "__pycache__" in dirpath:
        return False
    if filename == "__init__.py":
        return False
    if filename.endswith(".so"):
        return False
    if filename.startswith("test"):
        return False
    if filename.endswith(".py"):
        return True
    return False


# Path to directory containing gorgon_tools module
p = Path(os.path.abspath(__file__)).parents[2]

# Look for all .py files
python_files = [
    [
        os.sep.join([dirpath[len(str(p)) + 1 :], filename])
        for filename in filenames
        if include(dirpath, filename)
    ]
    for dirpath, _, filenames in os.walk(p / "gorgon_tools/")
]
# Flatten the list
python_files = [y for x in python_files for y in x]
# Convert from filenames to module names
module_names = [file.replace(os.sep, ".")[:-3] for file in python_files]

for n in module_names:
    print(n)


# Run test on all module names
@pytest.mark.parametrize("module_name", module_names)
def test_imports(module_name):
    """Test module import.

    For a given module name, try to import it. If an error is raised, it fails.

    Args:
    ----
        module_name (str): name of module, e.g. gorgon_tools.models.bowshock
    """
    # Don't check scripts module or version.py (these will always fail)
    if not module_name[13:20] == "scripts" and not module_name[13] == ".":
        exec("import " + module_name)
