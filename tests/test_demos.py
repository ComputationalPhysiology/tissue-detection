import subprocess
import shutil
import os
from pathlib import Path
import pytest


here = Path(__file__).absolute().parent
demodir = here / ".." / "demos"
demos = [demo for demo in demodir.iterdir() if demo.suffix == ".ipynb"]


@pytest.fixture(scope="module")
def jupytext():
    return shutil.which("jupytext")


@pytest.mark.parametrize("demo", demos)
def test_demos_run_without_errors(demo, jupytext):
    os.environ["MPLBACKEND"] = "agg"  # Disable plt.show() in matplotlib

    path = Path(demo).with_suffix(".py")
    subprocess.run([jupytext, demo, "-o", path])
    exec(path.read_text())
    path.unlink()
