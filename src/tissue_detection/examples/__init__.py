from pathlib import Path

_here = Path(__file__).absolute().parent
_files = tuple([f for f in _here.iterdir() if f.suffix in [".tif", ".TIF"]])
files = {f.stem: f for f in _files}
