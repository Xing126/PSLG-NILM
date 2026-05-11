import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_vendored_path = os.path.join(_project_root, "models", "claspy")

if os.path.isdir(_vendored_path) and _vendored_path not in __path__:
    __path__.append(_vendored_path)
