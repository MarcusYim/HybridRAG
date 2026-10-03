"""Entry point for `python -m circuitrag`.

Also tolerates being run as a plain file path (`python circuitrag/__main__.py`),
which Python would otherwise reject: executing a file directly leaves
`__package__` unset, so the relative imports in `cli` have no parent package to
resolve against. Putting the project root on `sys.path` and importing the
package by name gives them one.
"""

if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from circuitrag.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    main()