# Contributing to KnightVision

For a bug report, include the relevant board position, the move sequence, the expected result, and the actual result. For experiment issues, also include the Python/package versions and the script being run. Keep credentials and private data out of reports.

## Local workflow

Fork the repository on GitHub, clone your fork using its clone URL, then create a branch for one focused change.

From the repository root, run the rules tests with Python 3.11:

```bash
PYTHONPATH=core python3 -B -m unittest discover -s tests -v
```

This command needs no third-party packages. If you change engine behavior, add a focused regression test in `tests/` and rerun the suite. The [README](README.md#working-with-the-experiments) describes the additional requirements and current limits of the ML scripts.

In a pull request, explain the behavior changed, the checks run, and anything not validated. Avoid committing credentials, model checkpoints, training datasets, or generated logs.

## Credit and license

KnightVision is a project by [Arian Shamsabadi](https://github.com/TheRealShamsaba) and [BakerDmo](https://github.com/BakerDmo). Preserve the existing authorship credit and [MIT license](LICENSE).
