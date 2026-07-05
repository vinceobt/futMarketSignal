"""Enable `python -m futmarket ...` as an entry point equivalent to the
`futmarket` console script — handy when the venv isn't on PATH."""

from .cli import main

if __name__ == "__main__":
    main()
