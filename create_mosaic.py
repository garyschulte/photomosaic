#!/usr/bin/env python3
"""
Photo mosaic creator: square or hexagonal tiles.

Examples:
  python3 create_mosaic.py -s source.jpg                       # 100 hexes across
  python3 create_mosaic.py -s source.jpg --cols 60 --scale 3 --grout 2
  python3 create_mosaic.py -s source.jpg --shape square --cols 100 --rows 50
  python3 create_mosaic.py --help                               # all options

Corpus colour features are cached in .mosaic_cache/, so re-runs with the
same tile shape skip the slow corpus scan.
"""

import sys

from mosaic.cli import main

if __name__ == "__main__":
    sys.exit(main())
