#!/usr/bin/env python3
"""Stack tokenizer reconstruction grids in a declared candidate order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    manifest_rows = []
    width = None
    for mapping in args.image:
        name, separator, raw_path = mapping.partition("=")
        if not separator:
            raise ValueError(f"invalid image mapping: {mapping}")
        path = Path(raw_path)
        image = np.asarray(iio.imread(path))
        if width is None:
            width = image.shape[1]
        if image.shape[1] != width:
            raise ValueError(
                f"grid widths differ: expected {width}, got {image.shape[1]}"
            )
        if rows:
            rows.append(np.full((8, width, 3), 24, dtype=np.uint8))
        start_row = sum(row.shape[0] for row in rows)
        rows.append(image)
        manifest_rows.append(
            {
                "name": name,
                "source": str(path),
                "start_row": start_row,
                "end_row_exclusive": start_row + image.shape[0],
            }
        )

    output = np.concatenate(rows, axis=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output, output)
    args.manifest.write_text(
        json.dumps(
            {
                "output": str(args.output),
                "shape": list(output.shape),
                "rows": manifest_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
