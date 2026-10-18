"""Convert the official UCI Auto MPG text file to the analysis CSV format."""

import argparse
import csv
import shlex
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ORIGINS = {"1": "usa", "2": "europe", "3": "japan"}
HEADER = [
    "name", "mpg", "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin",
]


def parse_record(line):
    fields = shlex.split(line)
    if len(fields) != 9 or fields[7] not in ORIGINS:
        raise ValueError(f"Unexpected Auto MPG record: {line.rstrip()}")
    mpg, cylinders, displacement, horsepower, weight, acceleration, year, origin, name = fields
    return [
        name, mpg, cylinders, displacement,
        "" if horsepower == "?" else horsepower,
        weight, acceleration, year, ORIGINS[origin],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Downloaded UCI auto-mpg.data file")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "data" / "Automobile.csv")
    args = parser.parse_args()

    with args.source.open(encoding="utf-8") as source:
        rows = [parse_record(line) for line in source if line.strip()]
    if len(rows) != 398:
        raise ValueError(f"Expected 398 UCI Auto MPG records, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(HEADER)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
