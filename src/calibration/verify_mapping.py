import argparse
import csv
import json
import math
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--lut_path", required=True)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    with open(args.lut_path, "rb") as f:
        lut = pickle.load(f)

    pixel_errors = []
    workspace_errors = []
    n_rows = 0

    with open(args.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["ok"]) != 1:
                continue
            x = float(row["x"])
            y = float(row["y"])
            u = float(row["u"])
            v = float(row["v"])
            u2, v2 = lut.workspace_to_pixel(x, y)
            x2, y2 = lut.pixel_to_workspace(u, v)
            pixel_errors.append(math.hypot(u2 - u, v2 - v))
            workspace_errors.append(math.hypot(x2 - x, y2 - y))
            n_rows += 1

    pixel_errors = np.asarray(pixel_errors, dtype=np.float64)
    workspace_errors = np.asarray(workspace_errors, dtype=np.float64)
    report = {
        "n_rows": int(n_rows),
        "pixel_rmse": float(np.sqrt(np.mean(pixel_errors ** 2))) if len(pixel_errors) else None,
        "pixel_max": float(np.max(pixel_errors)) if len(pixel_errors) else None,
        "workspace_rmse": float(np.sqrt(np.mean(workspace_errors ** 2))) if len(workspace_errors) else None,
        "workspace_max": float(np.max(workspace_errors)) if len(workspace_errors) else None,
    }
    print(json.dumps(report, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
