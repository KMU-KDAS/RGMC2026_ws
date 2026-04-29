from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd


def show_bgr(image_bgr, title=None, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()


def load_csv_preview(csv_path, n=10):
    return pd.read_csv(csv_path).head(n)


def list_failed_points(failed_csv_path):
    p = Path(failed_csv_path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)
