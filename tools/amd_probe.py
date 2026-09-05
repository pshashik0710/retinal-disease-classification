"""Open every AMDNet23 image individually to find the one that crashes."""
import os, sys
import pandas as pd
from PIL import Image
import numpy as np

ROOT = r"D:\datasets\amdnet23"
m = pd.read_csv("manifests/amdnet23_clean.csv")

for i, rel in enumerate(m["Directory"], 1):
    p = os.path.join(ROOT, rel.replace("/", os.sep))
    print(f"{i:5}/{len(m)}  {rel}", flush=True)   # printed BEFORE opening
    try:
        with Image.open(p) as im:
            a = np.array(im.convert("RGB"))
    except Exception as e:
        print(f"        PYTHON ERROR: {e}", flush=True)

print("all opened")