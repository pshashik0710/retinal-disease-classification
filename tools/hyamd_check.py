import os, re
from collections import Counter
import pandas as pd

LAB = r"D:\datasets\hyamd\labels\labels.csv"
IMG = r"D:\datasets\hyamd\Images"

d = pd.read_csv(LAB)
files = set(os.listdir(IMG))
pat = re.compile(r"^(.+_[LRDE])(?:_(\d+))?$")


def to_file(image_id):
    m = pat.match(image_id)
    if not m:
        return None
    base, k = m.group(1), m.group(2)
    if k is None or int(k) == 1:
        return base + ".png"
    return f"{base}_{int(k) - 1}_.png"


cand = [to_file(i) for i in d.image_id]

print("label rows      :", len(d))
print("files on disk   :", len(files))
print("unmapped ids    :", sum(c is None for c in cand))
print("matched to file :", sum(c in files for c in cand), "/", len(d))
print("unique targets  :", len(set(cand)))

dups = [k for k, v in Counter(cand).items() if v > 1]
print("colliding files :", len(dups))
for k in dups[:5]:
    rows = d.loc[[c == k for c in cand], ["image_id", "side", "AMD"]]
    print(f"   {k}")
    print("     " + rows.to_string(index=False).replace("\n", "\n     "))

print("orphan files    :", len(files - set(cand)))
print("  ", sorted(files - set(cand))[:5])