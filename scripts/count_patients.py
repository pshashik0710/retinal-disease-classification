import os, re
from collections import defaultdict

ROOT = r"D:\datasets\kermany2018\OCT2017"
PAT = re.compile(r"^[A-Z]+-(\d+)-\d+", re.I)

for split in ("train", "val", "test"):
    print(f"\n=== {split} ===")
    split_pat = set()
    for cls in sorted(os.listdir(os.path.join(ROOT, split))):
        d = os.path.join(ROOT, split, cls)
        if not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d) if f.lower().endswith((".jpeg", ".jpg", ".png"))]
        pats = {m.group(1) for f in files if (m := PAT.match(f))}
        split_pat |= pats
        print(f"  {cls:<8} {len(files):>6} images  {len(pats):>5} patients  "
              f"{len(files)/max(len(pats),1):>5.1f} scans/patient")
    print(f"  TOTAL unique patients in {split}: {len(split_pat)}")
    globals()[f"p_{split}"] = split_pat

print("\n=== OFFICIAL SPLIT PATIENT OVERLAP ===")
print(f"  train n test: {len(p_train & p_test)}")
print(f"  train n val : {len(p_train & p_val)}")
print(f"  val   n test: {len(p_val & p_test)}")