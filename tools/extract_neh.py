"""Extract the NEH OCT dataset to D:, sanitising Windows-illegal path components."""

import os, zipfile

SRC = os.path.join(os.environ["USERPROFILE"], "Downloads",
                   "NEH_UT_2021RetinalOCTDataset.zip")
DST = r"D:\datasets\neh"

os.makedirs(DST, exist_ok=True)

def clean(p):
    parts = [c.rstrip(" .") for c in p.replace("\\", "/").split("/")
             if c not in ("", ".", "..")]
    return os.path.join(*parts) if parts else ""

n = 0
with zipfile.ZipFile(SRC) as z:
    for info in z.infolist():
        rel = clean(info.filename)
        if not rel or "__MACOSX" in rel:
            continue
        out = os.path.join(DST, rel)
        if info.is_dir():
            os.makedirs(out, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with z.open(info) as s, open(out, "wb") as d:
            d.write(s.read())
        n += 1
        if n % 2000 == 0:
            print(f"  {n} files...")

print(f"Done: {n} files -> {DST}")