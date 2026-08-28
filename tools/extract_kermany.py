import os, zipfile

SRC = r"C:\Users\HP\.cache\kagglehub\datasets\paultimothymooney\kermany2018\2.archive"
DST = r"D:\datasets\kermany2018"

os.makedirs(DST, exist_ok=True)

def clean(p):
    # strip trailing spaces/dots from each path component (illegal on Windows)
    parts = [c.rstrip(" .") for c in p.replace("\\", "/").split("/") if c not in ("", ".", "..")]
    return os.path.join(*parts) if parts else ""

n = 0
with zipfile.ZipFile(SRC) as z:
    for info in z.infolist():
        rel = clean(info.filename)
        if not rel:
            continue
        out = os.path.join(DST, rel)
        if info.is_dir():
            os.makedirs(out, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with z.open(info) as src, open(out, "wb") as dst:
            dst.write(src.read())
        n += 1
        if n % 5000 == 0:
            print(f"  {n} files...")

print(f"Done: {n} files -> {DST}")