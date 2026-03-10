from pathlib import Path

SEARCH_ROOTS=[Path.cwd(), Path.cwd()/"data", Path.cwd()/"Data", Path("/work/projects/machine_learning/data"), Path("/work/projects/machine_learning/Data")]
PREFERRED=("insurance.csv","penguins.csv","seeds.csv","adult_test.csv.zip","spacex_launch_dash.csv")

def _candidates():
    seen=set()
    ordered=[]
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for name in PREFERRED:
            p=root / name
            if p.exists():
                s=str(p)
                if s not in seen:
                    ordered.append(s); seen.add(s)
        for pat in ("*.csv","*.zip","*.tsv","*.txt"):
            for p in sorted(root.glob(pat)):
                s=str(p)
                if s not in seen:
                    ordered.append(s); seen.add(s)
    return ordered

def upload():
    return {p: b"" for p in _candidates()}

def download(filename):
    return filename
