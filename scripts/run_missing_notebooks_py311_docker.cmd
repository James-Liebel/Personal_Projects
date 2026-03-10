@echo off
setlocal

cd /d "%~dp0\.."

set "HOST_DIR=%CD%"
set "HOST_DIR=%HOST_DIR:\=/%"
set "DRIVE=%HOST_DIR:~0,1%"
set "PATH_NO_DRIVE=%HOST_DIR:~3%"
set "DOCKER_MOUNT=//%DRIVE%/%PATH_NO_DRIVE%"

set "DOCKER_CMD=python -m pip install --upgrade pip && python -m pip install nbformat nbclient ipykernel ipywidgets numpy pandas matplotlib seaborn scikit-learn plotly xgboost nltk more-itertools numpy-financial opencv-python-headless sympy torch tensorflow tensorflow-datasets && python -c \"import pathlib,urllib.request; pathlib.Path('projects/machine_learning/Data').mkdir(parents=True, exist_ok=True); pathlib.Path('projects/machine_learning/data').mkdir(parents=True, exist_ok=True); urllib.request.urlretrieve('https://raw.githubusercontent.com/MicrosoftLearning/mslearn-ml-basics/refs/heads/main/Labfiles/data/seeds.csv', 'projects/machine_learning/Data/seeds.csv'); urllib.request.urlretrieve('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv', 'projects/machine_learning/data/penguins.csv')\" && python -c \"from pathlib import Path; root=Path('scripts/stubs/google/colab'); root.mkdir(parents=True, exist_ok=True); (root / '__init__.py').write_text('from . import files\\n', encoding='utf-8'); (root / 'files.py').write_text('from pathlib import Path\\n\\nSEARCH_ROOTS=[Path.cwd(), Path.cwd()/\\\"data\\\", Path.cwd()/\\\"Data\\\", Path(\\\"/work/projects/machine_learning/data\\\"), Path(\\\"/work/projects/machine_learning/Data\\\")]\\nPREFERRED=(\\\"insurance.csv\\\",\\\"penguins.csv\\\",\\\"seeds.csv\\\",\\\"adult_test.csv.zip\\\",\\\"spacex_launch_dash.csv\\\")\\n\\ndef _candidates():\\n    seen=set()\\n    ordered=[]\\n    for root in SEARCH_ROOTS:\\n        if not root.exists():\\n            continue\\n        for name in PREFERRED:\\n            p=root / name\\n            if p.exists():\\n                s=str(p)\\n                if s not in seen:\\n                    ordered.append(s); seen.add(s)\\n        for pat in (\\\"*.csv\\\",\\\"*.tsv\\\",\\\"*.txt\\\",\\\"*.xlsx\\\",\\\"*.xls\\\"):\\n            for p in sorted(root.glob(pat)):\\n                s=str(p)\\n                if s not in seen:\\n                    ordered.append(s); seen.add(s)\\n    return ordered\\n\\ndef upload():\\n    return {p: b\\\"\\\" for p in _candidates()}\\n\\ndef download(filename):\\n    return filename\\n', encoding='utf-8')\" && python scripts/run_notebooks.py --only-missing-outputs"

echo Running notebooks with missing outputs in Python 3.11 Docker...
docker run --rm ^
  -v "%DOCKER_MOUNT%:/work" ^
  -w /work ^
  -e JUPYTER_ALLOW_INSECURE_WRITES=1 ^
  -e JUPYTER_RUNTIME_DIR=/work/.jupyter_runtime ^
  -e PYTHONPATH=/work/scripts/stubs ^
  python:3.11 /bin/sh -lc "%DOCKER_CMD%"

set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Some notebooks failed. See notebook_execution_report.txt for details.
)

exit /b %EXIT_CODE%
