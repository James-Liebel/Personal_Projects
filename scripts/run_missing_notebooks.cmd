@echo off
setlocal

cd /d "%~dp0\.."

set "JUPYTER_ALLOW_INSECURE_WRITES=1"
set "JUPYTER_RUNTIME_DIR=%CD%\.jupyter_runtime"
if not exist "%JUPYTER_RUNTIME_DIR%" mkdir "%JUPYTER_RUNTIME_DIR%"

echo Running notebooks with missing outputs...
python scripts\run_notebooks.py --only-missing-outputs
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Some notebooks failed. See notebook_execution_report.txt for details.
)

exit /b %EXIT_CODE%
