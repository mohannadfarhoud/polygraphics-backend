@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if defined POLYGRAPH_AI_PRIOR_PYTHON (
  set "PY_EXE=%POLYGRAPH_AI_PRIOR_PYTHON%"
) else (
  set "PY_EXE=%SCRIPT_DIR%..\.\venv\Scripts\python.exe"
)

if not exist "%PY_EXE%" (
  set "PY_EXE=python"
)

"%PY_EXE%" "%SCRIPT_DIR%ai_prior_command_adapter.py" %*
exit /b %ERRORLEVEL%

