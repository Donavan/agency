@echo off
SETLOCAL

:: Store the starting directory
pushd %CD%

:: Navigate to repo root if script is run from scripts directory
if "%CD:~-7%"=="scripts" (
    cd ..
)

:: Start the backend server (in a new window)
start cmd /k ".venv\scripts\activate.bat && python -m uvicorn src.agent_c_reference_apps.src.agent_c_reference_apps.react_fastapi.backend.main:app --host 0.0.0.0 --port 8000 --log-level info"

:: Start the frontend dev server (in a new window)
start cmd /k "cd src\agent_c_reference_apps\src\agent_c_reference_apps\react_fastapi\frontend && npm run dev"

:: Return to original directory
popd

ENDLOCAL