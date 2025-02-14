@echo off
SETLOCAL

:: Store the starting directory
pushd %CD%

:: Install the requirements
echo Installing dependencies.
cd src
pip install -e agent_c_core
pip install -e agent_c_tools
pip install -e my_agent_c
pip install -e agent_c_reference_apps


if errorlevel 1 (
    echo Failed to install the required python packages.
    popd
    exit /b 1
)

:: Install NPM packages
echo Installing NPM dependencies...
cd agent_c_reference_apps\src\agent_c_reference_apps\react_fastapi\frontend
npm install

if errorlevel 1 (
    echo Failed to install the NPM packages.
    popd
    exit /b 1
)

:: Return to the original directory
popd

ENDLOCAL
