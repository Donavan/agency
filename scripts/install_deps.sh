#!/usr/bin/env bash
set -e
# Install the requirements
echo "Installing deps"
cd src
pip install -e agent_c_core
pip install -e agent_c_tools
pip install -e my_agent_c
pip install -e agent_c_reference_apps


echo "Initial setup completed successfully."
echo "Remember to activate the virtual environment with 'source .venv/bin/activate' before you start working."

# Exit from the script without deactivating (since it's a new shell instance)
