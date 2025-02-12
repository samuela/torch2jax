#!/bin/bash

set -euo pipefail

# Check if we're in nix-shell/nix develop
if [[ -z "${IN_NIX_SHELL}" ]]; then
    echo "Error: This script must be run from within nix develop"
    echo "Please run 'nix develop' first, then try again"
    exit 1
fi

echo "Building and publishing torch2jax to PyPI"
echo
echo "First, get a PyPI token from:"
echo "https://pypi.org/manage/project/torch2jax/settings/"
echo
read -sp "Enter your PyPI token: " TOKEN
echo

# Build the package
echo -e "\nBuilding package..."
python3 -m build

# Upload to PyPI using environment variable
echo -e "\nUploading to PyPI..."
TWINE_USERNAME="__token__" TWINE_PASSWORD="$TOKEN" python3 -m twine upload dist/*

echo -e "\nPackage successfully published to PyPI!"
echo
echo "SECURITY REMINDER: Please delete the token you just used from PyPI:"
echo "https://pypi.org/manage/project/torch2jax/settings/"
echo
read -p "Press Enter once you've deleted the token..."
echo "Done!"
