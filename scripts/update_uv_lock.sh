#!/bin/bash
set -e

# Check if pyproject.toml exists in current directory
if [ ! -f "pyproject.toml" ]; then
    echo "No pyproject.toml found in current directory."
    exit 0
fi

# Run the uv command in the current directory
echo "Running uv $@ in $(pwd)"
uv "$@"