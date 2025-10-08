#!/bin/bash
set -e

# Find all directories containing a pyproject.toml file
UV_DIRS=$(find . -type f -name "pyproject.toml" -exec dirname {} \; | sort -u)

if [ -z "$UV_DIRS" ]; then
    echo "No uv directories found."
    exit 0
fi

# Run the uv command in each directory found
for dir in $UV_DIRS; do
    echo "Running uv $@ in $dir"
    cd "$dir" && uv "$@"
done