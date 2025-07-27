#!/bin/bash
# Setup script for git hooks to automate uv.lock management

set -e

echo "Setting up git hooks for uv.lock management automation..."

# Create the hooks directory if it doesn't exist
HOOKS_DIR=".git/hooks"
mkdir -p "$HOOKS_DIR"

# Copy post-merge hook
if [ -f ".githooks/post-merge" ]; then
    cp ".githooks/post-merge" "$HOOKS_DIR/post-merge"
    chmod +x "$HOOKS_DIR/post-merge"
    echo "Post-merge hook installed."
else
    echo "Warning: .githooks/post-merge not found."
    exit 1
fi

echo ""
echo "Git hooks setup complete. The post-merge hook will now automatically update uv.lock after merging changes."
echo "What happens now:"
echo "    - Pre-commit: Checks uv.lock consistency before committing."
echo "    - Post-merge: Auto updates uv.lock after merging changes."
echo ""