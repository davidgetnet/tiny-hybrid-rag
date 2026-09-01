#!/bin/sh

set -eu

if ! repo_root=$(git rev-parse --show-toplevel 2>/dev/null); then
    printf 'ERROR: Run this inside the Tiny Hybrid RAG Git repository.\n' >&2
    exit 2
fi

cd "$repo_root"
git config --local alias.precheck '!scripts/precheck.sh'

printf 'Configured repository-local alias:\n'
printf '    git precheck -> scripts/precheck.sh\n'
printf 'Stored in this clone only: %s/.git/config\n' "$repo_root"

