#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version_file="$repo_root/src/fdq/__about__.py"

current_version="$(sed -nE 's/^__version__ = "([^"]+)"/\1/p' "$version_file")"
if [[ -z "$current_version" ]]; then
    echo "Could not find __version__ in $version_file" >&2
    exit 1
fi

echo "Current version: $current_version"
read -r -p "Set version to: " target_version

if [[ -z "$target_version" ]]; then
    echo "No version entered; aborting." >&2
    exit 1
fi

if [[ ! "$target_version" =~ ^[0-9]+(\.[0-9]+){1,2}([a-zA-Z0-9.+_-]+)?$ ]]; then
    echo "Version '$target_version' does not look like a valid Python package version." >&2
    exit 1
fi

echo
echo "About to set:"
echo "  $current_version -> $target_version"
echo
read -r -p "Continue? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

tmp_file="$(mktemp)"
sed -E "s/^__version__ = \"[^\"]+\"/__version__ = \"$target_version\"/" "$version_file" > "$tmp_file"
mv "$tmp_file" "$version_file"

echo
echo "Updated $version_file"
echo
echo "Recommended release commands:"
echo "  git add src/fdq/__about__.py"
echo "  git commit -m \"Bump version to $target_version\""
echo "  git tag v$target_version"
echo "  git push origin HEAD --tags"
