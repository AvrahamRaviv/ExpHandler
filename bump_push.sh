#!/usr/bin/env bash
set -euo pipefail

# Bump semver tag (vMAJOR.MINOR.PATCH), keep sidebar version label in sync,
# push commits, push tag.
#
# Usage:
#   ./bump_push.sh                # patch bump (default)
#   ./bump_push.sh patch|minor|major
#
# Flow:
#   1. Compute next version from latest vX.Y.Z tag (fallback: sidebar.py).
#   2. Patch ui/sidebar.py version label.
#   3. Commit the bump, push, create annotated tag, push tag.

cd "$(dirname "$0")"

BUMP="${1:-patch}"
case "$BUMP" in
  patch|minor|major) ;;
  *) echo "Usage: $0 [patch|minor|major]" >&2; exit 1 ;;
esac

SIDEBAR="ui/sidebar.py"
[ -f "$SIDEBAR" ] || { echo "Missing $SIDEBAR" >&2; exit 1; }

# Refuse to run if there are uncommitted tracked changes (other than the
# sidebar version line, which we are about to rewrite). Untracked files
# are ignored.
if ! git diff --quiet -- ':!ui/sidebar.py'; then
  echo "Uncommitted tracked changes present. Commit them first:" >&2
  git status -s
  exit 1
fi

# Pick the latest vX.Y.Z tag. If none exist, parse the sidebar label.
LATEST=$(git tag --list 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname | head -n1 || true)
if [ -z "$LATEST" ]; then
  LATEST=$(grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' "$SIDEBAR" | head -n1 || true)
fi
[ -n "$LATEST" ] || { echo "Cannot determine current version" >&2; exit 1; }

MAJOR=$(echo "$LATEST" | sed -E 's/^v([0-9]+)\.([0-9]+)\.([0-9]+).*/\1/')
MINOR=$(echo "$LATEST" | sed -E 's/^v([0-9]+)\.([0-9]+)\.([0-9]+).*/\2/')
PATCH=$(echo "$LATEST" | sed -E 's/^v([0-9]+)\.([0-9]+)\.([0-9]+).*/\3/')

case "$BUMP" in
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  patch) PATCH=$((PATCH + 1)) ;;
esac

NEW="v${MAJOR}.${MINOR}.${PATCH}"
echo "Bumping ${LATEST} -> ${NEW} (${BUMP})"

# Refuse to overwrite an existing tag.
if git rev-parse -q --verify "refs/tags/${NEW}" >/dev/null; then
  echo "Tag ${NEW} already exists" >&2
  exit 1
fi

# Patch the sidebar version label.
python - "$SIDEBAR" "$NEW" <<'PY'
import re, sys, pathlib
path, new = sys.argv[1], sys.argv[2]
p = pathlib.Path(path)
text = p.read_text()
new_text = re.sub(
    r'QLabel\("v\d+\.\d+\.\d+"\)', f'QLabel("{new}")', text, count=1
)
if new_text == text:
    sys.exit("version label not found in sidebar.py")
p.write_text(new_text)
PY

git add "$SIDEBAR"
if ! git diff --cached --quiet; then
  git commit -m "Bump version to ${NEW}"
fi

git push
git tag -a "${NEW}" -m "Release ${NEW}"
git push origin "${NEW}"

echo "Released ${NEW}"
