#!/usr/bin/env bash
set -euo pipefail

DEST="input_data/images_dreambooth"

if [ -d "$DEST" ]; then
    echo "Folder '$DEST' already exists — skipping download."
    exit 0
fi

echo "Downloading DreamBooth dataset to '$DEST'..."

# Sparse-checkout only the dataset/ folder from the repo
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

git clone --filter=blob:none --sparse https://github.com/google/dreambooth.git "$TMP/repo"
git -C "$TMP/repo" sparse-checkout set dataset

mv "$TMP/repo/dataset" "$DEST"

echo "Done. Subjects downloaded:"
ls "$DEST"
