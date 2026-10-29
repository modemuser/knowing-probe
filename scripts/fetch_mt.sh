#!/bin/sh
# Step 0 inputs: Marks & Tegmark's true/false statement sets, pinned to the commit the
# runs used, checked against the sha256 recorded in runs/step0/*.json. Not committed
# here (third-party data); fetched into data/raw/geometry-of-truth/.
set -eu
COMMIT=5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb
DIR=data/raw/geometry-of-truth
mkdir -p "$DIR"
for f in cities sp_en_trans; do
  curl -sfL -o "$DIR/$f.csv" "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/$COMMIT/datasets/$f.csv"
done
shasum -a 256 -c <<SUMS
f2560a0c1758c69935b50125111b574b71896bb75930dd87b1d8c973c1fa086e  $DIR/cities.csv
9afa97c1b8af539a05f47655a80a5c7832c6fdb914885fd24cb4bcccbe2795b5  $DIR/sp_en_trans.csv
SUMS
