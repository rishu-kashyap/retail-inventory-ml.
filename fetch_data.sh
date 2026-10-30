#!/usr/bin/env bash
# Public mirror of the M5 competition files (same content as the Kaggle competition).
# Kaggle alternative: kaggle competitions download -c m5-forecasting-accuracy -p data/raw
set -e
BASE="https://huggingface.co/datasets/denephew/M5_Forecasting/resolve/main"
mkdir -p data/raw && cd data/raw
for f in calendar.csv sell_prices.csv sales_train_evaluation.csv; do
  [ -f "$f" ] && { echo "skip $f (exists)"; continue; }
  echo "downloading $f ..."
  curl -sL --retry 3 -o "$f" "$BASE/$f"
done
ls -lh
