#!/usr/bin/env bash
# Creates a tarball of essential exo data, excluding large Kalshi and Polymarket stores.
# Output: /tmp/exo_data_essential.tar.gz (~70-80 MB)

set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
OUT="/tmp/exo_data_essential.tar.gz"

echo "Exporting essential data from $DATA_DIR ..."

tar czf "$OUT" \
  -C "$DATA_DIR" \
  --exclude='./features/source=kalshi' \
  --exclude='./features/source=polymarket' \
  features/ risk_index/

SIZE=$(du -sh "$OUT" | cut -f1)
echo "Done. Archive: $OUT ($SIZE)"
echo ""
echo "Transfer to Railway volume:"
echo "  1. Deploy the app first so the volume is mounted."
echo "  2. Open a Railway shell: railway shell"
echo "  3. In a second terminal, run:"
echo "       base64 $OUT | railway shell --command \"base64 -d > /data/exo_data.tar.gz\""
echo "  4. Back in railway shell:"
echo "       tar xzf /data/exo_data.tar.gz -C /data && rm /data/exo_data.tar.gz"
echo ""
echo "Or serve locally with ngrok and wget from the Railway shell:"
echo "  Local:  python3 -m http.server 9000 --directory /tmp"
echo "  ngrok:  ngrok http 9000"
echo "  Shell:  wget https://<ngrok-url>/exo_data_essential.tar.gz -O /data/tmp.tar.gz"
echo "          tar xzf /data/tmp.tar.gz -C /data && rm /data/tmp.tar.gz"
