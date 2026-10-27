#!/usr/bin/env bash
# Build the site and copy it to the server. Run from the repo root:
#   SERVER=user@178.104.74.99 ./deploy/deploy.sh
# Needs web/public/knightvision.onnx present locally (it isn't in git; see web/README.md).
set -euo pipefail

SERVER="${SERVER:?set SERVER=user@host}"
TARGET="${TARGET:-/var/www/knightvision}"

cd "$(dirname "$0")/../web"
if [ ! -f public/knightvision.onnx ]; then
  echo "web/public/knightvision.onnx is missing; the game would not load. See web/README.md." >&2
  exit 1
fi

npm ci
npm run build

# --delete removes files from old builds; the fingerprinted assets change names on every build.
rsync -az --delete --info=stats1 dist/ "$SERVER:$TARGET/"
echo "Deployed to $SERVER:$TARGET"
