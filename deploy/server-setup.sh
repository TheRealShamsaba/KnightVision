#!/usr/bin/env bash
# Puts KnightVision live at knightvision.realcompanys.com. Run on the server, as root:
#   curl -fsSL https://raw.githubusercontent.com/TheRealShamsaba/KnightVision/v2/deploy/server-setup.sh | bash
# Optional: EMAIL=you@example.com for Let's Encrypt expiry notices.
#
# Careful with what's already there: it adds one nginx site and one folder, tests the config before
# reloading, and backs its change out if the test fails, so existing sites keep working.
set -euo pipefail

DOMAIN=knightvision.realcompanys.com
ROOT=/var/www/knightvision
RELEASE=https://github.com/TheRealShamsaba/KnightVision/releases/download/web-v2/knightvision-web-v2.tar.gz
SHA256=c93486649424ea3081b6f63f8a2b7104c3d1412b4863310c27e14e1fb893812b

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mStopped: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || fail "run this as root"

say "1/6 Checking what already serves the web"
if ! command -v nginx >/dev/null; then
  # Something else on port 80 (Apache, Caddy, a container) would clash with a new nginx.
  if ss -ltn 2>/dev/null | grep -qE ':80 '; then
    ss -ltnp | grep -E ':(80|443) ' || true
    fail "port 80 is used by something that isn't nginx (above). Tell Claude what it is; nothing was changed."
  fi
  command -v apt-get >/dev/null || fail "no nginx and no apt-get; tell Claude which OS this is. Nothing was changed."
  echo "nginx isn't installed; installing it."
  apt-get update -q && apt-get install -yq nginx
fi
nginx -v
nginx -t 2>&1 | tail -1 || fail "the existing nginx config already fails its own test; fix that first. Nothing was changed."

say "2/6 Downloading the site"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fL --progress-bar -o "$tmp/site.tar.gz" "$RELEASE"
echo "$SHA256  $tmp/site.tar.gz" | sha256sum -c - || fail "the download doesn't match the published build"

say "3/6 Installing it to $ROOT"
mkdir -p "$tmp/site"
tar -xzf "$tmp/site.tar.gz" -C "$tmp/site"
[ -f "$tmp/site/index.html" ] && [ -f "$tmp/site/knightvision.onnx" ] || fail "the archive is missing index.html or the model"
# Swap in the new copy whole, keeping the previous one beside it for a quick rollback.
rm -rf "$ROOT.prev"
[ -d "$ROOT" ] && mv "$ROOT" "$ROOT.prev"
mv "$tmp/site" "$ROOT"
chown -R www-data:www-data "$ROOT" 2>/dev/null || chmod -R a+rX "$ROOT"
du -sh "$ROOT"

say "4/6 Adding the nginx site"
if [ -d /etc/nginx/sites-available ]; then
  CONF=/etc/nginx/sites-available/$DOMAIN
  LINK=/etc/nginx/sites-enabled/$DOMAIN
else
  CONF=/etc/nginx/conf.d/$DOMAIN.conf
  LINK=
fi
cat > "$CONF" <<'NGINX'
# KnightVision at knightvision.realcompanys.com: a static site, no backend.
server {
    listen 80;
    listen [::]:80;
    server_name knightvision.realcompanys.com;

    root /var/www/knightvision;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # Vite fingerprints everything in /assets, so it can be cached for good.
    location /assets/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # ONNX Runtime's WebAssembly must be served as application/wasm (older mime.types lack it).
    location ~ \.wasm$ {
        default_type application/wasm;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # The 25 MB network: cache it a week so returning visitors don't download it again.
    location = /knightvision.onnx {
        default_type application/octet-stream;
        add_header Cache-Control "public, max-age=604800";
    }

    gzip on;
    gzip_types text/css application/javascript application/json image/svg+xml application/wasm;
}
NGINX
[ -n "$LINK" ] && ln -sf "$CONF" "$LINK"
if ! nginx -t; then
  rm -f "$CONF" ${LINK:+"$LINK"}
  fail "nginx rejected the new site, so it was removed again; your other sites are untouched"
fi
systemctl reload nginx
echo "nginx reloaded with $DOMAIN"

say "5/6 HTTPS"
here=$(curl -fs4 --max-time 5 https://api.ipify.org || hostname -I | awk '{print $1}')
there=$(getent ahostsv4 "$DOMAIN" | awk 'NR==1{print $1}' || true)
if [ "$there" != "$here" ]; then
  echo "$DOMAIN points to '${there:-nothing}', but this server is $here."
  echo "Add the DNS A record (knightvision -> $here) at GoDaddy, wait a few minutes, then run:"
  echo "  certbot --nginx -d $DOMAIN --redirect"
  echo "The site already works over http once DNS points here."
else
  command -v certbot >/dev/null || { apt-get update -q && apt-get install -yq certbot python3-certbot-nginx; }
  if [ -n "${EMAIL:-}" ]; then contact=(-m "$EMAIL"); else contact=(--register-unsafely-without-email); fi
  certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos "${contact[@]}" --redirect
fi

say "6/6 Checking it"
scheme=http; [ -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ] && scheme=https
for path in / /flight.html /knightvision.onnx; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --resolve "$DOMAIN:80:127.0.0.1" --resolve "$DOMAIN:443:127.0.0.1" -k "$scheme://$DOMAIN$path")
  echo "$scheme://$DOMAIN$path -> $code"
done
printf '\n\033[1mDone.\033[0m Previous copy (if any) kept at %s.prev\n' "$ROOT"
