#!/usr/bin/env bash
# Puts KnightVision live at knightvision.realcompanys.com. Run on the server, as root:
#   curl -fsSL https://raw.githubusercontent.com/TheRealShamsaba/KnightVision/v2/deploy/server-setup.sh | bash
# Root can't log in over ssh; come in as arian, then become root:
#   ssh -i ~/.ssh/realcompany-pro-1_ed25519 arian@178.104.74.99
#   sudo -i
# Optional: EMAIL=you@example.com for Let's Encrypt expiry notices.
#
# Careful with what's already there: it adds one nginx site and one folder, tests the config before
# reloading, and backs its change out if the test fails, so existing sites keep working.
set -euo pipefail

DOMAIN=knightvision.realcompanys.com
ROOT=/var/www/knightvision
RELEASE=https://github.com/TheRealShamsaba/KnightVision/releases/download/web-v2.2/knightvision-web-v2.2.tar.gz
SHA256=6f9f9fb4579cd47d787c96ac1c5f4e0a9f2af53fc315dd1d15985b42aa0daf11

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
[ -f "$CONF" ] && cp "$CONF" "$CONF.bak"
# What every server block for the site serves; shared by the http and https versions.
LOCATIONS='
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

    # ONNX Runtime ships WebAssembly and an ES module loader (.mjs). Older mime.types know
    # neither, and browsers refuse to run a module served as a plain download.
    location ~ \.wasm$ {
        default_type application/wasm;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    location ~ \.mjs$ {
        types { }
        default_type text/javascript;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # The 25 MB network: cache it a week so returning visitors do not download it again.
    location = /knightvision.onnx {
        default_type application/octet-stream;
        add_header Cache-Control "public, max-age=604800";
    }

    gzip on;
    gzip_types text/css application/javascript text/javascript application/json image/svg+xml application/wasm;
'
CERT=/etc/letsencrypt/live/$DOMAIN
SSL_EXTRA=
[ -f /etc/letsencrypt/options-ssl-nginx.conf ] && SSL_EXTRA="$SSL_EXTRA    include /etc/letsencrypt/options-ssl-nginx.conf;
"
[ -f /etc/letsencrypt/ssl-dhparams.pem ] && SSL_EXTRA="$SSL_EXTRA    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;
"
if [ -f "$CERT/fullchain.pem" ]; then
  # Already has a certificate (an earlier run): write the https site directly, so a rerun never drops it.
  cat > "$CONF" <<NGINX
# KnightVision at $DOMAIN: a static site, no backend. Written by deploy/server-setup.sh.
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;
    return 301 https://\$host\$request_uri;
}
server {
    listen 443 ssl;
    listen [::]:443 ssl;
    server_name $DOMAIN;
    ssl_certificate $CERT/fullchain.pem;
    ssl_certificate_key $CERT/privkey.pem;
$SSL_EXTRA$LOCATIONS}
NGINX
else
  cat > "$CONF" <<NGINX
# KnightVision at $DOMAIN: a static site, no backend. Written by deploy/server-setup.sh.
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;
$LOCATIONS}
NGINX
fi
[ -n "$LINK" ] && ln -sf "$CONF" "$LINK"
if ! nginx -t; then
  if [ -f "$CONF.bak" ]; then mv "$CONF.bak" "$CONF"; else rm -f "$CONF" ${LINK:+"$LINK"}; fi
  fail "nginx rejected the new config, so the previous one was put back; your other sites are untouched"
fi
rm -f "$CONF.bak"
systemctl reload nginx
echo "nginx reloaded with $DOMAIN"

say "5/6 HTTPS"
here=$(curl -fs4 --max-time 5 https://api.ipify.org || hostname -I | awk '{print $1}')
there=$(getent ahostsv4 "$DOMAIN" | awk 'NR==1{print $1}' || true)
if [ -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
  echo "Already has a certificate (renews automatically); the https site was written above."
elif [ "$there" != "$here" ]; then
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
