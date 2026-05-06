#!/bin/sh
# Runtime config injection for the Angular SPA.
#
# Angular bundles environment values at BUILD time. To reuse one image
# across deployments without rebuilding, we substitute the API base URL
# into index.html when the container starts. Browser code reads it from
# `window.__APP_CONFIG__.apiBaseUrl` on bootstrap.
#
# Default to empty string (= same-origin) so the nginx /api/ reverse
# proxy handles requests without the SPA needing to know any host.

set -eu

API_BASE_URL="${API_BASE_URL:-}"
# Full internal FQDN of the upstream API container app. In prod this comes
# from Terraform (azurerm_container_app_environment.aca.default_domain).
# Locally (no ACA DNS) the nginx /api/ block won't be reachable anyway.
API_UPSTREAM="${API_UPSTREAM:-localhost:8001}"

INDEX="/usr/share/nginx/html/index.html"
NGINX_CONF="/etc/nginx/conf.d/default.conf"

for f in "$INDEX" "$NGINX_CONF"; do
    if [ ! -f "$f" ]; then
        echo "[entrypoint] $f missing; image is broken" >&2
        exit 1
    fi
done

# Substitute placeholders. `|` delimiter avoids escaping URLs/hostnames.
sed -i "s|__API_BASE_URL__|${API_BASE_URL}|g" "$INDEX"
sed -i "s|__API_UPSTREAM__|${API_UPSTREAM}|g" "$NGINX_CONF"

echo "[entrypoint] runtime config injected (API_BASE_URL='${API_BASE_URL}', API_UPSTREAM='${API_UPSTREAM}')"
