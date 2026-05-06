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

INDEX="/usr/share/nginx/html/index.html"

if [ ! -f "$INDEX" ]; then
    echo "[entrypoint] $INDEX missing; image is broken" >&2
    exit 1
fi

# Use a delimiter unlikely to appear in URLs (|).
sed -i "s|__API_BASE_URL__|${API_BASE_URL}|g" "$INDEX"

echo "[entrypoint] runtime config injected (API_BASE_URL='${API_BASE_URL}')"
