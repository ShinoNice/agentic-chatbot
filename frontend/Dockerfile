# syntax=docker/dockerfile:1.7

# ─── Stage 1: Build the Angular bundle ───────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

# Install deps with the lockfile only — keeps the layer cacheable.
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ─── Stage 2: Serve via nginx ────────────────────────────────────────────────
FROM nginx:1.27-alpine

# Drop the default site so our SPA-aware config is the only one loaded.
RUN rm /etc/nginx/conf.d/default.conf

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Angular 19+ outputs the browser bundle under dist/<project>/browser/.
COPY --from=builder /app/dist/angular-app/browser /usr/share/nginx/html

# Runtime config injection: replaces __API_BASE_URL__ in index.html on
# container start. Files in /docker-entrypoint.d/ are auto-run by the
# nginx base image before nginx itself launches.
COPY frontend/entrypoint.sh /docker-entrypoint.d/40-runtime-config.sh
RUN chmod +x /docker-entrypoint.d/40-runtime-config.sh

EXPOSE 8080
