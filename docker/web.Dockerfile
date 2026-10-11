# syntax=docker/dockerfile:1
# Shpeed web image (P8-14). Build context is the repository root.
#   runtime (default): static build served by unprivileged nginx, /api proxied to the API
#   dev:               Vite dev server with hot reload and ./frontend mounted

FROM node:20-alpine AS deps
WORKDIR /app
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

FROM deps AS dev
COPY frontend/ ./
EXPOSE 8080
CMD ["pnpm", "dev", "--host", "0.0.0.0", "--port", "8080", "--strictPort"]

FROM deps AS build
COPY frontend/ ./
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL
RUN pnpm build

FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
