/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the Shpeed API, e.g. "/api". Unset on GitHub Pages, where static files are used. */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
