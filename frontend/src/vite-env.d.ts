/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Base URL of the Shpeed API, such as "/api".
   * Unset on GitHub Pages, where the static prediction files are used instead.
   */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
