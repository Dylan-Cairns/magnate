/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Build-time opt-in for the `?fixture=<id>` dev fixtures in a production
   * build. Vite statically replaces the value, so the default deployed build
   * (unset) keeps the fixture code tree-shaken out. Set it to `'true'` when
   * building a bundle for profiling/verification:
   *
   *   `VITE_ENABLE_DEV_FIXTURES=true yarn build`
   */
  readonly VITE_ENABLE_DEV_FIXTURES?: string;
}
