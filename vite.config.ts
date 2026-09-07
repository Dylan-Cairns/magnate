import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: './',
  plugins: [react()],
  server: {
    watch: {
      // The repository also contains a Python environment and generated training
      // artifacts. Watching them opens tens of thousands of unnecessary handles
      // on Windows and can delay the first local page load.
      ignored: [
        '**/.venv/**',
        '**/artifacts/**',
        '**/.tmp/**',
        '**/.pytest_cache/**',
        '**/.ruff_cache/**',
      ],
    },
  },
});
