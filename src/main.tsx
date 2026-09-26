import { lazy, StrictMode, Suspense } from 'react';
import { createRoot } from 'react-dom/client';

// Self-hosted fonts (no CDN). Latin subsets only, weights in use.
import '@fontsource/averia-serif-libre/latin-400.css';
import '@fontsource/averia-serif-libre/latin-700.css';
import '@fontsource/alegreya-sans/latin-400.css';
import '@fontsource/alegreya-sans/latin-500.css';
import '@fontsource/alegreya-sans/latin-700.css';

import './styles.css';

const LazyApp = lazy(() =>
  import('./App').then((module) => ({ default: module.App }))
);

function BootstrapShell() {
  return (
    <div className="app-bootstrap-shell" role="status" aria-live="polite">
      <section className="app-bootstrap-card">
        <h1 className="app-bootstrap-title">Magnate</h1>
        <p className="app-bootstrap-copy">Loading game interface...</p>
        <div className="app-bootstrap-bar" aria-hidden="true" />
      </section>
    </div>
  );
}

const root = document.getElementById('root');
if (!root) {
  throw new Error('Missing #root element in index.html.');
}

createRoot(root).render(
  <StrictMode>
    <Suspense fallback={<BootstrapShell />}>
      <LazyApp />
    </Suspense>
  </StrictMode>
);
