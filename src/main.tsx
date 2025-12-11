import { lazy, StrictMode, Suspense } from 'react';
import { createRoot } from 'react-dom/client';

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
