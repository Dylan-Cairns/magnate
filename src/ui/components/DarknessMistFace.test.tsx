import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { DarknessMistFace } from './DarknessMistFace';

describe('DarknessMistFace', () => {
  it('renders the settled mist without animation when stir is disabled', () => {
    const html = renderToStaticMarkup(
      <DarknessMistFace animationsEnabled stirEnabled={false} />
    );
    expect(html).toContain('data-face-effect="darkness"');
    expect(html).toContain('darkness-fog__base');
    expect(html).toContain('darkness-fog__cloud');
    expect(html).not.toContain('is-animated');
  });

  it('animates the drift when stir and animations are enabled', () => {
    const html = renderToStaticMarkup(
      <DarknessMistFace animationsEnabled stirEnabled />
    );
    expect(html).toContain('darkness-fog__drift is-animated');
    expect(html).toContain('darkness-fog__cloud darkness-fog__cloud-a');
    expect(html).toContain('darkness-fog__cloud darkness-fog__cloud-b');
  });

  it('does not animate when the animations preference is off', () => {
    const html = renderToStaticMarkup(
      <DarknessMistFace animationsEnabled={false} stirEnabled />
    );
    expect(html).toContain('darkness-fog__drift');
    expect(html).not.toContain('is-animated');
  });

  it('drives the stir duration from the shared timing constant', () => {
    const html = renderToStaticMarkup(
      <DarknessMistFace animationsEnabled stirEnabled />
    );
    expect(html).toContain('--darkness-stir-duration:5000ms');
  });

  it('shares one animation class regardless of the stir signal', () => {
    const mounted = renderToStaticMarkup(
      <DarknessMistFace animationsEnabled stirEnabled />
    );
    const signalled = renderToStaticMarkup(
      <DarknessMistFace
        animationsEnabled
        stirEnabled
        stirSignal="sell-card:27"
      />
    );
    // The signal only changes the restart token; it never changes the classes.
    expect(mounted).toContain('darkness-fog__drift is-animated');
    expect(signalled).toContain('darkness-fog__drift is-animated');
  });
});
