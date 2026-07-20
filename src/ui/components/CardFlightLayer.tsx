import type { CSSProperties } from 'react';

import { CARD_FLIGHT_DURATION_MS } from '../animations/timing';
import type { CardFlight } from '../animations/types';
import { CardTile } from './CardTile';

export function CardFlightLayer({
  flights,
  animationsEnabled,
}: {
  flights: readonly CardFlight[];
  animationsEnabled: boolean;
}) {
  if (flights.length === 0) {
    return null;
  }

  return (
    <div className="card-flight-layer" aria-hidden="true">
      {flights.map((flight) => {
        const dx = flight.endX - flight.startX;
        const dy = flight.endY - flight.startY;
        const renderWidth =
          flight.renderWidth && flight.renderWidth > 0
            ? flight.renderWidth
            : flight.startWidth;
        const renderHeight =
          flight.renderHeight && flight.renderHeight > 0
            ? flight.renderHeight
            : flight.startHeight;
        const startScaleX =
          renderWidth > 0 && Number.isFinite(flight.startWidth)
            ? flight.startWidth / renderWidth
            : 1;
        const startScaleY =
          renderHeight > 0 && Number.isFinite(flight.startHeight)
            ? flight.startHeight / renderHeight
            : 1;
        const endScaleX =
          renderWidth > 0 && Number.isFinite(flight.endWidth)
            ? flight.endWidth / renderWidth
            : 1;
        const endScaleY =
          renderHeight > 0 && Number.isFinite(flight.endHeight)
            ? flight.endHeight / renderHeight
            : 1;
        return (
          <div
            key={flight.id}
            className={`card-flight${flight.variant === 'draw' ? ' is-draw' : ''}`}
            style={
              {
                '--card-flight-start-x': `${flight.startX}px`,
                '--card-flight-start-y': `${flight.startY}px`,
                '--card-flight-dx': `${dx}px`,
                '--card-flight-dy': `${dy}px`,
                '--card-flight-delay-ms': `${flight.delayMs}ms`,
                '--card-flight-duration-ms': `${flight.durationMs ?? CARD_FLIGHT_DURATION_MS}ms`,
                '--card-flight-start-scale-x': `${Number.isFinite(startScaleX) ? startScaleX : 1}`,
                '--card-flight-start-scale-y': `${Number.isFinite(startScaleY) ? startScaleY : 1}`,
                '--card-flight-end-scale-x': `${Number.isFinite(endScaleX) ? endScaleX : 1}`,
                '--card-flight-end-scale-y': `${Number.isFinite(endScaleY) ? endScaleY : 1}`,
                width: `${renderWidth}px`,
                height: `${renderHeight}px`,
              } as CSSProperties
            }
          >
            {flight.visual === 'face' && flight.cardId ? (
              <CardTile
                cardId={flight.cardId}
                perspective={flight.perspective}
                inDevelopment={flight.isDeed}
                animateDeedProgress={animationsEnabled}
              />
            ) : (
              <CardTile hidden />
            )}
          </div>
        );
      })}
    </div>
  );
}
