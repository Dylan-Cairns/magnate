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
        const scaleX =
          flight.startWidth > 0 && Number.isFinite(flight.endWidth)
            ? flight.endWidth / flight.startWidth
            : 1;
        const scaleY =
          flight.startHeight > 0 && Number.isFinite(flight.endHeight)
            ? flight.endHeight / flight.startHeight
            : 1;
        return (
          <div
            key={flight.id}
            className={`card-flight${flight.variant === 'draw' ? ' is-draw' : ''}${flight.variant === 'terminal-clear' ? ' is-terminal-clear' : ''}`}
            style={
              {
                '--card-flight-start-x': `${flight.startX}px`,
                '--card-flight-start-y': `${flight.startY}px`,
                '--card-flight-dx': `${dx}px`,
                '--card-flight-dy': `${dy}px`,
                '--card-flight-delay-ms': `${flight.delayMs}ms`,
                '--card-flight-duration-ms': `${flight.durationMs ?? CARD_FLIGHT_DURATION_MS}ms`,
                '--card-flight-scale-x': `${Number.isFinite(scaleX) ? scaleX : 1}`,
                '--card-flight-scale-y': `${Number.isFinite(scaleY) ? scaleY : 1}`,
                width: `${flight.startWidth}px`,
                height: `${flight.startHeight}px`,
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
