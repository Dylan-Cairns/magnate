import type { CSSProperties } from 'react';

import { RESOURCE_FLIGHT_DURATION_MS } from '../animations/timing';
import type { ResourceFlight } from '../animations/types';
import { TokenChip } from './TokenComponents';

export function ResourceFlightLayer({
  flights,
}: {
  flights: readonly ResourceFlight[];
}) {
  if (flights.length === 0) {
    return null;
  }

  return (
    <div className="resource-flight-layer" aria-hidden="true">
      {flights.map((flight) => {
        const dx = flight.endX - flight.startX;
        const dy = flight.endY - flight.startY;
        const variantClass =
          flight.variant === 'tax-loss'
            ? ' is-tax-loss'
            : flight.variant === 'payment'
              ? ' is-payment'
              : ' is-transfer';
        return (
          <div
            key={flight.id}
            className={`resource-flight${variantClass}`}
            style={
              {
                '--resource-flight-start-x': `${flight.startX}px`,
                '--resource-flight-start-y': `${flight.startY}px`,
                '--resource-flight-dx': `${dx}px`,
                '--resource-flight-dy': `${dy}px`,
                '--resource-flight-delay-ms': `${flight.delayMs}ms`,
                '--resource-flight-duration-ms': `${flight.durationMs ?? RESOURCE_FLIGHT_DURATION_MS}ms`,
              } as CSSProperties
            }
          >
            <TokenChip
              suit={flight.suit}
              count={1}
              compact
              className="resource-flight-chip"
            />
          </div>
        );
      })}
    </div>
  );
}
