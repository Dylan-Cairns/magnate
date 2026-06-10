import type { Suit } from '../../engine/types';
import { SUIT_ICON_BY_SUIT } from '../suitIcons';
import { SUIT_TOKEN_BG } from './TokenComponents';
import React from 'react';

// Clockwise from top-left
const SUITS: Suit[] = ['Moons', 'Suns', 'Waves', 'Leaves', 'Wyrms', 'Knots'];

// [suitA, suitB, ranks] — numeral cards (ranks 2–9) only.
// Three pairs have no numeral cards and are omitted: Moons/Wyrms, Suns/Leaves, Waves/Knots.
const EDGES: [Suit, Suit, string][] = [
  ['Moons', 'Suns', '4 8 9'],
  ['Suns', 'Waves', '5'],
  ['Waves', 'Leaves', '2 4 8'],
  ['Leaves', 'Wyrms', '3'],
  ['Wyrms', 'Knots', '4 5 8'],
  ['Knots', 'Moons', '2'],
  ['Moons', 'Waves', '3 6'],
  ['Waves', 'Wyrms', '7 9'],
  ['Leaves', 'Knots', '6 9'],
  ['Knots', 'Suns', '3 7'],
  ['Leaves', 'Moons', '5 7'],
  ['Suns', 'Wyrms', '2 6'],
];

const CX = 170;
const CY = 155;
const R = 120;
const NODE_R = 27;

function nodePos(i: number): [number, number] {
  const rad = ((-30 + 60 * i) * Math.PI) / 180;
  return [CX + R * Math.sin(rad), CY - R * Math.cos(rad)];
}

function labelPos(
  [x1, y1]: [number, number],
  [x2, y2]: [number, number]
): [number, number] {
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dcx = mx - CX;
  const dcy = my - CY;
  const dist = Math.sqrt(dcx * dcx + dcy * dcy);
  if (dist < 2) {
    // Both diameter edges share the center as midpoint; place at 1/3 from
    // the first endpoint so the two labels don't overlap.
    const t = 0.4;
    return [x1 + (x2 - x1) * t, y1 + (y2 - y1) * t];
  }
  const push = 0;
  return [mx + (dcx / dist) * push, my + (dcy / dist) * push];
}

function lineAngleDeg(
  [x1, y1]: [number, number],
  [x2, y2]: [number, number]
): number {
  let angle = Math.atan2(y2 - y1, x2 - x1) * (180 / Math.PI);
  // Keep text upright — flip if it would render right-to-left
  if (angle > 90) angle -= 180;
  if (angle < -90) angle += 180;
  return angle;
}

const positions = SUITS.map((_, i) => nodePos(i));
const suitIndex = new Map<Suit, number>(SUITS.map((s, i) => [s, i]));

export function DecktetSuitDiagram() {
  return (
    <section className="panel suit-diagram-panel">
      <h2>Deck Map</h2>
      <svg
        viewBox="0 0 340 310"
        className="suit-diagram"
        aria-label="Decktet suit pair distribution"
      >
        {/* Lines in a compositing group so overlapping lines don't brighten */}
        <g opacity={0.08}>
          {EDGES.map(([suitA, suitB], i) => {
            const p1 = positions[suitIndex.get(suitA)!];
            const p2 = positions[suitIndex.get(suitB)!];
            return (
              <line
                key={i}
                x1={p1[0]}
                y1={p1[1]}
                x2={p2[0]}
                y2={p2[1]}
                stroke="white"
                strokeWidth={5}
              />
            );
          })}
        </g>
        {EDGES.map(([suitA, suitB, label], i) => {
          const p1 = positions[suitIndex.get(suitA)!];
          const p2 = positions[suitIndex.get(suitB)!];
          const [lx, ly] = labelPos(p1, p2);
          const angle = lineAngleDeg(p1, p2);
          return (
            <text
              key={i}
              x={lx}
              y={ly}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={15}
              fontFamily="inherit"
              transform={`rotate(${angle}, ${lx}, ${ly})`}
              style={
                {
                  paintOrder: 'stroke fill',
                  stroke: '#05080f',
                  strokeWidth: '4',
                  strokeLinejoin: 'round',
                  fill: '#fafcff',
                } as React.CSSProperties
              }
            >
              {label}
            </text>
          );
        })}
        {SUITS.map((suit, i) => {
          const [x, y] = positions[i];
          return (
            <g key={suit}>
              <circle
                cx={x}
                cy={y}
                r={NODE_R}
                fill={SUIT_TOKEN_BG[suit]}
                stroke="rgba(255,255,255,0.15)"
                strokeWidth={1}
              />
              <image
                href={SUIT_ICON_BY_SUIT[suit]}
                x={x - NODE_R + 2}
                y={y - NODE_R + 2}
                width={NODE_R * 2 - 4}
                height={NODE_R * 2 - 4}
              />
            </g>
          );
        })}
      </svg>
    </section>
  );
}
