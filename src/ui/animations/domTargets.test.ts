import { describe, expect, it } from 'vitest';

import {
  createAnimationDomTargets,
  deedTokenCenterInRail,
  elementCenter,
  parsePixelValue,
  type AnimationDomEnvironment,
} from './domTargets';

describe('domTargets', () => {
  it('selects the first visible resource token from the board pane', () => {
    const hidden = makeElement({ width: 0, height: 0 });
    const visible = makeElement({ left: 20, top: 40, width: 12, height: 12 });
    const selector =
      '[data-token-rail-player-id="PlayerA"] .rail-resources-row ' +
      '.token-chip[data-token-suit="Moons"]';
    const boardPane = makeElement({
      queryAll: new Map([[selector, [hidden, visible]]]),
    });
    const targets = createAnimationDomTargets(
      makeEnvironment(new Map([['.board-pane', boardPane]]))
    );

    expect(targets.resourceToken('PlayerA', 'Moons')).toBe(visible);
  });

  it('preserves hand source fallback order and deck top-card selection', () => {
    const hiddenHandCard = makeElement();
    const stackBottom = makeElement();
    const stackTop = makeElement();
    const deckStack = makeElement({
      queryAll: new Map([['.deck-pile-stack-card', [stackBottom, stackTop]]]),
    });
    const hiddenSelector =
      '.player-panel[data-player-id="PlayerA"] ' +
      '.card-tile[data-hand-owner-id="PlayerA"][data-hand-slot-kind="hidden"]';
    const targets = createAnimationDomTargets(
      makeEnvironment(
        new Map([
          [hiddenSelector, hiddenHandCard],
          ['.deck-pile-stack.is-deck', deckStack],
        ])
      )
    );

    expect(targets.handSource('PlayerA', '6')).toBe(hiddenHandCard);
    expect(targets.deckSource()).toBe(stackTop);
  });

  it('targets an empty human lane from the top of its frame', () => {
    const frame = makeElement({ left: 100, top: 200, width: 180, height: 300 });
    const lane = makeElement({
      queries: new Map([['.lane-stack-frame', frame]]),
    });
    const targets = createAnimationDomTargets(makeEnvironment());

    expect(targets.laneTargetCenter(lane, 80)).toEqual({
      x: 190,
      y: 240,
    });
  });

  it('keeps terminal deed lookup narrower than transfer lookup', () => {
    const genericCard = makeElement();
    const selector = '.district-strip .card-tile[data-card-id="6"]';
    const targets = createAnimationDomTargets(
      makeEnvironment(new Map([[selector, genericCard]]))
    );

    expect(targets.terminalDevelopingCard('6')).toBeNull();
    expect(targets.developingCard('6')).toBe(genericCard);
  });

  it('centers deed token slots around the rail midpoint', () => {
    const rail = makeElement({ left: 10, top: 20, width: 30, height: 100 });

    expect(elementCenter(rail)).toEqual({ x: 25, y: 70 });
    expect(deedTokenCenterInRail(rail, 20, 4, 0, 2)).toEqual({
      x: 25,
      y: 58,
    });
    expect(deedTokenCenterInRail(rail, 20, 4, 1, 2)).toEqual({
      x: 25,
      y: 82,
    });
    expect(parsePixelValue('', 2.56)).toBe(2.56);
  });
});

function makeEnvironment(
  queries: ReadonlyMap<string, HTMLElement> = new Map()
): AnimationDomEnvironment {
  return {
    isAvailable: () => true,
    querySelector: <T extends Element>(selector: string) =>
      (queries.get(selector) as T | undefined) ?? null,
    querySelectorAll: () => [],
    createElement: () => makeElement(),
    getComputedStyle: () =>
      ({
        getPropertyValue: () => '',
      }) as unknown as CSSStyleDeclaration,
    viewportHeight: () => 1000,
  };
}

function makeElement({
  left = 0,
  top = 0,
  width = 10,
  height = 10,
  queries = new Map(),
  queryAll = new Map(),
}: {
  left?: number;
  top?: number;
  width?: number;
  height?: number;
  queries?: ReadonlyMap<string, HTMLElement>;
  queryAll?: ReadonlyMap<string, readonly HTMLElement[]>;
} = {}): HTMLElement {
  return {
    classList: {
      contains: () => false,
    },
    getBoundingClientRect: () => makeRect(left, top, width, height),
    querySelector: <T extends Element>(selector: string) =>
      (queries.get(selector) as T | undefined) ?? null,
    querySelectorAll: <T extends Element>(selector: string) =>
      (queryAll.get(selector) ?? []) as unknown as readonly T[],
  } as unknown as HTMLElement;
}

function makeRect(
  left: number,
  top: number,
  width: number,
  height: number
): DOMRect {
  return {
    bottom: top + height,
    height,
    left,
    right: left + width,
    top,
    width,
    x: left,
    y: top,
    toJSON: () => ({}),
  };
}
