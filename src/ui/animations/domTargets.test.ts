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

  it('prefers explicit deck and discard animation anchors', () => {
    const deckAnchor = makeElement();
    const discardAnchor = makeElement();
    const deckFallback = makeElement();
    const discardFallback = makeElement();
    const deckStack = makeElement({
      queries: new Map([['.deck-pile-animation-anchor', deckAnchor]]),
      queryAll: new Map([['.deck-pile-stack-card', [deckFallback]]]),
    });
    const discardStack = makeElement({
      queries: new Map([['.discard-pile-animation-anchor', discardAnchor]]),
      queryAll: new Map([['.deck-pile-stack-card', [discardFallback]]]),
    });
    const targets = createAnimationDomTargets(
      makeEnvironment(
        new Map([
          ['.deck-pile-stack.is-deck', deckStack],
          ['.deck-pile-stack.is-discard', discardStack],
        ])
      )
    );

    expect(targets.deckSource()).toBe(deckAnchor);
    expect(targets.discardTarget()).toBe(discardAnchor);
  });

  it('prefers the centered hidden-hand animation anchor', () => {
    const leftHiddenCard = makeElement();
    const middleHiddenCard = makeElement();
    const rightHiddenCard = makeElement();
    const anchor = makeElement();
    const hiddenSelector =
      '.player-panel[data-player-id="PlayerB"] ' +
      '.card-tile[data-hand-owner-id="PlayerB"][data-hand-slot-kind="hidden"]';
    const anchorSelector =
      '.player-panel[data-player-id="PlayerB"] ' +
      '[data-hand-owner-id="PlayerB"][data-hand-animation-anchor="true"]';

    const targetsWithoutAnchor = createAnimationDomTargets(
      makeEnvironment(
        new Map(),
        new Map([
          [hiddenSelector, [leftHiddenCard, middleHiddenCard, rightHiddenCard]],
        ])
      )
    );
    expect(targetsWithoutAnchor.handSource('PlayerB', '6')).toBe(
      middleHiddenCard
    );

    const targetsWithAnchor = createAnimationDomTargets(
      makeEnvironment(
        new Map([[anchorSelector, anchor]]),
        new Map([
          [hiddenSelector, [leftHiddenCard, middleHiddenCard, rightHiddenCard]],
        ])
      )
    );
    expect(targetsWithAnchor.handSource('PlayerB', '6')).toBe(anchor);
    expect(targetsWithAnchor.handDrawTarget('PlayerB')).toBe(anchor);
  });

  it('measures the board card size from lane CSS variables', () => {
    const lane = makeElement({ width: 300, height: 500 });
    const fallback = makeElement({ width: 80, height: 120 });
    const targets = createAnimationDomTargets(
      makeEnvironment(
        new Map(),
        new Map(),
        new Map([
          ['--card-width', '96px'],
          ['--card-height', '140px'],
        ])
      )
    );

    expect(targets.laneCardSize(lane, fallback)).toEqual({
      width: 96,
      height: 140,
    });
  });

  it('uses the lane layout target for the next card size and center', () => {
    const target = makeElement({
      left: 120,
      top: 80,
      width: 96,
      height: 140,
    });
    const lane = makeElement({
      queries: new Map([['.lane-card-animation-target', target]]),
    });
    const fallback = makeElement({ width: 80, height: 120 });
    const targets = createAnimationDomTargets(makeEnvironment());

    expect(targets.laneCardSize(lane, fallback)).toEqual({
      width: 96,
      height: 140,
    });
    expect(targets.laneTargetCenter(lane, 120)).toEqual({ x: 168, y: 150 });
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
  queries: ReadonlyMap<string, HTMLElement> = new Map(),
  queryAll: ReadonlyMap<string, readonly HTMLElement[]> = new Map(),
  computedValues: ReadonlyMap<string, string> = new Map()
): AnimationDomEnvironment {
  return {
    isAvailable: () => true,
    querySelector: <T extends Element>(selector: string) =>
      (queries.get(selector) as T | undefined) ?? null,
    querySelectorAll: <T extends Element>(selector: string) =>
      (queryAll.get(selector) ?? []) as unknown as readonly T[],
    createElement: () => makeElement(),
    getComputedStyle: () =>
      ({
        getPropertyValue: (propertyName: string) =>
          computedValues.get(propertyName) ?? '',
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
  const style = {} as CSSStyleDeclaration;
  return {
    style,
    classList: {
      contains: () => false,
    },
    appendChild: () => undefined,
    remove: () => undefined,
    getBoundingClientRect: () =>
      makeRect(
        left,
        top,
        parseStylePixelValue(style.width, width),
        parseStylePixelValue(style.height, height)
      ),
    querySelector: <T extends Element>(selector: string) =>
      (queries.get(selector) as T | undefined) ?? null,
    querySelectorAll: <T extends Element>(selector: string) =>
      (queryAll.get(selector) ?? []) as unknown as readonly T[],
  } as unknown as HTMLElement;
}

function parseStylePixelValue(value: string, fallback: number): number {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
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
