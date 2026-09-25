import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import type { GameAction, GameState, PlayerId, Suit } from '../../engine/types';
import {
  committedHighlightTargets,
  highlightTargetKey,
  resourceGainSuits,
  sharedActionHighlightTargets,
  type HighlightTarget,
} from '../actionHighlights';
import {
  actionsForOpenPicker,
  type ActionPickerState,
} from '../actionPickerModel';

type HoverSource = 'menu' | 'picker';

const HighlightContext = createContext<{
  keys: ReadonlySet<string>;
  targets: readonly HighlightTarget[];
  hover: (actions: readonly GameAction[], source: HoverSource) => void;
  clear: () => void;
}>({ keys: new Set(), targets: [], hover: () => {}, clear: () => {} });

export function ActionHighlights({
  state,
  picker,
  legalActions,
  humanPlayerId,
  committedAction,
  committedActingPlayerId,
  children,
}: {
  state: GameState;
  picker: ActionPickerState | null;
  legalActions: readonly GameAction[];
  humanPlayerId: PlayerId;
  committedAction?: GameAction | null;
  committedActingPlayerId?: PlayerId | null;
  children: ReactNode;
}) {
  const [hovered, setHovered] = useState<{
    actions: readonly GameAction[];
    state: GameState;
    picker: ActionPickerState | null;
    source: HoverSource;
  } | null>(null);
  // Only the human player's own confirmed action keeps its hand card and
  // destination ghost live while the animation plays, so the card stays on top
  // from hover through the effect.
  const committedTargets = useMemo(
    () =>
      committedAction
        ? committedHighlightTargets(committedAction, {
            includePlacement: committedActingPlayerId === humanPlayerId,
          })
        : [],
    [committedAction, committedActingPlayerId, humanPlayerId]
  );
  const targets = useMemo(() => {
    const withCommitted = (
      list: readonly HighlightTarget[]
    ): HighlightTarget[] => [
      ...new Map(
        [...list, ...committedTargets].map((target) => [
          highlightTargetKey(target),
          target,
        ])
      ).values(),
    ];
    const persistent = sharedActionHighlightTargets(
      picker ? actionsForOpenPicker(picker, legalActions) : []
    );
    if (hovered?.state !== state || hovered.picker !== picker)
      return withCommitted(persistent);
    if (hovered.actions.some((action) => action.type === 'end-turn'))
      return withCommitted([]);
    const temporary = sharedActionHighlightTargets(hovered.actions);
    // A picker option can preview replacing a selected district or trade source.
    if (hovered.source === 'picker' && temporary.length > 0)
      return withCommitted(temporary);
    return withCommitted([...persistent, ...temporary]);
  }, [hovered, state, picker, legalActions, committedTargets]);
  const keys = useMemo(
    () => new Set(targets.map(highlightTargetKey)),
    [targets]
  );
  return (
    <HighlightContext.Provider
      value={{
        keys,
        targets,
        hover: (actions, source) =>
          setHovered({ actions, state, picker, source }),
        clear: () => setHovered(null),
      }}
    >
      {children}
    </HighlightContext.Provider>
  );
}

export function useActionHover(source: HoverSource = 'menu') {
  const { hover, clear } = useContext(HighlightContext);
  return (actions: readonly GameAction[]) => ({
    onMouseEnter: () => hover(actions, source),
    onMouseLeave: clear,
    onClickCapture: clear,
  });
}

export function useHighlightClass() {
  const { keys } = useContext(HighlightContext);
  return (target: HighlightTarget, enabled = true): string =>
    enabled && keys.has(highlightTargetKey(target))
      ? ' is-action-highlighted'
      : '';
}

export function useResourceHighlightClass() {
  const { keys } = useContext(HighlightContext);
  return (suit: Suit, enabled = true): string => {
    if (!enabled) {
      return '';
    }
    const highlighted =
      keys.has(highlightTargetKey({ kind: 'resource', suit, effect: 'gain' })) ||
      keys.has(
        highlightTargetKey({ kind: 'resource', suit, effect: 'spend' })
      );
    return highlighted ? ' is-action-highlighted' : '';
  };
}

export function useResourceGainSuits(): ReadonlySet<Suit> {
  const { targets } = useContext(HighlightContext);
  return useMemo(() => resourceGainSuits(targets), [targets]);
}

export function usePlacementGhost(districtId: string) {
  const { targets } = useContext(HighlightContext);
  const target = targets.find(
    (target) =>
      target.kind === 'district-lane' && target.districtId === districtId
  );
  return target?.kind === 'district-lane' ? target : undefined;
}
