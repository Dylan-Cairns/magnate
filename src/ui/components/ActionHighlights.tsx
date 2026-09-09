import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import type { GameAction, GameState } from '../../engine/types';
import {
  highlightTargetKey,
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
  children,
}: {
  state: GameState;
  picker: ActionPickerState | null;
  legalActions: readonly GameAction[];
  children: ReactNode;
}) {
  const [hovered, setHovered] = useState<{
    actions: readonly GameAction[];
    state: GameState;
    picker: ActionPickerState | null;
    source: HoverSource;
  } | null>(null);
  const targets = useMemo(() => {
    const persistent = sharedActionHighlightTargets(
      picker ? actionsForOpenPicker(picker, legalActions) : []
    );
    if (hovered?.state !== state || hovered.picker !== picker)
      return persistent;
    if (hovered.actions.some((action) => action.type === 'end-turn')) return [];
    const temporary = sharedActionHighlightTargets(hovered.actions);
    // A picker option can preview replacing a selected district or trade source.
    if (hovered.source === 'picker' && temporary.length > 0) return temporary;
    return [
      ...new Map(
        [...persistent, ...temporary].map((target) => [
          highlightTargetKey(target),
          target,
        ])
      ).values(),
    ];
  }, [hovered, state, picker, legalActions]);
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

export function usePlacementGhost(districtId: string) {
  const { targets } = useContext(HighlightContext);
  const target = targets.find(
    (target) =>
      target.kind === 'district-lane' && target.districtId === districtId
  );
  return target?.kind === 'district-lane' ? target : undefined;
}
