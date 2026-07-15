import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';

import { actionStableKey } from '../engine/actionSurface';
import {
  legalActionsForDecisionPlayer,
  toDecisionPlayerView,
} from '../engine/decisionActor';
import { rngFromSeed } from '../engine/rng';
import { createPolicyFromBotSpec, type BotSpec } from '../policies/botSpec';
import { getBotProfile } from '../policies/catalog';
import {
  scoreHeuristicV2Actions,
  selectHeuristicV2Action,
} from '../policies/heuristicScorerV2';
import { POLICY_RANDOM_SCHEME_VERSION } from '../policies/policyRandom';
import type {
  ActionPolicy,
  SearchDecisionDiagnostics,
} from '../policies/types';
import {
  strategicActionDeltasV0,
  strategicStateSummaryV0,
  type StrategicActionDeltaV0,
  type StrategicStateSummaryV0,
} from '../policies/strategicStateSummary';
import {
  STRATEGIC_POSITION_CATALOG_VERSION,
  createStrategicPositionCatalogV0,
  type StrategicPositionThemeV0,
  type StrategicPositionV0,
} from './strategicPositionCatalog';

export const STRATEGIC_POSITION_COMPARISON_SCHEMA_VERSION = 1 as const;
export const STRATEGIC_POSITION_COMPARISON_SEED_SCHEME =
  'strategic-position-common-random-v1' as const;
export const STRATEGIC_POSITION_FINGERPRINT_SCHEME =
  'sha256-canonical-json-v1' as const;
export const STRATEGIC_TD_800_VISIT_VARIANT_ID =
  'td-root-search-v2-800-visits' as const;
export const STRATEGIC_TD_800_HEURISTIC_ROOT_VARIANT_ID =
  'td-root-search-v2-800-visits-heuristic-root' as const;
export const STRATEGIC_TD_800_HEURISTIC_ROLLOUT_VARIANT_ID =
  'td-root-search-v2-800-visits-heuristic-rollout' as const;
export const STRATEGIC_TD_800_HEURISTIC_ROOT_ROLLOUT_VARIANT_ID =
  'td-root-search-v2-800-visits-heuristic-root-rollout' as const;

const STRATEGIC_DIAGNOSTIC_VARIANT_IDS = new Set<string>([
  STRATEGIC_TD_800_VISIT_VARIANT_ID,
  STRATEGIC_TD_800_HEURISTIC_ROOT_VARIANT_ID,
  STRATEGIC_TD_800_HEURISTIC_ROLLOUT_VARIANT_ID,
  STRATEGIC_TD_800_HEURISTIC_ROOT_ROLLOUT_VARIANT_ID,
]);

export type StrategicVariantDescriptorV0 =
  | {
      readonly kind: 'heuristic-v2-direct';
      readonly id: string;
      readonly label: string;
    }
  | {
      readonly kind: 'bot-spec';
      readonly id: string;
      readonly label: string;
      readonly spec: BotSpec;
    }
  | {
      readonly kind: 'custom';
      readonly id: string;
      readonly label: string;
      readonly implementationId: string;
    };

export interface StrategicComparisonVariantV0 {
  readonly descriptor: StrategicVariantDescriptorV0;
  /** Required only for custom descriptors; built-ins are constructed here. */
  readonly policy?: ActionPolicy;
  readonly inspectHeuristicV2?: boolean;
}

export interface StrategicFocusSignalV0 {
  readonly focusActionId: string;
  readonly actionKey: string;
  readonly heuristicScore: number | null;
  readonly heuristicRank: number | null;
  readonly heuristicPrior: number | null;
  readonly searchVisits: number | null;
  readonly searchMeanValue: number | null;
  readonly searchPrior: number | null;
}

export interface StrategicVariantDecisionV0 {
  readonly variantId: string;
  readonly selectedActionKey: string;
  readonly selectedFocusActionId: string | null;
  readonly matchesExpectedPreference: boolean | null;
  readonly latencyMs: number;
  readonly focusSignals: readonly StrategicFocusSignalV0[];
  readonly searchDiagnostics: SearchDecisionDiagnostics | null;
}

export interface StrategicPositionRepetitionV0 {
  readonly repetition: number;
  readonly sharedRandomSeed: string;
  readonly decisions: readonly StrategicVariantDecisionV0[];
}

export interface StrategicPositionComparisonCaseV0 {
  readonly positionId: string;
  readonly positionFingerprint: string;
  readonly randomGroupId: string;
  readonly title: string;
  readonly theme: StrategicPositionThemeV0;
  readonly thesis: string;
  readonly expectedFacts: readonly string[];
  readonly expectedPreference: StrategicPositionV0['expectedPreference'];
  readonly focusActions: StrategicPositionV0['focusActions'];
  readonly legalActionKeys: readonly string[];
  readonly focusActionDeltas: readonly StrategicActionDeltaV0[];
  readonly stateSummary: StrategicStateSummaryV0;
  readonly repetitions: readonly StrategicPositionRepetitionV0[];
}

export interface StrategicPositionComparisonRunV0 {
  readonly schemaVersion: typeof STRATEGIC_POSITION_COMPARISON_SCHEMA_VERSION;
  readonly catalogVersion: typeof STRATEGIC_POSITION_CATALOG_VERSION;
  readonly seedScheme: typeof STRATEGIC_POSITION_COMPARISON_SEED_SCHEME;
  readonly positionFingerprintScheme: typeof STRATEGIC_POSITION_FINGERPRINT_SCHEME;
  readonly policyRandomSchemeVersion: string;
  readonly repetitionStart: number;
  readonly repetitions: number;
  readonly variants: readonly StrategicVariantDescriptorV0[];
  readonly positions: readonly StrategicPositionComparisonCaseV0[];
}

export interface StrategicPositionComparisonProgressV0 {
  readonly completedDecisions: number;
  readonly totalDecisions: number;
  readonly positionId: string;
  readonly repetition: number;
  readonly variantId: string;
  readonly selectedActionKey: string;
}

export interface StrategicPositionComparisonOptionsV0 {
  readonly positions?: readonly StrategicPositionV0[];
  readonly variants?: readonly StrategicComparisonVariantV0[];
  readonly repetitionStart?: number;
  readonly repetitions?: number;
  readonly now?: () => number;
  readonly onProgress?: (
    progress: StrategicPositionComparisonProgressV0
  ) => void;
}

const directHeuristicV2Policy: ActionPolicy = {
  selectAction(context) {
    return selectHeuristicV2Action(context);
  },
};

export function createStrategicComparisonVariantCatalogV0(): StrategicComparisonVariantV0[] {
  const hard = structuredClone(getBotProfile('rollout-search-v2-hard').spec);
  const td = structuredClone(getBotProfile('td-root-search-v2-medium').spec);
  if (td.kind !== 'td-root-search') {
    throw new Error('TD V2 Medium must use a TD-root-search bot spec.');
  }
  const td800 = {
    ...structuredClone(td),
    id: STRATEGIC_TD_800_VISIT_VARIANT_ID,
    config: {
      ...structuredClone(td.config),
      worlds: 50,
    },
  };
  const td800HeuristicRoot = createTd800GuidanceAblationSpec(
    td800,
    STRATEGIC_TD_800_HEURISTIC_ROOT_VARIANT_ID,
    {
      root: 'heuristic',
      rollout: 'td',
      leaf: 'td',
    }
  );
  const td800HeuristicRollout = createTd800GuidanceAblationSpec(
    td800,
    STRATEGIC_TD_800_HEURISTIC_ROLLOUT_VARIANT_ID,
    {
      root: 'td',
      rollout: 'heuristic',
      leaf: 'td',
    }
  );
  const td800HeuristicRootRollout = createTd800GuidanceAblationSpec(
    td800,
    STRATEGIC_TD_800_HEURISTIC_ROOT_ROLLOUT_VARIANT_ID,
    {
      root: 'heuristic',
      rollout: 'heuristic',
      leaf: 'td',
    }
  );
  return [
    {
      descriptor: {
        kind: 'heuristic-v2-direct',
        id: 'heuristic-v2-direct',
        label: 'Heuristic v2 direct',
      },
      inspectHeuristicV2: true,
    },
    {
      descriptor: {
        kind: 'bot-spec',
        id: hard.id,
        label: 'V2 Hard',
        spec: hard,
      },
    },
    {
      descriptor: {
        kind: 'bot-spec',
        id: td.id,
        label: 'Current TD V2 Medium',
        spec: td,
      },
    },
    {
      descriptor: {
        kind: 'bot-spec',
        id: td800.id,
        label: 'Current TD V2, 800 root visits',
        spec: td800,
      },
    },
    {
      descriptor: {
        kind: 'bot-spec',
        id: td800HeuristicRoot.id,
        label: 'TD V2, 800 visits, heuristic root',
        spec: td800HeuristicRoot,
      },
    },
    {
      descriptor: {
        kind: 'bot-spec',
        id: td800HeuristicRollout.id,
        label: 'TD V2, 800 visits, heuristic rollout',
        spec: td800HeuristicRollout,
      },
    },
    {
      descriptor: {
        kind: 'bot-spec',
        id: td800HeuristicRootRollout.id,
        label: 'TD V2, 800 visits, heuristic root + rollout',
        spec: td800HeuristicRootRollout,
      },
    },
  ];
}

export function createDefaultStrategicComparisonVariantsV0(): StrategicComparisonVariantV0[] {
  return createStrategicComparisonVariantCatalogV0().filter(
    (variant) => !STRATEGIC_DIAGNOSTIC_VARIANT_IDS.has(variant.descriptor.id)
  );
}

function createTd800GuidanceAblationSpec(
  baseline: Extract<BotSpec, { kind: 'td-root-search' }>,
  id: string,
  guidance: NonNullable<
    Extract<BotSpec, { kind: 'td-root-search' }>['guidance']
  >
): Extract<BotSpec, { kind: 'td-root-search' }> {
  return {
    ...structuredClone(baseline),
    id,
    guidance: structuredClone(guidance),
    config: {
      ...structuredClone(baseline.config),
      heuristic: 'v2',
    },
  };
}

export async function runStrategicPositionComparisonV0(
  options: StrategicPositionComparisonOptionsV0 = {}
): Promise<StrategicPositionComparisonRunV0> {
  const positions = options.positions ?? createStrategicPositionCatalogV0();
  const variants =
    options.variants ?? createDefaultStrategicComparisonVariantsV0();
  const repetitions = options.repetitions ?? 1;
  const repetitionStart = options.repetitionStart ?? 0;
  const now = options.now ?? (() => performance.now());
  if (!Number.isSafeInteger(repetitions) || repetitions <= 0) {
    throw new Error(
      'Strategic comparison repetitions must be a positive safe integer.'
    );
  }
  if (!Number.isSafeInteger(repetitionStart) || repetitionStart < 0) {
    throw new Error(
      'Strategic comparison repetitionStart must be a nonnegative safe integer.'
    );
  }
  if (repetitionStart > Number.MAX_SAFE_INTEGER - (repetitions - 1)) {
    throw new Error(
      'Strategic comparison repetition range exceeds safe integers.'
    );
  }
  if (positions.length === 0 || variants.length === 0) {
    throw new Error('Strategic comparison requires positions and variants.');
  }
  assertUniqueVariantIds(variants);
  const resolvedVariants = variants.map((variant) => ({
    variant,
    policy: resolveVariantPolicy(variant),
  }));

  const totalDecisions = positions.length * repetitions * variants.length;
  let completedDecisions = 0;
  const cases: StrategicPositionComparisonCaseV0[] = [];

  for (const position of positions) {
    const legalActions = legalActionsForDecisionPlayer(
      position.state,
      position.perspectivePlayerId
    );
    const legalActionKeys = legalActions
      .map(actionStableKey)
      .sort((left, right) => left.localeCompare(right));
    const focusKeys = new Set(
      position.focusActions.map((focus) => focus.actionKey)
    );
    const focusActionDeltas = strategicActionDeltasV0(
      position.state,
      position.perspectivePlayerId
    ).filter((delta) => focusKeys.has(delta.actionKey));
    const stateSummary = strategicStateSummaryV0(
      position.state,
      position.perspectivePlayerId
    );
    const positionFingerprint = strategicPositionFingerprint({
      position,
      legalActionKeys,
      focusActionDeltas,
    });
    const caseRepetitions: StrategicPositionRepetitionV0[] = [];

    for (let offset = 0; offset < repetitions; offset += 1) {
      const repetition = repetitionStart + offset;
      const sharedRandomSeed = strategicComparisonSeed(
        position.pairId ?? position.id,
        repetition
      );
      const decisions: StrategicVariantDecisionV0[] = [];
      for (const { variant, policy } of resolvedVariants) {
        const decision = await evaluateVariant({
          position,
          variant,
          policy,
          sharedRandomSeed,
          now,
        });
        decisions.push(decision);
        completedDecisions += 1;
        options.onProgress?.({
          completedDecisions,
          totalDecisions,
          positionId: position.id,
          repetition,
          variantId: variant.descriptor.id,
          selectedActionKey: decision.selectedActionKey,
        });
      }
      caseRepetitions.push({ repetition, sharedRandomSeed, decisions });
    }

    cases.push({
      positionId: position.id,
      positionFingerprint,
      randomGroupId: position.pairId ?? position.id,
      title: position.title,
      theme: position.theme,
      thesis: position.thesis,
      expectedFacts: [...position.expectedFacts],
      expectedPreference: position.expectedPreference
        ? structuredClone(position.expectedPreference)
        : null,
      focusActions: structuredClone(position.focusActions),
      legalActionKeys,
      focusActionDeltas,
      stateSummary,
      repetitions: caseRepetitions,
    });
  }

  return {
    schemaVersion: STRATEGIC_POSITION_COMPARISON_SCHEMA_VERSION,
    catalogVersion: STRATEGIC_POSITION_CATALOG_VERSION,
    seedScheme: STRATEGIC_POSITION_COMPARISON_SEED_SCHEME,
    positionFingerprintScheme: STRATEGIC_POSITION_FINGERPRINT_SCHEME,
    policyRandomSchemeVersion: POLICY_RANDOM_SCHEME_VERSION,
    repetitionStart,
    repetitions,
    variants: variants.map((variant) => structuredClone(variant.descriptor)),
    positions: cases,
  };
}

export function strategicComparisonSeed(
  randomGroupId: string,
  repetition: number
): string {
  return `${STRATEGIC_POSITION_COMPARISON_SEED_SCHEME}:catalog:${String(STRATEGIC_POSITION_CATALOG_VERSION)}:group:${randomGroupId}:repetition:${String(repetition)}`;
}

async function evaluateVariant({
  position,
  variant,
  policy,
  sharedRandomSeed,
  now,
}: {
  position: StrategicPositionV0;
  variant: StrategicComparisonVariantV0;
  policy: ActionPolicy;
  sharedRandomSeed: string;
  now: () => number;
}): Promise<StrategicVariantDecisionV0> {
  const state = structuredClone(position.state);
  const legalActions = legalActionsForDecisionPlayer(
    state,
    position.perspectivePlayerId
  );
  const legalActionKeys = new Set(legalActions.map(actionStableKey));
  const view = toDecisionPlayerView(state, position.perspectivePlayerId);
  const inspectHeuristicV2 =
    variant.descriptor.kind === 'heuristic-v2-direct' ||
    variant.inspectHeuristicV2 === true;
  const heuristicByKey = inspectHeuristicV2
    ? new Map(
        scoreHeuristicV2Actions(legalActions, {
          state,
          view,
          legalActions,
        }).map((candidate) => [candidate.actionKey, candidate])
      )
    : new Map();
  let searchDiagnostics: SearchDecisionDiagnostics | undefined;
  const startedAt = now();
  const selected = await policy.selectAction({
    state,
    view,
    legalActions,
    randomSeed: sharedRandomSeed,
    random: rngFromSeed(sharedRandomSeed),
    onSearchDiagnostics(diagnostics) {
      if (searchDiagnostics) {
        throw new Error(
          `Strategic variant ${variant.descriptor.id} emitted duplicate diagnostics for ${position.id}.`
        );
      }
      searchDiagnostics = structuredClone(diagnostics);
    },
  });
  const latencyMs = now() - startedAt;
  if (!selected) {
    throw new Error(
      `Strategic variant ${variant.descriptor.id} selected no action for ${position.id}.`
    );
  }
  const selectedActionKey = actionStableKey(selected);
  if (!legalActionKeys.has(selectedActionKey)) {
    throw new Error(
      `Strategic variant ${variant.descriptor.id} selected illegal action ${selectedActionKey} for ${position.id}.`
    );
  }

  const searchByKey = new Map(
    (searchDiagnostics?.rootActions ?? []).map((candidate) => [
      candidate.actionKey,
      candidate,
    ])
  );
  const selectedFocus = position.focusActions.find(
    (focus) => focus.actionKey === selectedActionKey
  );

  return {
    variantId: variant.descriptor.id,
    selectedActionKey,
    selectedFocusActionId: selectedFocus?.id ?? null,
    matchesExpectedPreference: preferenceMatch(
      position.expectedPreference,
      selectedFocus?.id
    ),
    latencyMs,
    focusSignals: position.focusActions.map((focus) => {
      const heuristic = heuristicByKey.get(focus.actionKey);
      const search = searchByKey.get(focus.actionKey);
      return {
        focusActionId: focus.id,
        actionKey: focus.actionKey,
        heuristicScore: heuristic?.score ?? null,
        heuristicRank: heuristic?.rank ?? null,
        heuristicPrior: heuristic?.prior ?? null,
        searchVisits: search?.visits ?? null,
        searchMeanValue: search?.meanValue ?? null,
        searchPrior: search?.prior ?? null,
      };
    }),
    searchDiagnostics: searchDiagnostics ?? null,
  };
}

function preferenceMatch(
  preference: StrategicPositionV0['expectedPreference'],
  selectedFocusActionId: string | undefined
): boolean | null {
  if (!preference || !selectedFocusActionId) {
    return null;
  }
  if (selectedFocusActionId === preference.preferredFocusActionId) {
    return true;
  }
  return preference.overFocusActionIds.includes(selectedFocusActionId)
    ? false
    : null;
}

function resolveVariantPolicy(
  variant: StrategicComparisonVariantV0
): ActionPolicy {
  const descriptor = variant.descriptor;
  if (descriptor.kind === 'custom') {
    if (!variant.policy) {
      throw new Error(
        `Strategic custom variant ${descriptor.id} is missing its policy implementation.`
      );
    }
    return variant.policy;
  }
  if (variant.policy) {
    throw new Error(
      `Strategic built-in variant ${descriptor.id} cannot override its declared policy.`
    );
  }
  if (descriptor.kind === 'heuristic-v2-direct') {
    return directHeuristicV2Policy;
  }
  if (descriptor.id !== descriptor.spec.id) {
    throw new Error(
      `Strategic bot-spec variant id ${descriptor.id} does not match spec id ${descriptor.spec.id}.`
    );
  }
  return createPolicyFromBotSpec(structuredClone(descriptor.spec));
}

function strategicPositionFingerprint({
  position,
  legalActionKeys,
  focusActionDeltas,
}: {
  position: StrategicPositionV0;
  legalActionKeys: readonly string[];
  focusActionDeltas: readonly StrategicActionDeltaV0[];
}): string {
  const canonical = canonicalJson({
    catalogVersion: position.catalogVersion,
    id: position.id,
    title: position.title,
    theme: position.theme,
    perspectivePlayerId: position.perspectivePlayerId,
    thesis: position.thesis,
    expectedFacts: position.expectedFacts,
    pairId: position.pairId,
    state: position.state,
    focusActions: position.focusActions,
    expectedPreference: position.expectedPreference,
    ...(position.optionalityTrace
      ? { optionalityTrace: position.optionalityTrace }
      : {}),
    legalActionKeys,
    focusActionDeltas,
  });
  return `sha256:${createHash('sha256').update(canonical).digest('hex')}`;
}

function canonicalJson(value: unknown): string {
  if (value === null) {
    return 'null';
  }
  if (
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  ) {
    const encoded = JSON.stringify(value);
    if (encoded === undefined) {
      throw new Error('Strategic fingerprint cannot encode this value.');
    }
    return encoded;
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(',')}]`;
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter((entry) => entry[1] !== undefined)
      .sort(([left], [right]) => left.localeCompare(right));
    return `{${entries
      .map(
        ([key, entryValue]) =>
          `${JSON.stringify(key)}:${canonicalJson(entryValue)}`
      )
      .join(',')}}`;
  }
  throw new Error(
    `Strategic fingerprint cannot encode a ${typeof value} value.`
  );
}

function assertUniqueVariantIds(
  variants: readonly StrategicComparisonVariantV0[]
): void {
  const ids = new Set<string>();
  for (const variant of variants) {
    if (ids.has(variant.descriptor.id)) {
      throw new Error(
        `Strategic comparison has duplicate variant id ${variant.descriptor.id}.`
      );
    }
    ids.add(variant.descriptor.id);
  }
}
