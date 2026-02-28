import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

import { legalActions } from './engine/actionBuilders';
import { CARD_BY_ID, type CardId } from './engine/cards';
import { rngFromSeed } from './engine/rng';
import { createSession, stepToDecision } from './engine/session';
import {
  districtWinnersByPlayer,
  isTerminal,
  scoreLive,
} from './engine/scoring';
import type { BotProfileId } from './policies/catalog';
import {
  BOT_PROFILES,
  DEFAULT_BOT_PROFILE_ID,
  resolveBotProfile,
} from './policies/catalog';
import type {
  FinalScore,
  GameAction,
  GameState,
  PlayerId,
  Suit,
} from './engine/types';
import { toPlayerView } from './engine/view';
import {
  actionStableKey,
  buildHumanActionList,
  buildTradeSourceGroups,
  buildPickerOptions,
  cardSummary,
  describeAction,
  formatTokens,
  pickerStillLegal,
  pickerTitle,
  type ActionPickerQuery,
  type HumanActionListItem,
} from './ui/actionPresentation';
import {
  canUseTurnReset,
  shouldCaptureTurnResetAnchor,
  type TurnResetAnchor,
} from './ui/turnReset';
import { getCardImage } from './ui/cardImages';
import { DistrictColumn, PlayerTokenRail } from './ui/components/DistrictBoard';
import { PlayerPanel } from './ui/components/PlayerPanel';
import { RollResult } from './ui/components/RollResult';
import { TokenChip } from './ui/components/TokenComponents';
import { useDismissableLayer } from './ui/hooks/useDismissableLayer';
import {
  SUIT_TEXT_TOKEN,
  SUIT_TOKEN_REGEX,
  SUIT_TOKEN_TO_SUIT,
} from './ui/suitIcons';

const HUMAN_PLAYER: PlayerId = 'PlayerA';
const BOT_PLAYER: PlayerId = 'PlayerB';
const BOT_DELAY_MS = 450;
const PLAYER_HAND_SLOT_COUNT = 3;
const TRADE_POPOVER_WIDTH_PX = 220;
const TRADE_POPOVER_MIN_HEIGHT_PX = 188;
const TRADE_POPOVER_GAP_PX = 8;
const VIEWPORT_PADDING_PX = 10;

type ActionPickerState =
  | (ActionPickerQuery & {
      top: number;
      left: number;
    })
  | {
      kind: 'trade-source';
      top: number;
      left: number;
    };

function makeSeed(): string {
  return `seed-${Date.now()}`;
}

function createInitialState(seed: string): GameState {
  return createSession(seed, HUMAN_PLAYER);
}

function botRandomForState(state: GameState, profileId: string): () => number {
  return rngFromSeed(
    `${state.seed}:bot:${profileId}:turn:${state.turn}:phase:${state.phase}:log:${state.log.length}:actor:${state.activePlayerIndex}`
  );
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export function App() {
  const [seedInput, setSeedInput] = useState<string>(() => makeSeed());
  const [botProfileId, setBotProfileId] = useState<BotProfileId>(
    DEFAULT_BOT_PROFILE_ID
  );
  const [state, setState] = useState<GameState>(() =>
    createInitialState(seedInput)
  );
  const [error, setError] = useState<string | null>(null);
  const [botThinking, setBotThinking] = useState<boolean>(false);
  const [actionPicker, setActionPicker] = useState<ActionPickerState | null>(
    null
  );
  const [optionsMenuOpen, setOptionsMenuOpen] = useState<boolean>(false);
  const [turnResetAnchor, setTurnResetAnchor] =
    useState<TurnResetAnchor | null>(null);

  const stateRef = useRef(state);
  const actionPopoverRef = useRef<HTMLElement | null>(null);
  const optionsMenuRef = useRef<HTMLElement | null>(null);
  const optionsMenuButtonRef = useRef<HTMLButtonElement | null>(null);
  const closeActionPicker = useCallback(() => setActionPicker(null), []);
  const closeOptionsMenu = useCallback(() => setOptionsMenuOpen(false), []);
  const actionPopoverLayerRefs = useMemo(() => [actionPopoverRef], []);
  const optionsMenuLayerRefs = useMemo(
    () => [optionsMenuRef, optionsMenuButtonRef],
    []
  );
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const terminal = isTerminal(state);
  const isLastTurn = !terminal && (state.finalTurnsRemaining ?? 0) > 0;
  const activePlayerId =
    state.players[state.activePlayerIndex]?.id ?? HUMAN_PLAYER;
  const humanView = useMemo(() => toPlayerView(state, HUMAN_PLAYER), [state]);
  const resolvedBotProfile = useMemo(
    () => resolveBotProfile(botProfileId),
    [botProfileId]
  );
  const score = useMemo(() => state.finalScore ?? scoreLive(state), [state]);
  const wonDistrictsByPlayer = useMemo(
    () => districtWinnersByPlayer(state),
    [state]
  );
  const isSecondShuffle = humanView.deck.reshuffles > 0;
  const showSecondShuffleLabel =
    isSecondShuffle && !(terminal && humanView.deck.drawCount === 0);
  const humanPlayer = humanView.players.find(
    (player) => player.id === HUMAN_PLAYER
  );
  const botPlayer = humanView.players.find(
    (player) => player.id === BOT_PLAYER
  );

  const humanActions = useMemo(() => {
    if (terminal || activePlayerId !== HUMAN_PLAYER) {
      return [] as readonly GameAction[];
    }
    return legalActions(state);
  }, [activePlayerId, state, terminal]);

  const humanActionItems = useMemo(
    () => buildHumanActionList(humanActions),
    [humanActions]
  );
  const tradeSourceGroups = useMemo(
    () => buildTradeSourceGroups(humanActions),
    [humanActions]
  );
  const hasMultipleTradeSources = tradeSourceGroups.length > 1;
  const firstTradeGroupIndex = useMemo(
    () => humanActionItems.findIndex((item) => item.kind === 'trade-group'),
    [humanActionItems]
  );
  const visibleHumanActionItems = useMemo(() => {
    if (!hasMultipleTradeSources) {
      return humanActionItems;
    }
    return humanActionItems.filter(
      (item, index) =>
        item.kind !== 'trade-group' || index === firstTradeGroupIndex
    );
  }, [firstTradeGroupIndex, hasMultipleTradeSources, humanActionItems]);

  const actionPickerOptions = useMemo(() => {
    if (!actionPicker || actionPicker.kind === 'trade-source') {
      return [];
    }
    return buildPickerOptions(
      toPickerQuery(actionPicker),
      humanActions,
      SUIT_TEXT_TOKEN
    );
  }, [actionPicker, humanActions]);

  const actionPickerTitle = useMemo((): string => {
    if (!actionPicker) {
      return '';
    }
    if (actionPicker.kind === 'trade-source') {
      return 'Trade x3 from';
    }
    return pickerTitle(toPickerQuery(actionPicker), SUIT_TEXT_TOKEN);
  }, [actionPicker]);

  const canResetTurn = useMemo(
    () => canUseTurnReset(state, activePlayerId, HUMAN_PLAYER, turnResetAnchor),
    [activePlayerId, state, turnResetAnchor]
  );

  useEffect(() => {
    if (
      !shouldCaptureTurnResetAnchor(
        state,
        activePlayerId,
        HUMAN_PLAYER,
        turnResetAnchor
      )
    ) {
      return;
    }

    setTurnResetAnchor({
      turn: state.turn,
      playerId: HUMAN_PLAYER,
      state,
    });
  }, [activePlayerId, state, turnResetAnchor]);

  useEffect(() => {
    if (terminal || activePlayerId !== BOT_PLAYER) {
      setBotThinking(false);
      return;
    }

    let cancelled = false;
    setBotThinking(true);
    const timerId = window.setTimeout(() => {
      void (async () => {
        const current = stateRef.current;
        const currentActive = current.players[current.activePlayerIndex]?.id;
        if (cancelled || isTerminal(current) || currentActive !== BOT_PLAYER) {
          setBotThinking(false);
          return;
        }

        const actions = legalActions(current);
        if (actions.length === 0) {
          setError('Bot has no legal actions.');
          setBotThinking(false);
          return;
        }

        try {
          const botView = toPlayerView(current, BOT_PLAYER);
          const choice = await resolvedBotProfile.policy.selectAction({
            state: current,
            view: botView,
            legalActions: actions,
            random: botRandomForState(current, resolvedBotProfile.selected.id),
          });

          if (cancelled) {
            return;
          }

          if (!choice) {
            setError('Bot policy could not select an action.');
            return;
          }

          const next = stepToDecision(current, choice);
          setState(next);
          setError(null);
        } catch (err) {
          setError(`Bot action failed: ${errorMessage(err)}`);
        } finally {
          setBotThinking(false);
        }
      })();
    }, BOT_DELAY_MS);

    return () => {
      cancelled = true;
      window.clearTimeout(timerId);
    };
  }, [activePlayerId, resolvedBotProfile, state, terminal]);

  useEffect(() => {
    if (terminal || activePlayerId !== HUMAN_PLAYER) {
      closeActionPicker();
      return;
    }

    if (actionPicker) {
      if (actionPicker.kind === 'trade-source') {
        if (!hasMultipleTradeSources) {
          closeActionPicker();
        }
        return;
      }
      const stillLegal = pickerStillLegal(
        toPickerQuery(actionPicker),
        humanActions
      );
      if (!stillLegal) {
        closeActionPicker();
      }
    }
  }, [
    activePlayerId,
    humanActions,
    terminal,
    actionPicker,
    closeActionPicker,
    hasMultipleTradeSources,
  ]);

  useDismissableLayer({
    enabled: Boolean(actionPicker),
    onDismiss: closeActionPicker,
    insideRefs: actionPopoverLayerRefs,
    closeOnScroll: true,
  });

  useDismissableLayer({
    enabled: optionsMenuOpen,
    onDismiss: closeOptionsMenu,
    insideRefs: optionsMenuLayerRefs,
  });

  const handleHumanAction = (action: GameAction) => {
    if (terminal || activePlayerId !== HUMAN_PLAYER) {
      return;
    }

    try {
      const next = stepToDecision(state, action);
      setState(next);
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const handleReset = () => {
    const seed = seedInput.trim() || makeSeed();
    setSeedInput(seed);
    closeActionPicker();
    closeOptionsMenu();
    setTurnResetAnchor(null);

    try {
      setState(createInitialState(seed));
      setError(null);
      setBotThinking(false);
    } catch (err) {
      setError(`Failed to start game: ${errorMessage(err)}`);
    }
  };

  const handlePickerSelection = (action: GameAction) => {
    closeActionPicker();
    handleHumanAction(action);
  };

  const handleTurnReset = () => {
    if (!turnResetAnchor) {
      return;
    }
    if (
      !canUseTurnReset(state, activePlayerId, HUMAN_PLAYER, turnResetAnchor)
    ) {
      return;
    }

    closeActionPicker();
    setState(turnResetAnchor.state);
    setError(null);
    setBotThinking(false);
  };

  const openTradePicker = (
    give: Suit,
    trigger: HTMLButtonElement,
    optionCount: number
  ) => {
    const position = pickerPosition(trigger, optionCount);
    setActionPicker({ kind: 'trade', give, ...position });
  };

  const openTradeSourcePicker = (
    trigger: HTMLButtonElement,
    optionCount: number
  ) => {
    const position = pickerPosition(trigger, optionCount);
    setActionPicker({ kind: 'trade-source', ...position });
  };

  const selectTradeSource = (give: Suit) => {
    setActionPicker((current) => {
      if (!current || current.kind !== 'trade-source') {
        return current;
      }

      return {
        kind: 'trade',
        give,
        top: current.top,
        left: current.left,
      };
    });
  };

  const openDistrictPicker = (
    config: {
      actionType: 'buy-deed' | 'develop-outright';
      cardId: CardId;
      payment?: Partial<Record<Suit, number>>;
      paymentKey?: string;
    },
    trigger: HTMLButtonElement,
    optionCount: number
  ) => {
    const position = pickerPosition(trigger, optionCount);
    setActionPicker({
      kind: 'district',
      actionType: config.actionType,
      cardId: config.cardId,
      payment: config.payment,
      paymentKey: config.paymentKey,
      ...position,
    });
  };

  const openDeedPaymentPicker = (
    config: { cardId: CardId; districtId: string },
    trigger: HTMLButtonElement,
    optionCount: number
  ) => {
    const position = pickerPosition(trigger, optionCount);
    setActionPicker({
      kind: 'deed-payment',
      cardId: config.cardId,
      districtId: config.districtId,
      ...position,
    });
  };

  const pickerPosition = (
    trigger: HTMLButtonElement,
    optionCount: number
  ): { top: number; left: number } => {
    const rect = trigger.getBoundingClientRect();
    const rowCount = Math.max(1, Math.ceil(optionCount / 2));
    const estimatedHeight = Math.max(
      TRADE_POPOVER_MIN_HEIGHT_PX,
      116 + rowCount * 46
    );
    const maxLeft =
      window.innerWidth - TRADE_POPOVER_WIDTH_PX - VIEWPORT_PADDING_PX;
    const maxTop = window.innerHeight - estimatedHeight - VIEWPORT_PADDING_PX;

    const left = clamp(
      rect.right + TRADE_POPOVER_GAP_PX,
      VIEWPORT_PADDING_PX,
      maxLeft
    );
    const top = clamp(rect.top, VIEWPORT_PADDING_PX, maxTop);

    return { left, top };
  };

  if (!humanPlayer || !botPlayer) {
    return (
      <div className="app-shell">
        <section className="panel">
          <h1>Magnate</h1>
          <p>Could not load player data.</p>
        </section>
      </div>
    );
  }

  const recentLog = [...humanView.log].reverse();
  const deckStackCount = Math.min(3, humanView.deck.drawCount);
  const deckOverlayShiftClass =
    deckStackCount >= 3
      ? 'overlay-shift-2'
      : deckStackCount === 2
        ? 'overlay-shift-1'
        : 'overlay-shift-0';
  const discardStackCardIds = humanView.deck.discard.slice(0, 3).reverse();
  const discardCardDetails = humanView.deck.discard.map((cardId) => {
    const card = CARD_BY_ID[cardId];
    const rank =
      card.kind === 'Property' || card.kind === 'Crown'
        ? String(card.rank)
        : card.kind;
    const suitTokenText =
      card.kind === 'Excuse'
        ? ''
        : card.suits.map((suit) => SUIT_TEXT_TOKEN[suit]).join(' ');
    return {
      id: card.id,
      name: card.name,
      rank,
      suitTokenText,
    };
  });

  return (
    <div className="app-shell">
      {error && (
        <section className="error-banner">
          <strong>Engine Error:</strong> {error}
        </section>
      )}

      <main className="layout">
        <aside className="actions-pane">
          <section className="panel brand-panel">
            <div className="brand-header">
              <div className="brand-title-block">
                <h1>Magnate</h1>
                <p className="brand-subtitle">
                  For the throne of the Grand Duke
                </p>
              </div>
              <button
                ref={optionsMenuButtonRef}
                type="button"
                className={`hamburger-button${optionsMenuOpen ? ' is-open' : ''}`}
                aria-label="Game options"
                aria-controls="brand-options-menu"
                aria-expanded={optionsMenuOpen}
                onClick={() => setOptionsMenuOpen((open) => !open)}
              >
                <span />
                <span />
                <span />
              </button>
            </div>

            {optionsMenuOpen ? (
              <section
                id="brand-options-menu"
                ref={optionsMenuRef}
                className="brand-options-menu"
                aria-label="Game options"
              >
                <div className="brand-controls">
                  <input
                    id="seed-input"
                    aria-label="Seed"
                    className="seed-input"
                    value={seedInput}
                    onChange={(event) => setSeedInput(event.target.value)}
                  />
                  <button
                    className="reset-button"
                    type="button"
                    onClick={handleReset}
                  >
                    New Game
                  </button>
                </div>
                <div className="bot-profile-controls">
                  <label htmlFor="bot-profile-select">Bot Profile</label>
                  <select
                    id="bot-profile-select"
                    className="bot-profile-select"
                    value={botProfileId}
                    onChange={(event) =>
                      setBotProfileId(event.target.value as BotProfileId)
                    }
                  >
                    {BOT_PROFILES.map((profile) => (
                      <option key={profile.id} value={profile.id}>
                        {profile.label}
                      </option>
                    ))}
                  </select>
                  <p className="bot-profile-note">
                    {resolvedBotProfile.statusText}
                  </p>
                </div>
              </section>
            ) : null}
          </section>

          <section className="panel actions-panel">
            <div className="actions-heading">
              <h2>{terminal ? 'Game Over' : 'Actions'}</h2>
              {isLastTurn && <span className="last-turn-badge">Last Turn</span>}
            </div>
            <div className="actions-body">
              {terminal ? (
                <section
                  className="terminal-score-summary"
                  aria-label="Final score breakdown"
                >
                  <p className="score-result terminal-score-winner">
                    Winner: <strong>{winnerLabel(score.winner)}</strong>
                  </p>
                  <p className="score-line terminal-score-decider">
                    <span>Decided By</span>
                    <strong>{deciderLabel(score.decidedBy)}</strong>
                  </p>

                  <div className="terminal-score-players">
                    {([HUMAN_PLAYER, BOT_PLAYER] as const).map((playerId) => (
                      <article
                        key={`terminal-score-${playerId}`}
                        className="terminal-score-player"
                      >
                        <h3>{playerId === HUMAN_PLAYER ? 'You' : 'Bot'}</h3>
                        <p className="score-line">
                          <span>Districts Won</span>
                          <strong>
                            {formatDistrictList(wonDistrictsByPlayer[playerId])}
                          </strong>
                        </p>
                        <p className="score-line">
                          <span>Total Properties</span>
                          <strong>{score.rankTotals[playerId]}</strong>
                        </p>
                        <p className="score-line">
                          <span>Resources</span>
                          <strong>{score.resourceTotals[playerId]}</strong>
                        </p>
                      </article>
                    ))}
                  </div>
                </section>
              ) : activePlayerId === HUMAN_PLAYER ? (
                <div className="actions-human-layout">
                  <div className="actions-human-main">
                    {visibleHumanActionItems.length === 0 ? (
                      <p className="empty-note">No legal actions.</p>
                    ) : (
                      <div className="action-list">
                        {visibleHumanActionItems.map((item, index) => {
                          const categoryKey = actionCategoryForItem(item);
                          const previousCategoryKey =
                            index > 0
                              ? actionCategoryForItem(
                                  visibleHumanActionItems[index - 1]
                                )
                              : null;
                          const showCategory =
                            previousCategoryKey !== categoryKey;
                          const categoryLabel =
                            actionCategoryLabel(categoryKey);
                          const renderCategorizedAction = (
                            key: string,
                            button: ReactNode
                          ) => (
                            <div
                              key={key}
                              className={`action-entry${showCategory ? ' has-category' : ''}`}
                            >
                              {showCategory ? (
                                <p className="action-category">
                                  {categoryLabel}
                                </p>
                              ) : null}
                              {button}
                            </div>
                          );

                          if (item.kind === 'trade-group') {
                            if (hasMultipleTradeSources) {
                              return renderCategorizedAction(
                                'trade-source-group',
                                <button
                                  type="button"
                                  className="action-button has-submenu"
                                  onClick={(event) => {
                                    const trigger = event.currentTarget;
                                    if (actionPicker?.kind === 'trade-source') {
                                      closeActionPicker();
                                      return;
                                    }
                                    openTradeSourcePicker(
                                      trigger,
                                      tradeSourceGroups.length
                                    );
                                  }}
                                >
                                  <span className="action-text">
                                    Trade resources
                                  </span>
                                </button>
                              );
                            }

                            if (item.options.length === 1) {
                              const [onlyOption] = item.options;
                              return renderCategorizedAction(
                                `trade-direct-${item.give}`,
                                <button
                                  type="button"
                                  className="action-button"
                                  onClick={() => handleHumanAction(onlyOption)}
                                >
                                  <span className="action-text">
                                    {renderSuitText(
                                      describeAction(
                                        onlyOption,
                                        SUIT_TEXT_TOKEN
                                      )
                                    )}
                                  </span>
                                </button>
                              );
                            }

                            return renderCategorizedAction(
                              `trade-group-${item.give}`,
                              <button
                                type="button"
                                className="action-button has-submenu"
                                onClick={(event) => {
                                  const trigger = event.currentTarget;
                                  if (
                                    actionPicker?.kind === 'trade' &&
                                    actionPicker.give === item.give
                                  ) {
                                    closeActionPicker();
                                    return;
                                  }
                                  openTradePicker(
                                    item.give,
                                    trigger,
                                    item.options.length
                                  );
                                }}
                              >
                                <span className="action-text">
                                  {renderSuitText(
                                    `Trade ${SUIT_TEXT_TOKEN[item.give]}x3`
                                  )}
                                </span>
                              </button>
                            );
                          }

                          if (item.kind === 'buy-deed-group') {
                            if (item.options.length === 1) {
                              const [onlyOption] = item.options;
                              return renderCategorizedAction(
                                `buy-deed-direct-${actionStableKey(onlyOption)}`,
                                <button
                                  type="button"
                                  className="action-button"
                                  onClick={() => handleHumanAction(onlyOption)}
                                >
                                  <span className="action-text">
                                    {renderSuitText(
                                      describeAction(
                                        onlyOption,
                                        SUIT_TEXT_TOKEN
                                      )
                                    )}
                                  </span>
                                </button>
                              );
                            }

                            return renderCategorizedAction(
                              `buy-deed-group-${item.cardId}`,
                              <button
                                type="button"
                                className="action-button has-submenu"
                                onClick={(event) => {
                                  const trigger = event.currentTarget;
                                  if (
                                    actionPicker?.kind === 'district' &&
                                    actionPicker.actionType === 'buy-deed' &&
                                    actionPicker.cardId === item.cardId
                                  ) {
                                    closeActionPicker();
                                    return;
                                  }
                                  openDistrictPicker(
                                    {
                                      actionType: 'buy-deed',
                                      cardId: item.cardId,
                                    },
                                    trigger,
                                    item.options.length
                                  );
                                }}
                              >
                                <span className="action-text">
                                  {renderSuitText(
                                    `Buy deed ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)}`
                                  )}
                                </span>
                              </button>
                            );
                          }

                          if (item.kind === 'develop-deed-group') {
                            if (item.options.length === 1) {
                              const [onlyOption] = item.options;
                              return renderCategorizedAction(
                                `develop-deed-direct-${actionStableKey(onlyOption)}`,
                                <button
                                  type="button"
                                  className="action-button"
                                  onClick={() => handleHumanAction(onlyOption)}
                                >
                                  <span className="action-text">
                                    {renderSuitText(
                                      describeAction(
                                        onlyOption,
                                        SUIT_TEXT_TOKEN
                                      )
                                    )}
                                  </span>
                                </button>
                              );
                            }

                            return renderCategorizedAction(
                              `develop-deed-group-${item.cardId}-${item.districtId}`,
                              <button
                                type="button"
                                className="action-button has-submenu"
                                onClick={(event) => {
                                  const trigger = event.currentTarget;
                                  if (
                                    actionPicker?.kind === 'deed-payment' &&
                                    actionPicker.cardId === item.cardId &&
                                    actionPicker.districtId === item.districtId
                                  ) {
                                    closeActionPicker();
                                    return;
                                  }
                                  openDeedPaymentPicker(
                                    {
                                      cardId: item.cardId,
                                      districtId: item.districtId,
                                    },
                                    trigger,
                                    item.options.length
                                  );
                                }}
                              >
                                <span className="action-text">
                                  {renderSuitText(
                                    `Develop deed ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)} in ${item.districtId}`
                                  )}
                                </span>
                              </button>
                            );
                          }

                          if (item.kind === 'develop-outright-group') {
                            if (item.options.length === 1) {
                              const [onlyOption] = item.options;
                              return renderCategorizedAction(
                                `develop-outright-direct-${actionStableKey(onlyOption)}`,
                                <button
                                  type="button"
                                  className="action-button"
                                  onClick={() => handleHumanAction(onlyOption)}
                                >
                                  <span className="action-text">
                                    {renderSuitText(
                                      describeAction(
                                        onlyOption,
                                        SUIT_TEXT_TOKEN
                                      )
                                    )}
                                  </span>
                                </button>
                              );
                            }

                            return renderCategorizedAction(
                              `develop-outright-group-${item.cardId}-${item.paymentKey}`,
                              <button
                                type="button"
                                className="action-button has-submenu"
                                onClick={(event) => {
                                  const trigger = event.currentTarget;
                                  if (
                                    actionPicker?.kind === 'district' &&
                                    actionPicker.actionType ===
                                      'develop-outright' &&
                                    actionPicker.cardId === item.cardId &&
                                    actionPicker.paymentKey === item.paymentKey
                                  ) {
                                    closeActionPicker();
                                    return;
                                  }
                                  openDistrictPicker(
                                    {
                                      actionType: 'develop-outright',
                                      cardId: item.cardId,
                                      payment: item.payment,
                                      paymentKey: item.paymentKey,
                                    },
                                    trigger,
                                    item.options.length
                                  );
                                }}
                              >
                                <span className="action-text">
                                  {renderSuitText(
                                    `Develop ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)} (${formatTokens(item.payment, SUIT_TEXT_TOKEN)})`
                                  )}
                                </span>
                              </button>
                            );
                          }

                          return renderCategorizedAction(
                            actionStableKey(item.action),
                            <button
                              type="button"
                              className="action-button"
                              onClick={() => handleHumanAction(item.action)}
                            >
                              <span className="action-text">
                                {renderSuitText(
                                  describeAction(item.action, SUIT_TEXT_TOKEN)
                                )}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {canResetTurn ? (
                    <div className="actions-footer">
                      <button
                        key="reset-turn"
                        type="button"
                        className="action-button reset-turn-button"
                        onClick={handleTurnReset}
                      >
                        <span className="action-text">Reset turn</span>
                      </button>
                    </div>
                  ) : null}
                </div>
              ) : (
                <p className="empty-note">
                  {botThinking ? 'Bot is thinking...' : 'Waiting for bot...'}
                </p>
              )}
            </div>
          </section>

          <PlayerPanel
            title="You"
            player={humanPlayer}
            isActive={!terminal && humanView.activePlayerId === HUMAN_PLAYER}
            score={score}
            terminal={terminal}
            handSlotCount={PLAYER_HAND_SLOT_COUNT}
            botPlayerId={BOT_PLAYER}
          />
        </aside>

        <section className="board-pane">
          <PlayerTokenRail player={botPlayer} side="bot" />
          <div className="district-strip" aria-label="District board">
            {humanView.districts.map((district) => (
              <DistrictColumn
                key={district.id}
                district={district}
                humanPlayerId={HUMAN_PLAYER}
                botPlayerId={BOT_PLAYER}
              />
            ))}
          </div>
          <PlayerTokenRail player={humanPlayer} side="human" />
        </section>

        <aside className="info-pane">
          <PlayerPanel
            title="Bot"
            player={botPlayer}
            isActive={!terminal && humanView.activePlayerId === BOT_PLAYER}
            score={score}
            terminal={terminal}
            handSlotCount={PLAYER_HAND_SLOT_COUNT}
            botPlayerId={BOT_PLAYER}
          />

          <section className="panel">
            <h2>Roll Result</h2>
            <RollResult
              roll={humanView.lastIncomeRoll}
              taxSuit={humanView.lastTaxSuit}
            />
          </section>

          <section className="panel">
            <h2>Deck State</h2>
            <div className="deck-piles" aria-label="Deck and discard piles">
              <div className="deck-pile">
                <div
                  className={`deck-pile-stack is-deck ${deckOverlayShiftClass}`}
                  title="Cards remaining"
                  aria-label="Cards remaining"
                >
                  {deckStackCount === 0 ? (
                    <div className="deck-pile-card deck-pile-card-empty deck-pile-stack-card" />
                  ) : (
                    Array.from({ length: deckStackCount }).map((_, index) => (
                      <div
                        key={`deck-back-${index}`}
                        className="deck-pile-card deck-pile-card-back deck-pile-stack-card"
                      />
                    ))
                  )}
                  {showSecondShuffleLabel ? (
                    <span
                      className="deck-pile-overlay-label"
                      aria-hidden="true"
                    >
                      2nd shuffle
                    </span>
                  ) : null}
                </div>
                <strong className="deck-pile-count">
                  {humanView.deck.drawCount}
                </strong>
              </div>
              <div className="deck-pile">
                <div className="player-score-wrap discard-pile-wrap">
                  <div
                    className={`deck-pile-stack is-discard${discardStackCardIds.length > 1 ? ' is-fanned' : ''}`}
                    title="Discard pile"
                    aria-label="Discard pile"
                    tabIndex={0}
                  >
                    {discardStackCardIds.length > 0 ? (
                      discardStackCardIds.map((cardId, index) => (
                        <div
                          key={`discard-${cardId}-${index}`}
                          className="deck-pile-card deck-pile-card-discard deck-pile-stack-card"
                        >
                          <img
                            className="deck-pile-image"
                            src={getCardImage(cardId)}
                            alt=""
                          />
                        </div>
                      ))
                    ) : (
                      <div className="deck-pile-card deck-pile-card-empty deck-pile-stack-card" />
                    )}
                  </div>
                  <section
                    className="player-score-popover discard-pile-popover"
                    role="tooltip"
                    aria-label="Discard pile details"
                  >
                    <p className="score-result">
                      Discarded Cards:{' '}
                      <strong>{discardCardDetails.length}</strong>
                    </p>
                    {discardCardDetails.length === 0 ? (
                      <p className="score-line">
                        <span>None yet</span>
                        <strong>-</strong>
                      </p>
                    ) : (
                      <ol className="discard-pile-list">
                        {discardCardDetails.map((card, index) => (
                          <li
                            key={`discard-detail-${card.id}-${index}`}
                            className="discard-pile-item"
                          >
                            <p className="discard-pile-card-row">
                              <strong className="discard-pile-card-rank">
                                {card.rank}
                              </strong>
                              <span className="discard-pile-card-suits">
                                {card.suitTokenText.length > 0 ? (
                                  renderSuitText(card.suitTokenText)
                                ) : (
                                  <strong>-</strong>
                                )}
                              </span>
                              <span className="discard-pile-card-name">
                                {card.name}
                              </span>
                            </p>
                          </li>
                        ))}
                      </ol>
                    )}
                  </section>
                </div>
                <strong className="deck-pile-count">
                  {humanView.deck.discard.length}
                </strong>
              </div>
            </div>
          </section>

          <section className="panel log-panel">
            <h2>Log</h2>
            {recentLog.length === 0 ? (
              <p className="empty-note">No actions yet.</p>
            ) : (
              <ol className="log-list">
                {recentLog.map((entry, index) => (
                  <li
                    key={`${entry.turn}-${entry.phase}-${entry.summary}-${index}`}
                  >
                    <span className="log-turn">T{entry.turn}</span>
                    <span>{entry.player}</span>
                    <span>{entry.summary}</span>
                  </li>
                ))}
              </ol>
            )}
          </section>
        </aside>
      </main>

      {actionPicker && (
        <section
          ref={actionPopoverRef}
          className="panel trade-popover"
          role="dialog"
          aria-label="Choose follow-up action option"
          style={{
            top: `${actionPicker.top}px`,
            left: `${actionPicker.left}px`,
          }}
        >
          <h2>{renderSuitText(actionPickerTitle)}</h2>

          {actionPicker.kind === 'trade-source' ? (
            tradeSourceGroups.length === 0 ? (
              <p className="empty-note">No options available.</p>
            ) : (
              <div className="trade-choice-list">
                {tradeSourceGroups.map((group) => (
                  <button
                    key={`trade-source-${group.give}`}
                    type="button"
                    className="trade-choice-button"
                    onClick={() => selectTradeSource(group.give)}
                  >
                    {renderSuitText(`${SUIT_TEXT_TOKEN[group.give]} x3`)}
                  </button>
                ))}
              </div>
            )
          ) : actionPickerOptions.length === 0 ? (
            <p className="empty-note">No options available.</p>
          ) : (
            <div className="trade-choice-list">
              {actionPickerOptions.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  className="trade-choice-button"
                  onClick={() => handlePickerSelection(option.action)}
                >
                  {renderSuitText(option.label)}
                </button>
              ))}
            </div>
          )}

          <button
            type="button"
            className="trade-cancel-button"
            onClick={closeActionPicker}
          >
            Cancel
          </button>
        </section>
      )}
    </div>
  );
}

function actionCategoryForItem(item: HumanActionListItem): string {
  switch (item.kind) {
    case 'trade-group':
      return 'trade';
    case 'buy-deed-group':
      return 'buy-deed';
    case 'develop-deed-group':
      return 'develop-deed';
    case 'develop-outright-group':
      return 'develop-outright';
    case 'action':
      return item.action.type;
  }
}

function actionCategoryLabel(category: string): string {
  switch (category) {
    case 'trade':
      return 'Trade';
    case 'buy-deed':
      return 'Buy Deed';
    case 'develop-deed':
      return 'Develop Deed';
    case 'develop-outright':
      return 'Develop Outright';
    case 'sell-card':
      return 'Sell Card';
    case 'choose-income-suit':
      return 'Choose Income';
    case 'end-turn':
      return 'End Turn';
    default:
      return category;
  }
}

function winnerLabel(winner: FinalScore['winner']): string {
  if (winner === 'Draw') {
    return 'Tie';
  }
  return winner === HUMAN_PLAYER ? 'You' : 'Bot';
}

function deciderLabel(decidedBy: FinalScore['decidedBy']): string {
  switch (decidedBy) {
    case 'districts':
      return 'Districts';
    case 'rank-total':
      return 'Total Properties';
    case 'resources':
      return 'Resources';
    case 'draw':
      return 'Tie';
  }
}

function formatDistrictList(districtIds: readonly string[]): string {
  if (districtIds.length === 0) {
    return 'None';
  }
  return districtIds.join(', ');
}

function clamp(value: number, min: number, max: number): number {
  if (max < min) {
    return min;
  }
  return Math.max(min, Math.min(value, max));
}

function renderSuitText(text: string): ReactNode {
  if (!text) {
    return text;
  }

  const nodes: ReactNode[] = [];
  let cursor = 0;

  for (const match of text.matchAll(SUIT_TOKEN_REGEX)) {
    const index = match.index ?? 0;
    const token = match[0];
    const suit = SUIT_TOKEN_TO_SUIT[token];

    if (index > cursor) {
      nodes.push(text.slice(cursor, index));
    }

    if (suit) {
      nodes.push(
        <TokenChip
          key={`suit-${index}-${suit}`}
          suit={suit}
          count={1}
          compact
          className="inline-token-chip"
        />
      );
    } else {
      nodes.push(token);
    }

    cursor = index + token.length;
  }

  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }

  return nodes.length > 0 ? <>{nodes}</> : text;
}

function toPickerQuery(
  picker: Exclude<ActionPickerState, { kind: 'trade-source' }>
): ActionPickerQuery {
  if (picker.kind === 'trade') {
    return { kind: 'trade', give: picker.give };
  }
  if (picker.kind === 'deed-payment') {
    return {
      kind: 'deed-payment',
      cardId: picker.cardId,
      districtId: picker.districtId,
    };
  }
  return {
    kind: 'district',
    actionType: picker.actionType,
    cardId: picker.cardId,
    payment: picker.payment,
    paymentKey: picker.paymentKey,
  };
}
