import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';

import { legalActions } from './engine/actionBuilders';
import cubeDieIcon from './assets/icons/cube.png';
import dodecahedronDieIcon from './assets/icons/dodecahedron.png';
import { CARD_BY_ID, PAWN_CARDS, type CardId } from './engine/cards';
import { rngFromSeed } from './engine/rng';
import { createSession, stepToDecision } from './engine/session';
import { isTerminal, scoreLive } from './engine/scoring';
import { developmentCost, findProperty, SUITS } from './engine/stateHelpers';
import type { BotProfileId } from './policies/catalog';
import {
  BOT_PROFILES,
  DEFAULT_BOT_PROFILE_ID,
  resolveBotProfile,
} from './policies/catalog';
import type {
  DistrictStack,
  DistrictState,
  FinalScore,
  GameAction,
  GameState,
  ObservedPlayerState,
  PlayerId,
  ResourcePool,
  Suit,
} from './engine/types';
import { toPlayerView } from './engine/view';
import {
  actionStableKey,
  buildHumanActionList,
  buildPickerOptions,
  cardSummary,
  describeAction,
  formatTokens,
  pickerStillLegal,
  pickerTitle,
  type ActionPickerQuery,
} from './ui/actionPresentation';
import {
  canUseTurnReset,
  shouldCaptureTurnResetAnchor,
  type TurnResetAnchor,
} from './ui/turnReset';
import { getCardImage } from './ui/cardImages';
import {
  SuitIcon,
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

type ActionPickerState = ActionPickerQuery & {
  top: number;
  left: number;
};

type CardPerspective = 'human' | 'bot';

const SUIT_TOKEN_BG: Record<Suit, string> = {
  Moons: '#e4e7eb',
  Suns: '#f7cc95',
  Waves: '#cfe3f5',
  Leaves: '#dfc8b2',
  Wyrms: '#bfe3b3',
  Knots: '#f6f4bf',
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
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const terminal = isTerminal(state);
  const isLastTurn = !terminal && (state.finalTurnsRemaining ?? 0) > 0;
  const activePlayer = state.players[state.activePlayerIndex];
  if (!activePlayer) {
    throw new Error(
      `Active player index is out of bounds: ${state.activePlayerIndex}`
    );
  }
  const activePlayerId = activePlayer.id;
  const humanView = useMemo(() => toPlayerView(state, HUMAN_PLAYER), [state]);
  const resolvedBotProfile = useMemo(
    () => resolveBotProfile(botProfileId),
    [botProfileId]
  );
  const score = useMemo(() => state.finalScore ?? scoreLive(state), [state]);
  const reshufflesRemaining = Math.max(0, 2 - humanView.deck.reshuffles);

  const playersById = useMemo(
    () => new Map(humanView.players.map((player) => [player.id, player])),
    [humanView.players]
  );
  const humanPlayer = playersById.get(HUMAN_PLAYER);
  const botPlayer = playersById.get(BOT_PLAYER);

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

  const actionPickerOptions = useMemo(() => {
    if (!actionPicker) {
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
      setActionPicker(null);
      return;
    }

    if (actionPicker) {
      const stillLegal = pickerStillLegal(
        toPickerQuery(actionPicker),
        humanActions
      );
      if (!stillLegal) {
        setActionPicker(null);
      }
    }
  }, [activePlayerId, humanActions, terminal, actionPicker]);

  useEffect(() => {
    if (!actionPicker) {
      return;
    }

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) {
        return;
      }

      if (actionPopoverRef.current?.contains(target)) {
        return;
      }

      setActionPicker(null);
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setActionPicker(null);
      }
    };

    const handleScroll = () => {
      setActionPicker(null);
    };

    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('keydown', handleEscape);
    window.addEventListener('scroll', handleScroll, true);

    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('keydown', handleEscape);
      window.removeEventListener('scroll', handleScroll, true);
    };
  }, [actionPicker]);

  useEffect(() => {
    if (!optionsMenuOpen) {
      return;
    }

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) {
        return;
      }

      if (optionsMenuRef.current?.contains(target)) {
        return;
      }

      if (optionsMenuButtonRef.current?.contains(target)) {
        return;
      }

      setOptionsMenuOpen(false);
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOptionsMenuOpen(false);
      }
    };

    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('keydown', handleEscape);

    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('keydown', handleEscape);
    };
  }, [optionsMenuOpen]);

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
    setActionPicker(null);
    setOptionsMenuOpen(false);
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
    setActionPicker(null);
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

    setActionPicker(null);
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
  const discardStackCardIds = humanView.deck.discard.slice(0, 3).reverse();

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
              <h2>Actions</h2>
              {isLastTurn && <span className="last-turn-badge">Last Turn</span>}
            </div>
            <div className="actions-body">
              {terminal ? (
                <p className="empty-note">Game over.</p>
              ) : activePlayerId === HUMAN_PLAYER ? (
                <div className="actions-human-layout">
                  <div className="actions-human-main">
                    {humanActionItems.length === 0 ? (
                      <p className="empty-note">No legal actions.</p>
                    ) : (
                      <div className="action-list">
                        {humanActionItems.map((item) => {
                          if (item.kind === 'trade-group') {
                            if (item.options.length === 1) {
                              const [onlyOption] = item.options;
                              return (
                                <button
                                  key={`trade-direct-${item.give}`}
                                  type="button"
                                  className="action-button"
                                  onClick={() => handleHumanAction(onlyOption)}
                                >
                                  <span className="action-kind">trade</span>
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

                            return (
                              <button
                                key={`trade-group-${item.give}`}
                                type="button"
                                className="action-button has-submenu"
                                onClick={(event) => {
                                  const trigger = event.currentTarget;
                                  if (
                                    actionPicker?.kind === 'trade' &&
                                    actionPicker.give === item.give
                                  ) {
                                    setActionPicker(null);
                                    return;
                                  }
                                  openTradePicker(
                                    item.give,
                                    trigger,
                                    item.options.length
                                  );
                                }}
                              >
                                <span className="action-kind">trade</span>
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
                              return (
                                <button
                                  key={`buy-deed-direct-${actionStableKey(onlyOption)}`}
                                  type="button"
                                  className="action-button"
                                  onClick={() => handleHumanAction(onlyOption)}
                                >
                                  <span className="action-kind">
                                    {onlyOption.type}
                                  </span>
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

                            return (
                              <button
                                key={`buy-deed-group-${item.cardId}`}
                                type="button"
                                className="action-button has-submenu"
                                onClick={(event) => {
                                  const trigger = event.currentTarget;
                                  if (
                                    actionPicker?.kind === 'district' &&
                                    actionPicker.actionType === 'buy-deed' &&
                                    actionPicker.cardId === item.cardId
                                  ) {
                                    setActionPicker(null);
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
                                <span className="action-kind">buy-deed</span>
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
                              return (
                                <button
                                  key={`develop-deed-direct-${actionStableKey(onlyOption)}`}
                                  type="button"
                                  className="action-button"
                                  onClick={() => handleHumanAction(onlyOption)}
                                >
                                  <span className="action-kind">
                                    {onlyOption.type}
                                  </span>
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

                            return (
                              <button
                                key={`develop-deed-group-${item.cardId}-${item.districtId}`}
                                type="button"
                                className="action-button has-submenu"
                                onClick={(event) => {
                                  const trigger = event.currentTarget;
                                  if (
                                    actionPicker?.kind === 'deed-payment' &&
                                    actionPicker.cardId === item.cardId &&
                                    actionPicker.districtId === item.districtId
                                  ) {
                                    setActionPicker(null);
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
                                <span className="action-kind">
                                  develop-deed
                                </span>
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
                              return (
                                <button
                                  key={`develop-outright-direct-${actionStableKey(onlyOption)}`}
                                  type="button"
                                  className="action-button"
                                  onClick={() => handleHumanAction(onlyOption)}
                                >
                                  <span className="action-kind">
                                    {onlyOption.type}
                                  </span>
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

                            return (
                              <button
                                key={`develop-outright-group-${item.cardId}-${item.paymentKey}`}
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
                                    setActionPicker(null);
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
                                <span className="action-kind">
                                  develop-outright
                                </span>
                                <span className="action-text">
                                  {renderSuitText(
                                    `Develop ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)} (${formatTokens(item.payment, SUIT_TEXT_TOKEN)})`
                                  )}
                                </span>
                              </button>
                            );
                          }

                          return (
                            <button
                              key={actionStableKey(item.action)}
                              type="button"
                              className="action-button"
                              onClick={() => handleHumanAction(item.action)}
                            >
                              <span className="action-kind">
                                {item.action.type}
                              </span>
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
                        <span className="action-kind">reset-turn</span>
                        <span className="action-text">reset turn</span>
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
            isActive={humanView.activePlayerId === HUMAN_PLAYER}
            score={score}
            terminal={terminal}
          />
        </aside>

        <section className="board-pane">
          <PlayerTokenRail player={botPlayer} side="bot" />
          <div className="district-strip" aria-label="District board">
            {humanView.districts.map((district) => (
              <DistrictColumn key={district.id} district={district} />
            ))}
          </div>
          <PlayerTokenRail player={humanPlayer} side="human" />
        </section>

        <aside className="info-pane">
          <PlayerPanel
            title="Bot"
            player={botPlayer}
            isActive={humanView.activePlayerId === BOT_PLAYER}
            score={score}
            terminal={terminal}
          />

        <aside className="info-pane">
          <section className="panel brand-panel">
            <h1>Magnate</h1>
            <p className="brand-subtitle">For the throne of the Grand Duke</p>
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
                  <option
                    key={profile.id}
                    value={profile.id}
                    disabled={!profile.available}
                  >
                    {profile.label}
                  </option>
                ))}
              </select>
              <p className="bot-profile-note">
                {resolvedBotProfile.statusText}
              </p>
            </div>
          </section>

          <section className="panel">
            <h2>Roll Result</h2>
            <RollResult roll={humanView.lastIncomeRoll} taxSuit={humanView.lastTaxSuit} />
          </section>

          <section className="panel">
            <h2>Deck State</h2>
            <p className="meta-line deck-reshuffles-line">
              <span>Reshuffles Remaining</span>
              <strong>{reshufflesRemaining}</strong>
            </p>
            <div className="deck-piles" aria-label="Deck and discard piles">
              <div className="deck-pile">
                <div className="deck-pile-stack is-deck" title="Cards remaining" aria-label="Cards remaining">
                  {deckStackCount === 0 ? (
                    <div className="deck-pile-card deck-pile-card-empty deck-pile-stack-card" />
                  ) : (
                    Array.from({ length: deckStackCount }).map((_, index) => (
                      <div key={`deck-back-${index}`} className="deck-pile-card deck-pile-card-back deck-pile-stack-card" />
                    ))
                  )}
                </div>
                <strong className="deck-pile-count">{humanView.deck.drawCount}</strong>
              </div>
              <div className="deck-pile">
                <div
                  className={`deck-pile-stack is-discard${discardStackCardIds.length > 1 ? ' is-fanned' : ''}`}
                  title="Discard pile"
                  aria-label="Discard pile"
                >
                  {discardStackCardIds.length > 0 ? (
                    discardStackCardIds.map((cardId, index) => (
                      <div
                        key={`discard-${cardId}-${index}`}
                        className="deck-pile-card deck-pile-card-discard deck-pile-stack-card"
                      >
                        <img className="deck-pile-image" src={getCardImage(cardId)} alt="" />
                      </div>
                    ))
                  ) : (
                    <div className="deck-pile-card deck-pile-card-empty deck-pile-stack-card" />
                  )}
                </div>
                <strong className="deck-pile-count">{humanView.deck.discard.length}</strong>
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

          {actionPickerOptions.length === 0 ? (
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
            onClick={() => setActionPicker(null)}
          >
            Cancel
          </button>
        </section>
      )}
    </div>
  );
}

function PlayerPanel({
  title,
  player,
  isActive,
  score,
  terminal,
}: {
  title: string;
  player: ObservedPlayerState;
  isActive: boolean;
  score: FinalScore;
  terminal: boolean;
}) {
  const handCardCount = player.handHidden
    ? player.handCount
    : player.hand.length;
  const handSlots = Math.max(PLAYER_HAND_SLOT_COUNT, handCardCount);
  const cardPerspective: CardPerspective =
    player.id === BOT_PLAYER ? 'bot' : 'human';
  const crownTokens = crownsToTokens(player.crowns);
  const districtScore = score.districtPoints[player.id];
  const scoreHeadline = terminal ? 'Winner' : 'Leader';

  return (
    <section className={`player-panel${isActive ? ' is-active' : ''}`}>
      <header className="player-header">
        <div className="player-title-line">
          <h2>{title}</h2>
          <div className="player-score-wrap">
            <span className="player-score-badge" tabIndex={0}>
              {districtScore} VP
            </span>
            <section
              className="player-score-popover"
              role="tooltip"
              aria-label="Score details"
            >
              <p className="score-result">
                {scoreHeadline}: <strong>{score.winner}</strong> (
                {score.decidedBy})
              </p>
              <ScoreLine
                label="Districts"
                a={score.districtPoints.PlayerA}
                b={score.districtPoints.PlayerB}
              />
              <ScoreLine
                label="Rank Total"
                a={score.rankTotals.PlayerA}
                b={score.rankTotals.PlayerB}
              />
              <ScoreLine
                label="Resources"
                a={score.resourceTotals.PlayerA}
                b={score.resourceTotals.PlayerB}
              />
            </section>
          </div>
        </div>
        <span className="player-meta">{player.id}</span>
      </header>

      <div className="player-row">
        <div className="player-section resources-section">
          <h3>Resources</h3>
          <TokenRow tokens={player.resources} fixedSuitSlots />
        </div>

        <div className="player-section crowns-section">
          <h3>Crowns</h3>
          <TokenRow
            tokens={crownTokens}
            compact
            className="crowns-token-row"
            emptyLabel="None"
          />
        </div>

        <div className="player-section hand-section">
          <h3>Hand</h3>
          <div className="card-row-wrap fixed-slots">
            {Array.from({ length: handSlots }).map((_, index) => {
              if (player.handHidden) {
                return index < player.handCount ? (
                  <CardTile key={`hidden-${player.id}-${index}`} hidden />
                ) : (
                  <CardTile
                    key={`hidden-slot-${player.id}-${index}`}
                    placeholder
                  />
                );
              }

              const cardId = player.hand[index];
              if (!cardId) {
                return (
                  <CardTile
                    key={`hand-slot-${player.id}-${index}`}
                    placeholder
                  />
                );
              }

              return <CardTile key={`${cardId}-${index}`} cardId={cardId} />;
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

function PlayerTokenRail({
  player,
  side,
}: {
  player: ObservedPlayerState;
  side: 'bot' | 'human';
}) {
  const crownSuits = crownsToSuits(player.crowns);

  const crowns = (
    <div className="token-rail-group">
      <h3>Crowns</h3>
      {crownSuits.length > 0 ? (
        <div className="token-row compact crowns-rail-row">
          {crownSuits.map((suit, index) => (
            <TokenChip
              key={`crown-${player.id}-${suit}-${index}`}
              suit={suit}
              count={1}
              compact
              className="crown-rail-chip"
            />
          ))}
        </div>
      ) : (
        <span className="empty-note">None</span>
      )}
    </div>
  );

  const resources = (
    <div className="token-rail-group">
      <h3>Resources</h3>
      <TokenRow tokens={player.resources} compact fixedSuitSlots className="rail-resources-row" />
    </div>
  );

  return (
    <section className={`token-rail token-rail-${side}`} aria-label={`${player.id} crowns and resources`}>
      <div className="token-rail-inner">
        {side === 'human' ? (
          <>
            {crowns}
            {resources}
          </>
        ) : (
          <>
            {resources}
            {crowns}
          </>
        )}
      </div>
    </section>
  );
}

function DistrictColumn({ district }: { district: DistrictState }) {
  const markerName = districtMarkerName(district.markerSuitMask);

  return (
    <article className="district-column">
      <DistrictLane playerId={BOT_PLAYER} stack={district.stacks[BOT_PLAYER]} />

      <header className="district-header">
        <span className="district-id">{district.id}</span>
        <strong className="district-marker-name" title={markerName}>
          {markerName}
        </strong>
        {district.markerSuitMask.length > 0 ? (
          <TokenRow
            className="district-marker-tokens"
            tokens={district.markerSuitMask.reduce<
              Partial<Record<Suit, number>>
            >((acc, suit) => {
              acc[suit] = 1;
              return acc;
            }, {})}
            compact
          />
        ) : (
          <span
            className="district-marker-tokens district-marker-placeholder"
            aria-hidden="true"
          />
        )}
      </header>

      <DistrictLane
        playerId={HUMAN_PLAYER}
        stack={district.stacks[HUMAN_PLAYER]}
      />
    </article>
  );
}

function DistrictLane({
  playerId,
  stack,
}: {
  playerId: PlayerId;
  stack: DistrictStack;
}) {
  const deedProperty = stack.deed ? findProperty(stack.deed.cardId) : undefined;
  const deedTarget = deedProperty ? developmentCost(deedProperty) : undefined;
  const perspective: CardPerspective =
    playerId === BOT_PLAYER ? 'bot' : 'human';
  const laneCards: Array<{
    key: string;
    cardId: CardId;
    deedTokens?: Partial<Record<Suit, number>>;
    deedProgress?: number;
    deedTarget?: number;
  }> = stack.developed.map((cardId, index) => ({
    key: `developed-${cardId}-${index}`,
    cardId,
  }));

  if (stack.deed) {
    laneCards.push({
      key: `deed-${stack.deed.cardId}`,
      cardId: stack.deed.cardId,
      deedTokens: stack.deed.tokens,
      deedProgress: stack.deed.progress,
      deedTarget,
    });
  }

  const laneStyle = {
    '--stack-count': laneCards.length,
  } as CSSProperties;

  return (
    <section
      className={`district-lane${playerId === BOT_PLAYER ? ' is-bot' : ' is-human'}`}
    >
      <div
        className={`lane-stack-frame${playerId === BOT_PLAYER ? ' is-bot' : ''}`}
      >
        {laneCards.length > 0 ? (
          <div
            className={`lane-stack ${playerId === BOT_PLAYER ? 'is-bot' : 'is-human'}`}
            style={laneStyle}
          >
            {laneCards.map((laneCard, index) => (
              <div
                key={laneCard.key}
                className="lane-stack-card"
                style={
                  {
                    '--stack-position': index,
                    '--stack-z': index + 1,
                  } as CSSProperties
                }
              >
                <CardTile
                  cardId={laneCard.cardId}
                  deedTokens={laneCard.deedTokens}
                  deedProgress={laneCard.deedProgress}
                  deedTarget={laneCard.deedTarget}
                  perspective={perspective}
                />
              </div>
            ))}
          </div>
        ) : null}
      </div>
    </section>
  );
}

function CardTile({
  cardId,
  hidden,
  placeholder,
  deedTokens,
  deedProgress,
  deedTarget,
  perspective = 'human',
}: {
  cardId?: CardId;
  hidden?: boolean;
  placeholder?: boolean;
  deedTokens?: Partial<Record<Suit, number>>;
  deedProgress?: number;
  deedTarget?: number;
  perspective?: CardPerspective;
}) {
  if (placeholder) {
    return <div className="card-tile card-placeholder" aria-hidden="true" />;
  }

  if (hidden) {
    return <div className="card-tile card-back" title="Hidden card" />;
  }

  if (!cardId) {
    return null;
  }

  const card = CARD_BY_ID[cardId];
  const suits = card.kind === 'Excuse' ? [] : [...card.suits];
  const rank =
    card.kind === 'Property' || card.kind === 'Crown'
      ? String(card.rank)
      : card.kind === 'Pawn'
        ? 'P'
        : 'X';
  const hasDeedTokens = deedTokens
    ? tokenEntries(deedTokens).length > 0
    : false;
  const hasDeedProgress =
    deedProgress !== undefined && deedTarget !== undefined;
  const cardImage = getCardImage(cardId);

  const metadataRow = (
    <div className="card-row card-meta">
      <div className="card-meta-leading">
        <span className="card-rank">{rank}</span>
        <div className="card-suits-row">
          {suits.length > 0 ? (
            suits.map((suit) => (
              <SuitIcon
                key={`${cardId}-${suit}`}
                suit={suit}
                className="card-suit-icon"
              />
            ))
          ) : (
            <span className="card-suit-placeholder" />
          )}
        </div>
      </div>
      {hasDeedProgress ? (
        <div className="deed-progress">
          {deedProgress}/{deedTarget}
        </div>
      ) : (
        <span className="deed-progress-placeholder" aria-hidden="true" />
      )}
    </div>
  );

  const imageBody = (
    <div className="card-row card-body">
      <div className="card-image-frame" aria-hidden="true">
        <img className="card-image" src={cardImage} alt="" />
      </div>
      {hasDeedTokens && deedTokens ? (
        <TokenRow tokens={deedTokens} compact className="card-token-row" />
      ) : null}
    </div>
  );

  return (
    <div
      className={`card-tile${perspective === 'bot' ? ' perspective-bot' : ''}`}
      title={card.name}
    >
      {perspective === 'bot' ? imageBody : metadataRow}
      {perspective === 'bot' ? metadataRow : imageBody}
    </div>
  );
}

function TokenRow({
  tokens,
  compact,
  emptyLabel,
  fixedSuitSlots,
  className,
}: {
  tokens: Partial<Record<Suit, number>> | ResourcePool;
  compact?: boolean;
  emptyLabel?: string;
  fixedSuitSlots?: boolean;
  className?: string;
}) {
  const entries = fixedSuitSlots
    ? SUITS.map((suit) => ({ suit, count: tokens[suit] ?? 0 }))
    : tokenEntries(tokens);

  if (!fixedSuitSlots && entries.length === 0) {
    return <span className="empty-note">{emptyLabel ?? 'None'}</span>;
  }

  return (
    <div
      className={`token-row${compact ? ' compact' : ''}${fixedSuitSlots ? ' fixed-suits' : ''}${
        className ? ` ${className}` : ''
      }`}
    >
      {entries.map(({ suit, count }) => (
        <TokenChip key={suit} suit={suit} count={count} compact={compact} />
      ))}
    </div>
  );
}

function TokenChip({
  suit,
  count,
  compact,
  className,
}: {
  suit: Suit;
  count: number;
  compact?: boolean;
  className?: string;
}) {
  const isEmpty = count === 0;
  return (
    <span
      className={`token-chip${compact ? ' compact' : ''}${isEmpty ? ' empty' : ''}${
        className ? ` ${className}` : ''
      }`}
      title={`${suit} x${count}`}
      style={{ '--token-bg': SUIT_TOKEN_BG[suit] } as CSSProperties}
    >
      <SuitIcon suit={suit} className="chip-suit-icon" />
      {count > 1 && <span className="token-count">x{count}</span>}
    </span>
  );
}

function ScoreLine({ label, a, b }: { label: string; a: number; b: number }) {
  return (
    <p className="score-line">
      <span>{label}</span>
      <strong>
        A {a} - B {b}
      </strong>
    </p>
  );
}

function tokenEntries(
  tokens: Partial<Record<Suit, number>> | ResourcePool
): Array<{ suit: Suit; count: number }> {
  return SUITS.map((suit) => ({ suit, count: tokens[suit] ?? 0 })).filter(
    (entry) => entry.count > 0
  );
}

function crownsToTokens(
  crowns: readonly CardId[]
): Partial<Record<Suit, number>> {
  const tokens: Partial<Record<Suit, number>> = {};
  for (const crownId of crowns) {
    const card = CARD_BY_ID[crownId];
    if (!card || card.kind !== 'Crown') {
      continue;
    }
    suits.push(card.suits[0]);
  }
  return suits;
}

function RollResult({
  roll,
  taxSuit,
}: {
  roll: { die1: number; die2: number } | undefined;
  taxSuit: Suit | undefined;
}) {
  if (!roll) {
    return <p className="roll-value">-</p>;
  }

  return (
    <div className="roll-value" aria-label="Roll result">
      <span className="roll-item">
        <img
          src={dodecahedronDieIcon}
          alt="d10"
          title="d10"
          className="roll-die-icon"
        />
        <strong>{roll.die1}</strong>
      </span>
      <span className="roll-item">
        <img
          src={dodecahedronDieIcon}
          alt="d10"
          title="d10"
          className="roll-die-icon"
        />
        <strong>{roll.die2}</strong>
      </span>
      <span className="roll-item">
        <img src={cubeDieIcon} alt="d6" title="d6" className="roll-die-icon" />
        {taxSuit ? <TokenChip suit={taxSuit} count={1} compact className="roll-tax-chip" /> : <strong>-</strong>}
      </span>
    </div>
  );
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

function toPickerQuery(picker: ActionPickerState): ActionPickerQuery {
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

function districtMarkerName(markerSuitMask: readonly Suit[]): string {
  if (markerSuitMask.length === 0) {
    return 'the EXCUSE';
  }

  const key = suitMaskKey(markerSuitMask);
  const match = PAWN_CARDS.find((pawn) => suitMaskKey(pawn.suits) === key);
  return match?.name ?? 'Unknown Marker';
}

function suitMaskKey(suits: readonly Suit[]): string {
  return [...suits].sort().join('|');
}
