import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';

import { legalActions } from './engine/actionBuilders';
import cubeDieIcon from './assets/icons/cube.png';
import dodecahedronDieIcon from './assets/icons/dodecahedron.png';
import { CARD_BY_ID, PAWN_CARDS, type CardId } from './engine/cards';
import { createSession, stepToDecision } from './engine/session';
import { isTerminal, scoreLive } from './engine/scoring';
import { developmentCost, findProperty, SUITS } from './engine/stateHelpers';
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
import { randomPolicy } from './policies/randomPolicy';
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

const HUMAN_PLAYER: PlayerId = 'PlayerA';
const BOT_PLAYER: PlayerId = 'PlayerB';
const BOT_DELAY_MS = 450;
const PLAYER_CROWN_SLOT_COUNT = 3;
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

const SUIT_EMOJI: Record<Suit, string> = {
  Moons: '🌙',
  Suns: '☀️',
  Waves: '🌊',
  Leaves: '🍃',
  Wyrms: '🐉',
  Knots: '🪢',
};

function makeSeed(): string {
  return `seed-${Date.now()}`;
}

function createInitialState(seed: string): GameState {
  return createSession(seed, HUMAN_PLAYER);
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

export function App() {
  const [seedInput, setSeedInput] = useState<string>(() => makeSeed());
  const [state, setState] = useState<GameState>(() => createInitialState(seedInput));
  const [error, setError] = useState<string | null>(null);
  const [botThinking, setBotThinking] = useState<boolean>(false);
  const [actionPicker, setActionPicker] = useState<ActionPickerState | null>(null);

  const stateRef = useRef(state);
  const actionPopoverRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const terminal = isTerminal(state);
  const isLastTurn = !terminal && (state.finalTurnsRemaining ?? 0) > 0;
  const activePlayerId = state.players[state.activePlayerIndex]?.id ?? HUMAN_PLAYER;
  const humanView = useMemo(() => toPlayerView(state, HUMAN_PLAYER), [state]);
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

  const humanActionItems = useMemo(() => buildHumanActionList(humanActions), [humanActions]);

  const actionPickerOptions = useMemo(() => {
    if (!actionPicker) {
      return [];
    }
    return buildPickerOptions(toPickerQuery(actionPicker), humanActions, SUIT_EMOJI);
  }, [actionPicker, humanActions]);

  const actionPickerTitle = useMemo((): string => {
    if (!actionPicker) {
      return '';
    }
    return pickerTitle(toPickerQuery(actionPicker), SUIT_EMOJI);
  }, [actionPicker]);

  useEffect(() => {
    if (terminal || activePlayerId !== BOT_PLAYER) {
      setBotThinking(false);
      return;
    }

    setBotThinking(true);
    const timerId = window.setTimeout(() => {
      const current = stateRef.current;
      const currentActive = current.players[current.activePlayerIndex]?.id;
      if (isTerminal(current) || currentActive !== BOT_PLAYER) {
        setBotThinking(false);
        return;
      }

      const actions = legalActions(current);
      if (actions.length === 0) {
        setError('Bot has no legal actions.');
        setBotThinking(false);
        return;
      }
      const botView = toPlayerView(current, BOT_PLAYER);
      const choice = randomPolicy.selectAction({
        view: botView,
        legalActions: actions,
        random: Math.random,
      });
      if (!choice) {
        setError('Bot policy could not select an action.');
        setBotThinking(false);
        return;
      }

      try {
        const next = stepToDecision(current, choice);
        setState(next);
        setError(null);
      } catch (err) {
        setError(`Bot action failed: ${errorMessage(err)}`);
      } finally {
        setBotThinking(false);
      }
    }, BOT_DELAY_MS);

    return () => {
      window.clearTimeout(timerId);
    };
  }, [activePlayerId, state, terminal]);

  useEffect(() => {
    if (terminal || activePlayerId !== HUMAN_PLAYER) {
      setActionPicker(null);
      return;
    }

    if (actionPicker) {
      const stillLegal = pickerStillLegal(toPickerQuery(actionPicker), humanActions);
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

  const openTradePicker = (give: Suit, trigger: HTMLButtonElement, optionCount: number) => {
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

  const pickerPosition = (trigger: HTMLButtonElement, optionCount: number): { top: number; left: number } => {
    const rect = trigger.getBoundingClientRect();
    const rowCount = Math.max(1, Math.ceil(optionCount / 2));
    const estimatedHeight = Math.max(TRADE_POPOVER_MIN_HEIGHT_PX, 116 + rowCount * 46);
    const maxLeft = window.innerWidth - TRADE_POPOVER_WIDTH_PX - VIEWPORT_PADDING_PX;
    const maxTop = window.innerHeight - estimatedHeight - VIEWPORT_PADDING_PX;

    const left = clamp(rect.right + TRADE_POPOVER_GAP_PX, VIEWPORT_PADDING_PX, maxLeft);
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

  return (
    <div className="app-shell">
      {error && (
        <section className="error-banner">
          <strong>Engine Error:</strong> {error}
        </section>
      )}

      <main className="layout">
        <aside className="actions-pane">
          <section className="panel actions-panel">
            <div className="actions-heading">
              <h2>Actions</h2>
              {isLastTurn && <span className="last-turn-badge">Last Turn</span>}
            </div>
            <div className="actions-body">
              {terminal ? (
                <p className="empty-note">Game over.</p>
              ) : activePlayerId === HUMAN_PLAYER ? (
                humanActionItems.length === 0 ? (
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
                              <span className="action-text">{describeAction(onlyOption, SUIT_EMOJI)}</span>
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
                              if (actionPicker?.kind === 'trade' && actionPicker.give === item.give) {
                                setActionPicker(null);
                                return;
                              }
                              openTradePicker(item.give, trigger, item.options.length);
                            }}
                          >
                            <span className="action-kind">trade</span>
                            <span className="action-text">Trade {SUIT_EMOJI[item.give]}x3</span>
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
                              <span className="action-kind">{onlyOption.type}</span>
                              <span className="action-text">{describeAction(onlyOption, SUIT_EMOJI)}</span>
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
                            <span className="action-text">Buy deed {cardSummary(item.cardId, SUIT_EMOJI)}</span>
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
                              <span className="action-kind">{onlyOption.type}</span>
                              <span className="action-text">{describeAction(onlyOption, SUIT_EMOJI)}</span>
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
                            <span className="action-kind">develop-deed</span>
                            <span className="action-text">
                              Develop deed {cardSummary(item.cardId, SUIT_EMOJI)} in {item.districtId}
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
                              <span className="action-kind">{onlyOption.type}</span>
                              <span className="action-text">{describeAction(onlyOption, SUIT_EMOJI)}</span>
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
                                actionPicker.actionType === 'develop-outright' &&
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
                            <span className="action-kind">develop-outright</span>
                            <span className="action-text">
                              Develop {cardSummary(item.cardId, SUIT_EMOJI)} ({formatTokens(item.payment, SUIT_EMOJI)})
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
                          <span className="action-kind">{item.action.type}</span>
                          <span className="action-text">{describeAction(item.action, SUIT_EMOJI)}</span>
                        </button>
                      );
                    })}
                  </div>
                )
              ) : (
                <p className="empty-note">{botThinking ? 'Bot is thinking...' : 'Waiting for bot...'}</p>
              )}
            </div>
          </section>
        </aside>

        <section className="board-pane">
          <PlayerPanel
            title="Bot"
            player={botPlayer}
            isActive={humanView.activePlayerId === BOT_PLAYER}
          />

          <div className="district-strip" aria-label="District board">
            {humanView.districts.map((district) => (
              <DistrictColumn key={district.id} district={district} />
            ))}
          </div>

          <PlayerPanel
            title="You"
            player={humanPlayer}
            isActive={humanView.activePlayerId === HUMAN_PLAYER}
          />
        </section>

        <aside className="info-pane">
          <section className="panel brand-panel">
            <h1>Magnate</h1>
            <div className="brand-controls">
              <input
                id="seed-input"
                aria-label="Seed"
                className="seed-input"
                value={seedInput}
                onChange={(event) => setSeedInput(event.target.value)}
              />
              <button className="reset-button" type="button" onClick={handleReset}>
                New Game
              </button>
            </div>
          </section>

          <section className="panel">
            <h2>Live Score</h2>
            <ScorePanel score={score} terminal={terminal} />
          </section>

          <section className="panel">
            <h2>Deck State</h2>
            <div className="meta-grid">
              <p className="meta-line">
                <span>Cards Remaining</span>
                <strong>{humanView.deck.drawCount}</strong>
              </p>
              <p className="meta-line">
                <span>Reshuffles Remaining</span>
                <strong>{reshufflesRemaining}</strong>
              </p>
            </div>
          </section>

          <section className="panel">
            <h2>Roll Result</h2>
            <RollResult roll={humanView.lastIncomeRoll} taxSuit={humanView.lastTaxSuit} />
          </section>

          <section className="panel log-panel">
            <h2>Log</h2>
            {recentLog.length === 0 ? (
              <p className="empty-note">No actions yet.</p>
            ) : (
              <ol className="log-list">
                {recentLog.map((entry, index) => (
                  <li key={`${entry.turn}-${entry.phase}-${entry.summary}-${index}`}>
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
          style={{ top: `${actionPicker.top}px`, left: `${actionPicker.left}px` }}
        >
          <h2>{actionPickerTitle}</h2>

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
                  {option.label}
                </button>
              ))}
            </div>
          )}

          <button type="button" className="trade-cancel-button" onClick={() => setActionPicker(null)}>
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
}: {
  title: string;
  player: ObservedPlayerState;
  isActive: boolean;
}) {
  const crownSlots = Math.max(PLAYER_CROWN_SLOT_COUNT, player.crowns.length);
  const handCardCount = player.handHidden ? player.handCount : player.hand.length;
  const handSlots = Math.max(PLAYER_HAND_SLOT_COUNT, handCardCount);
  const cardPerspective: CardPerspective = player.id === BOT_PLAYER ? 'bot' : 'human';

  return (
    <section className={`player-panel${isActive ? ' is-active' : ''}`}>
      <header className="player-header">
        <h2>{title}</h2>
        <span className="player-meta">
          {player.id} · Hand {player.handCount}
        </span>
      </header>

      <div className="player-row">
        <div className="player-section resources-section">
          <h3>Resources</h3>
          <TokenRow tokens={player.resources} fixedSuitSlots />
        </div>

        <div className="player-section crowns-section">
          <h3>Crowns</h3>
          <div className="card-row-wrap fixed-slots">
            {Array.from({ length: crownSlots }).map((_, index) => {
              const cardId = player.crowns[index];
              if (!cardId) {
                return <CardTile key={`crown-slot-${player.id}-${index}`} placeholder />;
              }

              return <CardTile key={`${cardId}-${index}`} cardId={cardId} perspective={cardPerspective} />;
            })}
          </div>
        </div>

        <div className="player-section hand-section">
          <h3>{player.handHidden ? 'Hidden Hand' : 'Hand'}</h3>
          <div className="card-row-wrap fixed-slots">
            {Array.from({ length: handSlots }).map((_, index) => {
              if (player.handHidden) {
                return index < player.handCount ? (
                  <CardTile key={`hidden-${player.id}-${index}`} hidden />
                ) : (
                  <CardTile key={`hidden-slot-${player.id}-${index}`} placeholder />
                );
              }

              const cardId = player.hand[index];
              if (!cardId) {
                return <CardTile key={`hand-slot-${player.id}-${index}`} placeholder />;
              }

              return <CardTile key={`${cardId}-${index}`} cardId={cardId} />;
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

function DistrictColumn({ district }: { district: DistrictState }) {
  const markerName = districtMarkerName(district.markerSuitMask);

  return (
    <article className="district-column">
      <DistrictLane label="Bot" playerId={BOT_PLAYER} stack={district.stacks[BOT_PLAYER]} />

      <header className="district-header">
        <span className="district-id">{district.id}</span>
        <strong className="district-marker-name" title={markerName}>
          {markerName}
        </strong>
        {district.markerSuitMask.length > 0 ? (
          <TokenRow
            className="district-marker-tokens"
            tokens={district.markerSuitMask.reduce<Partial<Record<Suit, number>>>((acc, suit) => {
              acc[suit] = 1;
              return acc;
            }, {})}
            compact
          />
        ) : null}
      </header>

      <DistrictLane label="You" playerId={HUMAN_PLAYER} stack={district.stacks[HUMAN_PLAYER]} />
    </article>
  );
}

function DistrictLane({
  label,
  playerId,
  stack,
}: {
  label: string;
  playerId: PlayerId;
  stack: DistrictStack;
}) {
  const deedProperty = stack.deed ? findProperty(stack.deed.cardId) : undefined;
  const deedTarget = deedProperty ? developmentCost(deedProperty) : undefined;
  const perspective: CardPerspective = playerId === BOT_PLAYER ? 'bot' : 'human';
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
    <section className={`district-lane${playerId === BOT_PLAYER ? ' is-bot' : ' is-human'}`}>
      <header>{label}</header>
      <div className={`lane-stack-frame${playerId === BOT_PLAYER ? ' is-bot' : ''}`}>
        {laneCards.length > 0 ? (
          <div className={`lane-stack ${playerId === BOT_PLAYER ? 'is-bot' : 'is-human'}`} style={laneStyle}>
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
        ) : (
          <span className="empty-note">No cards</span>
        )}
      </div>
    </section>
  );
}

function CardTile({
  cardId,
  hidden,
  compact,
  placeholder,
  deedTokens,
  deedProgress,
  deedTarget,
  perspective = 'human',
}: {
  cardId?: CardId;
  hidden?: boolean;
  compact?: boolean;
  placeholder?: boolean;
  deedTokens?: Partial<Record<Suit, number>>;
  deedProgress?: number;
  deedTarget?: number;
  perspective?: CardPerspective;
}) {
  if (placeholder) {
    return <div className={`card-tile card-placeholder${compact ? ' compact' : ''}`} aria-hidden="true" />;
  }

  if (hidden) {
    return <div className={`card-tile card-back${compact ? ' compact' : ''}`} title="Hidden card" />;
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
  const hasDeedTokens = deedTokens ? tokenEntries(deedTokens).length > 0 : false;
  const hasDeedProgress = deedProgress !== undefined && deedTarget !== undefined;

  const rankAndSuits = (
    <div className="card-row card-top">
      <span className="card-rank">{rank}</span>
      <div className="card-suits-row">
        {suits.length > 0 ? (
          suits.map((suit) => <span key={`${cardId}-${suit}`}>{SUIT_EMOJI[suit]}</span>)
        ) : (
          <span className="card-suit-placeholder" />
        )}
      </div>
    </div>
  );

  const progressSlot = (
    <div className="card-progress-slot">
      {hasDeedProgress ? (
        <div className="deed-progress">
          {deedProgress}/{deedTarget}
        </div>
      ) : (
        <span className="deed-progress-placeholder" aria-hidden="true" />
      )}
    </div>
  );

  return (
    <div className={`card-tile${compact ? ' compact' : ''}${perspective === 'bot' ? ' perspective-bot' : ''}`} title={card.name}>
      {perspective === 'bot' ? progressSlot : rankAndSuits}

      <div className="card-row card-body">
        {hasDeedTokens && deedTokens ? (
          <TokenRow tokens={deedTokens} compact className="card-token-row" />
        ) : (
          <span className="card-chip-placeholder" />
        )}
      </div>

      {perspective === 'bot' ? rankAndSuits : progressSlot}
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

function TokenChip({ suit, count, compact }: { suit: Suit; count: number; compact?: boolean }) {
  const isEmpty = count === 0;
  return (
    <span className={`token-chip${compact ? ' compact' : ''}${isEmpty ? ' empty' : ''}`} title={`${suit} x${count}`}>
      <span>{SUIT_EMOJI[suit]}</span>
      {count > 1 && <span className="token-count">x{count}</span>}
    </span>
  );
}

function ScorePanel({
  score,
  terminal,
}: {
  score: FinalScore;
  terminal: boolean;
}) {
  return (
    <div className="score-grid">
      <p className="score-result">
        {terminal ? 'Winner' : 'Leader'}: <strong>{score.winner}</strong> ({score.decidedBy})
      </p>
      <ScoreLine label="Districts" a={score.districtPoints.PlayerA} b={score.districtPoints.PlayerB} />
      <ScoreLine label="Rank Total" a={score.rankTotals.PlayerA} b={score.rankTotals.PlayerB} />
      <ScoreLine label="Resources" a={score.resourceTotals.PlayerA} b={score.resourceTotals.PlayerB} />
    </div>
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

function tokenEntries(tokens: Partial<Record<Suit, number>> | ResourcePool): Array<{ suit: Suit; count: number }> {
  return SUITS.map((suit) => ({ suit, count: tokens[suit] ?? 0 })).filter(
    (entry) => entry.count > 0
  );
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
        <img src={dodecahedronDieIcon} alt="d10" title="d10" className="roll-die-icon" />
        <strong>{roll.die1}</strong>
      </span>
      <span className="roll-item">
        <img src={dodecahedronDieIcon} alt="d10" title="d10" className="roll-die-icon" />
        <strong>{roll.die2}</strong>
      </span>
      <span className="roll-item">
        <img src={cubeDieIcon} alt="d6" title="d6" className="roll-die-icon" />
        <strong>{taxSuit ?? '-'}</strong>
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

function toPickerQuery(picker: ActionPickerState): ActionPickerQuery {
  if (picker.kind === 'trade') {
    return { kind: 'trade', give: picker.give };
  }
  if (picker.kind === 'deed-payment') {
    return { kind: 'deed-payment', cardId: picker.cardId, districtId: picker.districtId };
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
