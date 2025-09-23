import { useEffect, useMemo, useRef, useState } from 'react';

import { legalActions } from './engine/actionBuilders';
import { CARD_BY_ID, type CardId } from './engine/cards';
import { newGame } from './engine/game';
import { applyAction } from './engine/reducer';
import { isTerminal, scoreGame } from './engine/scoring';
import { developmentCost, findProperty, SUITS } from './engine/stateHelpers';
import { advanceToDecision } from './engine/turnFlow';
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

const HUMAN_PLAYER: PlayerId = 'PlayerA';
const BOT_PLAYER: PlayerId = 'PlayerB';
const BOT_DELAY_MS = 450;
const PLAYER_CROWN_SLOT_COUNT = 3;
const PLAYER_HAND_SLOT_COUNT = 3;
const TRADE_POPOVER_WIDTH_PX = 220;
const TRADE_POPOVER_MIN_HEIGHT_PX = 188;
const TRADE_POPOVER_GAP_PX = 8;
const VIEWPORT_PADDING_PX = 10;

type TradeAction = Extract<GameAction, { type: 'trade' }>;

type TradePickerState = {
  give: Suit;
  top: number;
  left: number;
};

type HumanActionListItem =
  | { kind: 'action'; action: Exclude<GameAction, { type: 'trade' }> }
  | { kind: 'trade-group'; give: Suit };

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
  return advanceToDecision(newGame(seed, { firstPlayer: HUMAN_PLAYER }));
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
  const [tradePicker, setTradePicker] = useState<TradePickerState | null>(null);

  const stateRef = useRef(state);
  const tradePopoverRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const terminal = isTerminal(state);
  const activePlayerId = state.players[state.activePlayerIndex]?.id ?? HUMAN_PLAYER;
  const humanView = useMemo(() => toPlayerView(state, HUMAN_PLAYER), [state]);
  const score = useMemo(() => state.finalScore ?? scoreGame(state), [state]);

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

  const tradePickerOptions = useMemo(() => {
    if (!tradePicker) {
      return [] as Suit[];
    }

    const receives = humanActions
      .filter((action): action is TradeAction => action.type === 'trade' && action.give === tradePicker.give)
      .map((action) => action.receive);

    return Array.from(new Set(receives));
  }, [humanActions, tradePicker]);

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

      const choice = actions[Math.floor(Math.random() * actions.length)];

      try {
        const next = advanceToDecision(applyAction(current, choice));
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
      setTradePicker(null);
      return;
    }

    if (tradePicker) {
      const stillLegal = humanActions.some(
        (action): action is TradeAction => action.type === 'trade' && action.give === tradePicker.give
      );
      if (!stillLegal) {
        setTradePicker(null);
      }
    }
  }, [activePlayerId, humanActions, terminal, tradePicker]);

  useEffect(() => {
    if (!tradePicker) {
      return;
    }

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) {
        return;
      }

      if (tradePopoverRef.current?.contains(target)) {
        return;
      }

      setTradePicker(null);
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setTradePicker(null);
      }
    };

    const handleScroll = () => {
      setTradePicker(null);
    };

    window.addEventListener('pointerdown', handlePointerDown);
    window.addEventListener('keydown', handleEscape);
    window.addEventListener('scroll', handleScroll, true);

    return () => {
      window.removeEventListener('pointerdown', handlePointerDown);
      window.removeEventListener('keydown', handleEscape);
      window.removeEventListener('scroll', handleScroll, true);
    };
  }, [tradePicker]);

  const handleHumanAction = (action: GameAction) => {
    if (terminal || activePlayerId !== HUMAN_PLAYER) {
      return;
    }

    try {
      const next = advanceToDecision(applyAction(state, action));
      setState(next);
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const handleReset = () => {
    const seed = seedInput.trim() || makeSeed();
    setSeedInput(seed);
    setTradePicker(null);

    try {
      setState(createInitialState(seed));
      setError(null);
      setBotThinking(false);
    } catch (err) {
      setError(`Failed to start game: ${errorMessage(err)}`);
    }
  };

  const handleTradeSelection = (receive: Suit) => {
    if (!tradePicker) {
      return;
    }

    const action: TradeAction = {
      type: 'trade',
      give: tradePicker.give,
      receive,
    };

    setTradePicker(null);
    handleHumanAction(action);
  };

  const openTradePicker = (give: Suit, trigger: HTMLButtonElement) => {
    const rect = trigger.getBoundingClientRect();
    const optionCount = humanActions.filter(
      (action): action is TradeAction => action.type === 'trade' && action.give === give
    ).length;
    const rowCount = Math.max(1, Math.ceil(optionCount / 2));
    const estimatedHeight = Math.max(TRADE_POPOVER_MIN_HEIGHT_PX, 116 + rowCount * 46);
    const maxLeft = window.innerWidth - TRADE_POPOVER_WIDTH_PX - VIEWPORT_PADDING_PX;
    const maxTop = window.innerHeight - estimatedHeight - VIEWPORT_PADDING_PX;

    const left = clamp(rect.right + TRADE_POPOVER_GAP_PX, VIEWPORT_PADDING_PX, maxLeft);
    const top = clamp(rect.top, VIEWPORT_PADDING_PX, maxTop);

    setTradePicker({ give, left, top });
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

  const recentLog = humanView.log.slice(-12).reverse();

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
            <h2>Actions</h2>
            <div className="actions-body">
              {terminal ? (
                <p className="empty-note">Game over.</p>
              ) : activePlayerId === HUMAN_PLAYER ? (
                humanActionItems.length === 0 ? (
                  <p className="empty-note">No legal actions.</p>
                ) : (
                  <div className="action-list">
                    {humanActionItems.map((item, index) => {
                      if (item.kind === 'trade-group') {
                        return (
                          <button
                            key={`trade-group-${item.give}-${index}`}
                            type="button"
                            className="action-button has-submenu"
                            onClick={(event) => {
                              const trigger = event.currentTarget;
                              if (tradePicker?.give === item.give) {
                                setTradePicker(null);
                                return;
                              }
                              openTradePicker(item.give, trigger);
                            }}
                          >
                            <span className="action-kind">trade</span>
                            <span className="action-text">Trade {SUIT_EMOJI[item.give]}x3</span>
                          </button>
                        );
                      }

                      return (
                        <button
                          key={`${item.action.type}-${index}-${JSON.stringify(item.action)}`}
                          type="button"
                          className="action-button"
                          onClick={() => handleHumanAction(item.action)}
                        >
                          <span className="action-kind">{item.action.type}</span>
                          <span className="action-text">{describeAction(item.action)}</span>
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
          </section>

          <section className="panel controls-panel">
            <label className="seed-label" htmlFor="seed-input">
              Seed
            </label>
            <input
              id="seed-input"
              className="seed-input"
              value={seedInput}
              onChange={(event) => setSeedInput(event.target.value)}
            />
            <button className="reset-button" type="button" onClick={handleReset}>
              New Game
            </button>
          </section>

          <section className="panel">
            <h2>Live Score</h2>
            <ScorePanel score={score} terminal={terminal} />
          </section>

          <section className="panel">
            <h2>Roll Result</h2>
            <p className="roll-value">{formatRoll(humanView.lastIncomeRoll)}</p>
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

      {tradePicker && (
        <section
          ref={tradePopoverRef}
          className="panel trade-popover"
          role="dialog"
          aria-label="Choose trade receive suit"
          style={{ top: `${tradePicker.top}px`, left: `${tradePicker.left}px` }}
        >
          <h2>Trade {SUIT_EMOJI[tradePicker.give]}x3 for</h2>

          {tradePickerOptions.length === 0 ? (
            <p className="empty-note">No trade targets available.</p>
          ) : (
            <div className="trade-choice-list">
              {tradePickerOptions.map((suit) => (
                <button
                  key={`trade-choice-${tradePicker.give}-${suit}`}
                  type="button"
                  className="trade-choice-button"
                  onClick={() => handleTradeSelection(suit)}
                >
                  {SUIT_EMOJI[suit]} x1
                </button>
              ))}
            </div>
          )}

          <button type="button" className="trade-cancel-button" onClick={() => setTradePicker(null)}>
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
                return <CardTile key={`crown-slot-${player.id}-${index}`} compact placeholder />;
              }

              return <CardTile key={`${cardId}-${index}`} cardId={cardId} compact />;
            })}
          </div>
        </div>

        <div className="player-section hand-section">
          <h3>{player.handHidden ? 'Hidden Hand' : 'Hand'}</h3>
          <div className="card-row-wrap fixed-slots">
            {Array.from({ length: handSlots }).map((_, index) => {
              if (player.handHidden) {
                return index < player.handCount ? (
                  <CardTile key={`hidden-${player.id}-${index}`} hidden compact />
                ) : (
                  <CardTile key={`hidden-slot-${player.id}-${index}`} compact placeholder />
                );
              }

              const cardId = player.hand[index];
              if (!cardId) {
                return <CardTile key={`hand-slot-${player.id}-${index}`} compact placeholder />;
              }

              return <CardTile key={`${cardId}-${index}`} cardId={cardId} compact />;
            })}
          </div>
        </div>
      </div>
    </section>
  );
}

function DistrictColumn({ district }: { district: DistrictState }) {
  return (
    <article className="district-column">
      <header className="district-header">
        <strong>{district.id}</strong>
        {district.markerSuitMask.length > 0 ? (
          <TokenRow
            tokens={district.markerSuitMask.reduce<Partial<Record<Suit, number>>>((acc, suit) => {
              acc[suit] = 1;
              return acc;
            }, {})}
            compact
          />
        ) : (
          <span className="excuse-pill">Excuse</span>
        )}
      </header>

      <DistrictLane label="Bot" stack={district.stacks[BOT_PLAYER]} />
      <DistrictLane label="You" stack={district.stacks[HUMAN_PLAYER]} />
    </article>
  );
}

function DistrictLane({ label, stack }: { label: string; stack: DistrictStack }) {
  const deedProperty = stack.deed ? findProperty(stack.deed.cardId) : undefined;
  const deedTarget = deedProperty ? developmentCost(deedProperty) : undefined;
  const empty = stack.developed.length === 0 && !stack.deed;

  return (
    <section className="district-lane">
      <header>{label}</header>
      <div className="lane-cards">
        {stack.developed.map((cardId, index) => (
          <CardTile key={`${cardId}-${index}`} cardId={cardId} />
        ))}
        {stack.deed && (
          <CardTile
            cardId={stack.deed.cardId}
            deedTokens={stack.deed.tokens}
            deedProgress={stack.deed.progress}
            deedTarget={deedTarget}
          />
        )}
        {empty && <span className="empty-note">No cards</span>}
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
}: {
  cardId?: CardId;
  hidden?: boolean;
  compact?: boolean;
  placeholder?: boolean;
  deedTokens?: Partial<Record<Suit, number>>;
  deedProgress?: number;
  deedTarget?: number;
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

  return (
    <div className={`card-tile${compact ? ' compact' : ''}`} title={card.name}>
      <div className="card-row card-top">
        <span className="card-rank">{rank}</span>
        <div className="card-suits">
          {suits.map((suit) => (
            <span key={`${cardId}-${suit}`}>{SUIT_EMOJI[suit]}</span>
          ))}
        </div>
      </div>

      <div className="card-row card-bottom">
        {deedTokens && tokenEntries(deedTokens).length > 0 ? (
          <TokenRow tokens={deedTokens} compact />
        ) : (
          <span className="card-bottom-placeholder" />
        )}
      </div>

      {deedProgress !== undefined && deedTarget !== undefined && (
        <div className="deed-progress">
          {deedProgress}/{deedTarget}
        </div>
      )}
    </div>
  );
}

function TokenRow({
  tokens,
  compact,
  emptyLabel,
  fixedSuitSlots,
}: {
  tokens: Partial<Record<Suit, number>> | ResourcePool;
  compact?: boolean;
  emptyLabel?: string;
  fixedSuitSlots?: boolean;
}) {
  const entries = fixedSuitSlots
    ? SUITS.map((suit) => ({ suit, count: tokens[suit] ?? 0 }))
    : tokenEntries(tokens);

  if (!fixedSuitSlots && entries.length === 0) {
    return <span className="empty-note">{emptyLabel ?? 'None'}</span>;
  }

  return (
    <div className={`token-row${compact ? ' compact' : ''}${fixedSuitSlots ? ' fixed-suits' : ''}`}>
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

function formatRoll(roll: { die1: number; die2: number } | undefined): string {
  if (!roll) {
    return '-';
  }
  return `${roll.die1} / ${roll.die2}`;
}

function describeAction(action: GameAction): string {
  switch (action.type) {
    case 'end-turn':
      return 'Draw card and end turn';
    case 'trade':
      return `Trade ${SUIT_EMOJI[action.give]}x3 for ${SUIT_EMOJI[action.receive]}x1`;
    case 'sell-card':
      return `Sell ${cardSummary(action.cardId)}`;
    case 'buy-deed':
      return `Buy deed ${cardSummary(action.cardId)} in ${action.districtId}`;
    case 'develop-deed':
      return `Develop deed ${cardSummary(action.cardId)} in ${action.districtId} (${formatTokens(
        action.tokens
      )})`;
    case 'develop-outright':
      return `Develop ${cardSummary(action.cardId)} in ${action.districtId} (${formatTokens(
        action.payment
      )})`;
    case 'choose-income-suit':
      return `Choose ${SUIT_EMOJI[action.suit]} income for ${cardSummary(action.cardId)} in ${action.districtId}`;
  }
}

function cardSummary(cardId: CardId): string {
  const card = CARD_BY_ID[cardId];
  const rank =
    card.kind === 'Property' || card.kind === 'Crown'
      ? String(card.rank)
      : card.kind === 'Pawn'
        ? 'P'
        : 'X';
  const suits = card.kind === 'Excuse' ? '' : card.suits.map((suit) => SUIT_EMOJI[suit]).join('');
  return `${rank}${suits}`;
}

function formatTokens(tokens: Partial<Record<Suit, number>>): string {
  const entries = tokenEntries(tokens);
  if (entries.length === 0) {
    return '-';
  }
  return entries.map((entry) => `${SUIT_EMOJI[entry.suit]}x${entry.count}`).join(' ');
}

function clamp(value: number, min: number, max: number): number {
  if (max < min) {
    return min;
  }
  return Math.max(min, Math.min(value, max));
}

function buildHumanActionList(actions: readonly GameAction[]): HumanActionListItem[] {
  const result: HumanActionListItem[] = [];
  const seenTradeGive = new Set<Suit>();

  for (const action of actions) {
    if (action.type === 'trade') {
      if (seenTradeGive.has(action.give)) {
        continue;
      }
      seenTradeGive.add(action.give);
      result.push({ kind: 'trade-group', give: action.give });
      continue;
    }

    result.push({ kind: 'action', action });
  }

  return result;
}
