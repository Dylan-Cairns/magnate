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

  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const terminal = isTerminal(state);
  const activePlayerId = state.players[state.activePlayerIndex]?.id ?? HUMAN_PLAYER;
  const humanView = useMemo(() => toPlayerView(state, HUMAN_PLAYER), [state]);
  const score = useMemo(() => state.finalScore ?? scoreGame(state), [state]);
  const statusText = useMemo(() => deriveStatusLabel(humanView), [humanView]);

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

    try {
      setState(createInitialState(seed));
      setError(null);
      setBotThinking(false);
    } catch (err) {
      setError(`Failed to start game: ${errorMessage(err)}`);
    }
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
      <header className="top-bar">
        <div>
          <h1>Magnate</h1>
          <p className="subtitle">Human vs random bot</p>
        </div>
        <div className="controls">
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
        </div>
      </header>

      <section className="hud-strip">
        <HudItem label="Turn" value={String(humanView.turn)} />
        <HudItem label="Status" value={statusText} />
        <HudItem label="Active" value={humanView.activePlayerId} />
        <HudItem label="Income Roll" value={formatRoll(humanView.lastIncomeRoll)} />
        <HudItem
          label="Final Turns"
          value={humanView.finalTurnsRemaining !== undefined ? String(humanView.finalTurnsRemaining) : '-'}
        />
      </section>

      {error && (
        <section className="error-banner">
          <strong>Engine Error:</strong> {error}
        </section>
      )}

      <main className="layout">
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

        <aside className="side-pane">
          <section className="panel">
            <h2>{terminal ? 'Final Score' : 'Live Score'}</h2>
            <ScorePanel score={score} terminal={terminal} />
          </section>

          <section className="panel">
            <h2>Actions</h2>
            {terminal ? (
              <p className="empty-note">Game over.</p>
            ) : activePlayerId === HUMAN_PLAYER ? (
              humanActions.length === 0 ? (
                <p className="empty-note">No legal actions.</p>
              ) : (
                <div className="action-list">
                  {humanActions.map((action, index) => (
                    <button
                      key={`${action.type}-${index}-${JSON.stringify(action)}`}
                      type="button"
                      className="action-button"
                      onClick={() => handleHumanAction(action)}
                    >
                      <span className="action-kind">{action.type}</span>
                      <span className="action-text">{describeAction(action)}</span>
                    </button>
                  ))}
                </div>
              )
            ) : (
              <p className="empty-note">{botThinking ? 'Bot is thinking...' : 'Waiting for bot...'}</p>
            )}
          </section>

          <section className="panel">
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
    </div>
  );
}

function HudItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="hud-item">
      <span>{label}</span>
      <strong>{value}</strong>
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
  return (
    <section className={`player-panel${isActive ? ' is-active' : ''}`}>
      <header className="player-header">
        <h2>{title}</h2>
        <span className="player-meta">
          {player.id} · Hand {player.handCount}
        </span>
      </header>

      <div className="player-section">
        <h3>Resources</h3>
        <TokenRow tokens={player.resources} emptyLabel="No resources" />
      </div>

      <div className="player-section">
        <h3>Crowns</h3>
        <div className="card-row-wrap">
          {player.crowns.map((cardId, index) => (
            <CardTile key={`${cardId}-${index}`} cardId={cardId} compact />
          ))}
        </div>
      </div>

      <div className="player-section">
        <h3>{player.handHidden ? 'Hidden Hand' : 'Hand'}</h3>
        <div className="card-row-wrap">
          {player.handHidden
            ? Array.from({ length: player.handCount }).map((_, index) => (
                <CardTile key={`hidden-${player.id}-${index}`} hidden />
              ))
            : player.hand.map((cardId, index) => (
                <CardTile key={`${cardId}-${index}`} cardId={cardId} />
              ))}
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
  deedTokens,
  deedProgress,
  deedTarget,
}: {
  cardId?: CardId;
  hidden?: boolean;
  compact?: boolean;
  deedTokens?: Partial<Record<Suit, number>>;
  deedProgress?: number;
  deedTarget?: number;
}) {
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
}: {
  tokens: Partial<Record<Suit, number>> | ResourcePool;
  compact?: boolean;
  emptyLabel?: string;
}) {
  const entries = tokenEntries(tokens);
  if (entries.length === 0) {
    return <span className="empty-note">{emptyLabel ?? 'None'}</span>;
  }

  return (
    <div className={`token-row${compact ? ' compact' : ''}`}>
      {entries.map(({ suit, count }) => (
        <TokenChip key={suit} suit={suit} count={count} compact={compact} />
      ))}
    </div>
  );
}

function TokenChip({ suit, count, compact }: { suit: Suit; count: number; compact?: boolean }) {
  return (
    <span className={`token-chip${compact ? ' compact' : ''}`} title={`${suit} x${count}`}>
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

function deriveStatusLabel(view: ReturnType<typeof toPlayerView>): string {
  if (view.phase === 'GameOver') {
    return 'Game over';
  }
  if (view.phase === 'CollectIncome' && (view.pendingIncomeChoices?.length ?? 0) > 0) {
    return 'Choose income suit';
  }
  if (view.phase === 'ActionWindow') {
    return view.cardPlayedThisTurn
      ? 'Optional actions / End turn'
      : 'Choose turn action';
  }

  return 'Resolving turn';
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
