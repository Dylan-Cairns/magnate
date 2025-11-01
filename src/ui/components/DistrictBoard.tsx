import type { CSSProperties } from 'react';

import { CARD_BY_ID, PAWN_CARDS, type CardId } from '../../engine/cards';
import { districtScore } from '../../engine/scoring';
import { developmentCost, findProperty } from '../../engine/stateHelpers';
import type { DistrictStack, DistrictState, ObservedPlayerState, PlayerId, Suit } from '../../engine/types';
import { CardTile, type CardPerspective } from './CardTile';
import { TokenRow } from './TokenComponents';

function crownsToSuits(crowns: readonly CardId[]): Suit[] {
  const suits: Suit[] = [];
  for (const crownId of crowns) {
    const card = CARD_BY_ID[crownId];
    if (!card || card.kind !== 'Crown') {
      continue;
    }
    suits.push(card.suits[0]);
  }
  return suits;
}

function suitMaskKey(suits: readonly Suit[]): string {
  return [...suits].sort().join('|');
}

function districtMarkerName(markerSuitMask: readonly Suit[]): string {
  if (markerSuitMask.length === 0) {
    return 'Excuse';
  }

  const marker = PAWN_CARDS.find((card) => suitMaskKey(card.suits) === suitMaskKey(markerSuitMask));
  return marker?.name ?? markerSuitMask.join('/');
}

function markerSuitTokens(markerSuitMask: readonly Suit[]): Partial<Record<Suit, number>> {
  const tokens: Partial<Record<Suit, number>> = {};
  for (const suit of markerSuitMask) {
    tokens[suit] = 1;
  }
  return tokens;
}

function DistrictLane({
  playerId,
  stack,
  botPlayerId,
}: {
  playerId: PlayerId;
  stack: DistrictStack;
  botPlayerId: PlayerId;
}) {
  const deedProperty = stack.deed ? findProperty(stack.deed.cardId) : undefined;
  const deedTarget = deedProperty ? developmentCost(deedProperty) : undefined;
  const perspective: CardPerspective = playerId === botPlayerId ? 'bot' : 'human';
  const laneCards: Array<{
    key: string;
    cardId: CardId;
    deedTokens?: Partial<Record<Suit, number>>;
    deedProgress?: number;
    deedTarget?: number;
    inDevelopment?: boolean;
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
      inDevelopment: true,
    });
  }

  const laneStyle = {
    '--stack-count': laneCards.length,
  } as CSSProperties;

  return (
    <section className={`district-lane${playerId === botPlayerId ? ' is-bot' : ' is-human'}`}>
      <div className={`lane-stack-frame${playerId === botPlayerId ? ' is-bot' : ''}`}>
        {laneCards.length > 0 ? (
          <div className={`lane-stack ${playerId === botPlayerId ? 'is-bot' : 'is-human'}`} style={laneStyle}>
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
                  inDevelopment={laneCard.inDevelopment}
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

export function DistrictColumn({
  district,
  humanPlayerId,
  botPlayerId,
}: {
  district: DistrictState;
  humanPlayerId: PlayerId;
  botPlayerId: PlayerId;
}) {
  const markerName = districtMarkerName(district.markerSuitMask);
  const botDistrictScore = districtScore(district.stacks[botPlayerId]);
  const humanDistrictScore = districtScore(district.stacks[humanPlayerId]);
  const botLeadsDistrict = botDistrictScore > humanDistrictScore;
  const humanLeadsDistrict = humanDistrictScore > botDistrictScore;

  return (
    <article className="district-column">
      <DistrictLane
        playerId={botPlayerId}
        stack={district.stacks[botPlayerId]}
        botPlayerId={botPlayerId}
      />

      <div className="district-header-wrap">
        <span
          className={`district-lane-score district-lane-score-bot${botLeadsDistrict ? ' is-leading' : ''}`}
          aria-label={`District score: ${botDistrictScore}`}
        >
          {botDistrictScore}
        </span>
        <header className="district-header" title={markerName}>
          <span className="district-id">{district.id}</span>
          <strong className="district-marker-name" title={markerName}>
            {markerName}
          </strong>
          {district.markerSuitMask.length > 0 ? (
            <TokenRow className="district-marker-tokens" tokens={markerSuitTokens(district.markerSuitMask)} compact />
          ) : (
            <span className="district-marker-tokens district-marker-placeholder" aria-hidden="true" />
          )}
        </header>
        <span
          className={`district-lane-score district-lane-score-human${humanLeadsDistrict ? ' is-leading' : ''}`}
          aria-label={`District score: ${humanDistrictScore}`}
        >
          {humanDistrictScore}
        </span>
      </div>

      <DistrictLane
        playerId={humanPlayerId}
        stack={district.stacks[humanPlayerId]}
        botPlayerId={botPlayerId}
      />
    </article>
  );
}

export function PlayerTokenRail({
  player,
  side,
}: {
  player: ObservedPlayerState;
  side: 'human' | 'bot';
}) {
  const crowns = (
    <div className="token-rail-group">
      <h3>Crowns</h3>
      <TokenRow
        className="crowns-rail-row"
        tokens={crownsToSuits(player.crowns).reduce<Partial<Record<Suit, number>>>((acc, suit) => {
          acc[suit] = (acc[suit] ?? 0) + 1;
          return acc;
        }, {})}
        compact
      />
    </div>
  );

  const resources = (
    <div className="token-rail-group">
      <h3>Resources</h3>
      <TokenRow className="rail-resources-row" tokens={player.resources} compact fixedSuitSlots />
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
