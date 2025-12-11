import { useEffect, useRef, useState } from 'react';

import { CARD_BY_ID, type CardId } from '../../engine/cards';
import type { PlayerId, Suit } from '../../engine/types';
import {
  CARD_BACK_IMAGE,
  getCardImage,
  isCardImageUrlReady,
  preloadCardImageUrl,
} from '../cardImages';
import { SuitIcon } from '../suitIcons';
import { TokenChip, tokenEntries } from './TokenComponents';
import {
  buildDeedProgressArcPath,
  canonicalDeedProgressRatio,
  DEED_PROGRESS_ANIMATION_DURATION_MS,
  DEED_PROGRESS_RING_RADIUS,
  shouldAnimateDeedProgress,
  tweenAnimatedDeedProgressRatio,
} from './deedProgress';
import { layoutDeedTokensBySide } from './deedTokenLayout';

export type CardPerspective = 'human' | 'bot';
const LAST_DEED_PROGRESS_RATIO_BY_CARD = new Map<CardId, number>();

export function CardTile({
  cardId,
  hidden,
  placeholder,
  deedTokens,
  deedProgress,
  deedTarget,
  inDevelopment,
  perspective = 'human',
  handOwnerId,
  handCardId,
  handSlotKind,
  animateDeedProgress = true,
}: {
  cardId?: CardId;
  hidden?: boolean;
  placeholder?: boolean;
  deedTokens?: Partial<Record<Suit, number>>;
  deedProgress?: number;
  deedTarget?: number;
  inDevelopment?: boolean;
  perspective?: CardPerspective;
  handOwnerId?: PlayerId;
  handCardId?: CardId;
  handSlotKind?: 'occupied' | 'hidden' | 'empty';
  animateDeedProgress?: boolean;
}) {
  if (placeholder) {
    return (
      <div
        className="card-tile card-placeholder"
        aria-hidden="true"
        data-hand-owner-id={handOwnerId}
        data-hand-card-id={handCardId}
        data-hand-slot-kind={handSlotKind}
      />
    );
  }

  if (hidden) {
    return (
      <div
        className="card-tile card-back"
        title="Hidden card"
        data-hand-owner-id={handOwnerId}
        data-hand-card-id={handCardId}
        data-hand-slot-kind={handSlotKind}
      />
    );
  }

  if (!cardId) {
    return null;
  }

  const card = CARD_BY_ID[cardId];
  const cardImage = getCardImage(cardId);
  const suits = card.kind === 'Excuse' ? [] : [...card.suits];
  const rank =
    card.kind === 'Property' || card.kind === 'Crown'
      ? String(card.rank)
      : card.kind === 'Pawn'
        ? 'P'
        : 'X';
  const deedTokenEntries = deedTokens ? tokenEntries(deedTokens) : [];
  const hasDeedTokens = deedTokenEntries.length > 0;
  const showDeedTokenRails = Boolean(inDevelopment) || hasDeedTokens;
  const deedTokensBySide = layoutDeedTokensBySide(cardId, perspective, deedTokenEntries, {
    resetWhenEmpty: Boolean(inDevelopment) && deedTokenEntries.length === 0,
  });
  const hasDeedProgress = deedProgress !== undefined && deedTarget !== undefined;
  const progressValue = deedProgress ?? 0;
  const progressTarget = deedTarget ?? 0;
  const deedProgressRatio = canonicalDeedProgressRatio(progressValue, progressTarget);
  const [cardImageReady, setCardImageReady] = useState<boolean>(() =>
    isCardImageUrlReady(cardImage)
  );
  const [animatedDeedProgressRatio, setAnimatedDeedProgressRatio] = useState<number>(deedProgressRatio);
  const animatedRatioRef = useRef(animatedDeedProgressRatio);
  const initializedRef = useRef(false);
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (isCardImageUrlReady(cardImage)) {
      setCardImageReady(true);
      return () => {
        cancelled = true;
      };
    }

    setCardImageReady(false);
    void preloadCardImageUrl(cardImage)
      .then(() => {
        if (cancelled) {
          return;
        }
        setCardImageReady(true);
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        setCardImageReady(false);
      });

    return () => {
      cancelled = true;
    };
  }, [cardImage]);

  useEffect(() => {
    animatedRatioRef.current = animatedDeedProgressRatio;
  }, [animatedDeedProgressRatio]);

  useEffect(() => {
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    const targetRatio = deedProgressRatio;
    let currentRatio = animatedRatioRef.current;

    if (!initializedRef.current) {
      initializedRef.current = true;
      const rememberedRatio = LAST_DEED_PROGRESS_RATIO_BY_CARD.get(cardId);
      if (rememberedRatio !== undefined) {
        currentRatio = rememberedRatio;
        animatedRatioRef.current = rememberedRatio;
        setAnimatedDeedProgressRatio(rememberedRatio);
      }
    }

    if (
      !animateDeedProgress
      || !shouldAnimateDeedProgress(currentRatio, targetRatio)
    ) {
      animatedRatioRef.current = targetRatio;
      setAnimatedDeedProgressRatio(targetRatio);
      LAST_DEED_PROGRESS_RATIO_BY_CARD.set(cardId, targetRatio);
      return;
    }

    let startTime: number | null = null;
    const fromRatio = currentRatio;

    const tick = (timestamp: number) => {
      if (startTime === null) {
        startTime = timestamp;
      }
      const elapsed = timestamp - startTime;
      const nextRatio = tweenAnimatedDeedProgressRatio(
        fromRatio,
        targetRatio,
        elapsed,
        DEED_PROGRESS_ANIMATION_DURATION_MS
      );
      animatedRatioRef.current = nextRatio;
      setAnimatedDeedProgressRatio(nextRatio);
      LAST_DEED_PROGRESS_RATIO_BY_CARD.set(cardId, nextRatio);

      if (elapsed < DEED_PROGRESS_ANIMATION_DURATION_MS) {
        animationFrameRef.current = window.requestAnimationFrame(tick);
        return;
      }

      animatedRatioRef.current = targetRatio;
      setAnimatedDeedProgressRatio(targetRatio);
      LAST_DEED_PROGRESS_RATIO_BY_CARD.set(cardId, targetRatio);
      animationFrameRef.current = null;
    };

    animationFrameRef.current = window.requestAnimationFrame(tick);
    return () => {
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [animateDeedProgress, cardId, deedProgressRatio]);

  const displayedDeedProgressRatio = hasDeedProgress ? animatedDeedProgressRatio : deedProgressRatio;
  const deedProgressArcPath = buildDeedProgressArcPath(displayedDeedProgressRatio);
  const displayedCardImage = cardImageReady ? cardImage : CARD_BACK_IMAGE;

  const metadataRow = (
    <div className="card-row card-meta">
      <div className="card-meta-leading">
        <span className="card-rank">{rank}</span>
        <div className="card-suits-row">
          {suits.length > 0 ? (
            suits.map((suit) => <SuitIcon key={`${cardId}-${suit}`} suit={suit} className="card-suit-icon" />)
          ) : (
            <span className="card-suit-placeholder" />
          )}
        </div>
      </div>
      {hasDeedProgress ? (
        <div
          className="deed-progress"
          title="development progress"
          aria-label="development progress"
        >
          <svg className="deed-progress-ring" viewBox="0 0 36 36" aria-hidden="true">
            <circle
              className="deed-progress-ring-track"
              cx="18"
              cy="18"
              r={DEED_PROGRESS_RING_RADIUS}
            />
            {displayedDeedProgressRatio >= 1 ? (
              <circle className="deed-progress-ring-value" cx="18" cy="18" r={DEED_PROGRESS_RING_RADIUS} />
            ) : deedProgressArcPath ? (
              <path className="deed-progress-ring-value" d={deedProgressArcPath} />
            ) : null}
          </svg>
          <span className="deed-progress-value">
            {deedProgress}/{deedTarget}
          </span>
        </div>
      ) : (
        <span className="deed-progress-placeholder" aria-hidden="true" />
      )}
    </div>
  );

  const imageBody = (
    <div className="card-row card-body">
      <div className="card-image-frame" aria-hidden="true">
        <img className="card-image" src={displayedCardImage} alt="" />
      </div>
      {showDeedTokenRails ? (
        <>
          <div
            className="card-side-token-rail card-side-token-rail-left"
            data-deed-token-rail="left"
            aria-hidden="true"
          >
            {deedTokensBySide.left.map((entry) => (
              <TokenChip key={`left-${cardId}-${entry.suit}`} suit={entry.suit} count={entry.count} compact />
            ))}
          </div>
          <div
            className="card-side-token-rail card-side-token-rail-right"
            data-deed-token-rail="right"
            aria-hidden="true"
          >
            {deedTokensBySide.right.map((entry) => (
              <TokenChip key={`right-${cardId}-${entry.suit}`} suit={entry.suit} count={entry.count} compact />
            ))}
          </div>
        </>
      ) : null}
    </div>
  );

  return (
    <div
      className={`card-tile${perspective === 'bot' ? ' perspective-bot' : ''}${inDevelopment ? ' is-in-development' : ''}`}
      title={card.name}
      data-card-id={cardId}
      data-in-development={inDevelopment ? 'true' : undefined}
      data-hand-owner-id={handOwnerId}
      data-hand-card-id={handCardId}
      data-hand-slot-kind={handSlotKind}
    >
      {perspective === 'bot' ? imageBody : metadataRow}
      {perspective === 'bot' ? metadataRow : imageBody}
    </div>
  );
}
