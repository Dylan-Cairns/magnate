import { useEffect, useState, type ReactNode } from 'react';

import type { CardId } from '../../engine/cards';
import type {
  FinalScore,
  GameAction,
  PlayerId,
  Suit,
} from '../../engine/types';
import {
  actionCategoryForItem,
  actionCategoryLabel,
  buildDevelopOutrightGroupPresentation,
} from '../actionPanelModel';
import type { ActionPickerState } from '../actionPickerModel';
import {
  actionStableKey,
  cardSummary,
  describeAction,
  formatTokens,
  type HumanActionListItem,
} from '../actionPresentation';
import { SUIT_TEXT_TOKEN } from '../suitIcons';
import { SuitText } from './SuitText';
import { TerminalScoreSummary } from './TerminalScoreSummary';

export type DistrictPickerConfig = {
  actionType: 'buy-deed';
  cardId: CardId;
};

export type DeedPaymentPickerConfig = {
  cardId: CardId;
  districtId: string;
};

export type IncomeChoicePickerConfig = {
  playerId: PlayerId;
  cardId: CardId;
  districtId: string;
};

export function ActionsPanel({
  terminal,
  isLastTurn,
  score,
  wonDistrictsByPlayer,
  activePlayerId,
  humanPlayerId,
  botPlayerId,
  visibleActionItems,
  isIncomeChoicePhase,
  hasMultipleTradeSources,
  actionPicker,
  canResetTurn,
  botThinking,
  hideBotWaitMessageDuringTurnCycleLock,
  humanActionUiBlockedByAnimation,
  humanActionUiBlockedByTurnCycleAnimation,
  onAction,
  onResetTurn,
  onClosePicker,
  onOpenTradeCombinedPicker,
  onOpenTradePicker,
  onOpenDistrictPicker,
  onOpenDevelopOutrightCombinedPicker,
  onOpenDevelopOutrightDistrictOnlyPicker,
  onOpenDeedPaymentPicker,
  onOpenIncomeChoicePicker,
}: {
  terminal: boolean;
  isLastTurn: boolean;
  score: FinalScore;
  wonDistrictsByPlayer: Record<PlayerId, readonly string[]>;
  activePlayerId: PlayerId;
  humanPlayerId: PlayerId;
  botPlayerId: PlayerId;
  visibleActionItems: readonly HumanActionListItem[];
  isIncomeChoicePhase: boolean;
  hasMultipleTradeSources: boolean;
  actionPicker: ActionPickerState | null;
  canResetTurn: boolean;
  botThinking: boolean;
  hideBotWaitMessageDuringTurnCycleLock: boolean;
  humanActionUiBlockedByAnimation: boolean;
  humanActionUiBlockedByTurnCycleAnimation: boolean;
  onAction: (action: GameAction) => void;
  onResetTurn: () => void;
  onClosePicker: () => void;
  onOpenTradeCombinedPicker: (trigger: HTMLButtonElement) => void;
  onOpenTradePicker: (
    give: Suit,
    trigger: HTMLButtonElement,
    optionCount: number
  ) => void;
  onOpenDistrictPicker: (
    config: DistrictPickerConfig,
    trigger: HTMLButtonElement,
    optionCount: number
  ) => void;
  onOpenDevelopOutrightCombinedPicker: (
    cardId: CardId,
    trigger: HTMLButtonElement,
    optionCount: number
  ) => void;
  onOpenDevelopOutrightDistrictOnlyPicker: (
    cardId: CardId,
    trigger: HTMLButtonElement,
    optionCount: number
  ) => void;
  onOpenDeedPaymentPicker: (
    config: DeedPaymentPickerConfig,
    trigger: HTMLButtonElement,
    optionCount: number
  ) => void;
  onOpenIncomeChoicePicker: (
    config: IncomeChoicePickerConfig,
    trigger: HTMLButtonElement,
    optionCount: number
  ) => void;
}) {
  const hasVisibleIncomeChoiceActions = visibleActionItems.some(
    (item) =>
      (item.kind === 'action' && item.action.type === 'choose-income-suit') ||
      item.kind === 'income-choice-group'
  );

  return (
    <section className="panel actions-panel">
      <div className="actions-heading">
        <h2>{terminal ? 'Game Over' : 'Actions'}</h2>
        {isLastTurn && <span className="last-turn-badge">Last Turn</span>}
      </div>
      <div className="actions-body">
        {terminal ? (
          <TerminalScoreSummary
            score={score}
            wonDistrictsByPlayer={wonDistrictsByPlayer}
            humanPlayerId={humanPlayerId}
            botPlayerId={botPlayerId}
          />
        ) : activePlayerId === humanPlayerId ||
          hasVisibleIncomeChoiceActions ? (
          <div className="actions-human-layout">
            <div className="actions-human-main">
              {humanActionUiBlockedByTurnCycleAnimation ? (
                <p className="empty-note">Resolving income and taxation...</p>
              ) : humanActionUiBlockedByAnimation ? null : visibleActionItems.length ===
                0 ? (
                <p className="empty-note">
                  {isIncomeChoicePhase ? (
                    botThinking ? (
                      <BotThinkingText />
                    ) : (
                      'Resolving income choices...'
                    )
                  ) : (
                    'No legal actions.'
                  )}
                </p>
              ) : (
                <div className="action-list">
                  {visibleActionItems.map((item, index) => {
                    const categoryKey = actionCategoryForItem(item);
                    const previousCategoryKey =
                      index > 0
                        ? actionCategoryForItem(visibleActionItems[index - 1])
                        : null;
                    const showCategory = previousCategoryKey !== categoryKey;
                    const categoryLabel = actionCategoryLabel(categoryKey);
                    const renderCategorizedAction = (
                      key: string,
                      button: ReactNode
                    ) => (
                      <div
                        key={key}
                        className={`action-entry${showCategory ? ' has-category' : ''}`}
                      >
                        {showCategory ? (
                          <p className="action-category">{categoryLabel}</p>
                        ) : null}
                        {button}
                      </div>
                    );

                    const renderPickerGroupAction = (
                      key: string,
                      label: string,
                      isOpen: boolean,
                      onOpen: (trigger: HTMLButtonElement) => void
                    ) =>
                      renderCategorizedAction(
                        key,
                        <button
                          type="button"
                          className="action-button has-submenu"
                          onClick={(event) => {
                            if (isOpen) {
                              onClosePicker();
                              return;
                            }
                            onOpen(event.currentTarget);
                          }}
                        >
                          <span className="action-text">
                            <SuitText text={label} />
                          </span>
                        </button>
                      );

                    const renderDirectAction = (
                      key: string,
                      action: GameAction
                    ) =>
                      renderCategorizedAction(
                        key,
                        <ActionButton
                          text={describeAction(action, SUIT_TEXT_TOKEN)}
                          onClick={() => onAction(action)}
                        />
                      );

                    if (item.kind === 'trade-group') {
                      if (hasMultipleTradeSources) {
                        return renderCategorizedAction(
                          'trade-source-group',
                          <button
                            type="button"
                            className="action-button has-submenu"
                            onClick={(event) => {
                              if (actionPicker?.kind === 'trade-combined') {
                                onClosePicker();
                                return;
                              }
                              onOpenTradeCombinedPicker(event.currentTarget);
                            }}
                          >
                            <span className="action-text">Trade resources</span>
                          </button>
                        );
                      }

                      if (item.options.length === 1) {
                        const [onlyOption] = item.options;
                        return renderCategorizedAction(
                          `trade-direct-${item.give}`,
                          <ActionButton
                            text={describeAction(onlyOption, SUIT_TEXT_TOKEN)}
                            onClick={() => onAction(onlyOption)}
                          />
                        );
                      }

                      return renderCategorizedAction(
                        `trade-group-${item.give}`,
                        <button
                          type="button"
                          className="action-button has-submenu"
                          onClick={(event) => {
                            if (
                              actionPicker?.kind === 'trade' &&
                              actionPicker.give === item.give
                            ) {
                              onClosePicker();
                              return;
                            }
                            onOpenTradePicker(
                              item.give,
                              event.currentTarget,
                              item.options.length
                            );
                          }}
                        >
                          <span className="action-text">
                            <SuitText
                              text={`Trade ${SUIT_TEXT_TOKEN[item.give]}x3`}
                            />
                          </span>
                        </button>
                      );
                    }

                    if (item.kind === 'buy-deed-group') {
                      if (item.options.length === 1) {
                        const [onlyOption] = item.options;
                        return renderDirectAction(
                          `buy-deed-direct-${actionStableKey(onlyOption)}`,
                          onlyOption
                        );
                      }

                      return renderPickerGroupAction(
                        `buy-deed-group-${item.cardId}`,
                        `Buy deed ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)}`,
                        actionPicker?.kind === 'district' &&
                          actionPicker.actionType === 'buy-deed' &&
                          actionPicker.cardId === item.cardId,
                        (trigger) =>
                          onOpenDistrictPicker(
                            { actionType: 'buy-deed', cardId: item.cardId },
                            trigger,
                            item.options.length
                          )
                      );
                    }

                    if (item.kind === 'develop-deed-group') {
                      if (item.options.length === 1) {
                        const [onlyOption] = item.options;
                        return renderDirectAction(
                          `develop-deed-direct-${actionStableKey(onlyOption)}`,
                          onlyOption
                        );
                      }

                      return renderPickerGroupAction(
                        `develop-deed-group-${item.cardId}-${item.districtId}`,
                        `Develop deed ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)} in ${item.districtId}`,
                        actionPicker?.kind === 'deed-payment' &&
                          actionPicker.cardId === item.cardId &&
                          actionPicker.districtId === item.districtId,
                        (trigger) =>
                          onOpenDeedPaymentPicker(
                            {
                              cardId: item.cardId,
                              districtId: item.districtId,
                            },
                            trigger,
                            item.options.length
                          )
                      );
                    }

                    if (item.kind === 'develop-outright-group') {
                      if (item.options.length === 1) {
                        const [onlyOption] = item.options;
                        return renderCategorizedAction(
                          `develop-outright-direct-${actionStableKey(onlyOption)}`,
                          <ActionButton
                            text={describeAction(onlyOption, SUIT_TEXT_TOKEN)}
                            onClick={() => onAction(onlyOption)}
                          />
                        );
                      }

                      const presentation =
                        buildDevelopOutrightGroupPresentation(item.options);

                      return renderCategorizedAction(
                        `develop-outright-group-${item.cardId}`,
                        <button
                          type="button"
                          className="action-button has-submenu"
                          onClick={(event) => {
                            if (
                              (actionPicker?.kind ===
                                'develop-outright-combined' ||
                                actionPicker?.kind ===
                                  'develop-outright-district') &&
                              actionPicker.cardId === item.cardId
                            ) {
                              onClosePicker();
                              return;
                            }
                            if (presentation.hasSinglePaymentPattern) {
                              onOpenDevelopOutrightDistrictOnlyPicker(
                                item.cardId,
                                event.currentTarget,
                                presentation.districtCount
                              );
                              return;
                            }
                            onOpenDevelopOutrightCombinedPicker(
                              item.cardId,
                              event.currentTarget,
                              item.options.length
                            );
                          }}
                        >
                          <span className="action-text">
                            <SuitText
                              text={
                                presentation.hasSinglePaymentPattern &&
                                presentation.firstPayment
                                  ? `Develop ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)} (${formatTokens(presentation.firstPayment, SUIT_TEXT_TOKEN)})`
                                  : `Develop ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)}`
                              }
                            />
                          </span>
                        </button>
                      );
                    }

                    if (item.kind === 'income-choice-group') {
                      return renderPickerGroupAction(
                        `income-choice-group-${item.playerId}-${item.districtId}-${item.cardId}`,
                        `Choose income ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)} in ${item.districtId}`,
                        actionPicker?.kind === 'income-choice' &&
                          actionPicker.playerId === item.playerId &&
                          actionPicker.cardId === item.cardId &&
                          actionPicker.districtId === item.districtId,
                        (trigger) =>
                          onOpenIncomeChoicePicker(
                            {
                              playerId: item.playerId,
                              cardId: item.cardId,
                              districtId: item.districtId,
                            },
                            trigger,
                            item.options.length
                          )
                      );
                    }

                    return renderCategorizedAction(
                      actionStableKey(item.action),
                      <ActionButton
                        text={describeAction(item.action, SUIT_TEXT_TOKEN)}
                        onClick={() => onAction(item.action)}
                      />
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
                  onClick={onResetTurn}
                >
                  <span className="action-text">Reset turn</span>
                </button>
              </div>
            ) : null}
          </div>
        ) : hideBotWaitMessageDuringTurnCycleLock ? null : (
          <p className="empty-note">
            {isIncomeChoicePhase && !botThinking ? (
              'Resolving income choices...'
            ) : (
              <BotThinkingText />
            )}
          </p>
        )}
      </div>
    </section>
  );
}

const BOT_THINKING_DOT_INTERVAL_MS = 700;

function BotThinkingText() {
  const [dotCount, setDotCount] = useState<number>(1);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setDotCount((current) => (current === 3 ? 1 : current + 1));
    }, BOT_THINKING_DOT_INTERVAL_MS);

    return () => window.clearInterval(intervalId);
  }, []);

  return (
    <span className="bot-thinking-text" aria-label="Bot is thinking...">
      <span aria-hidden="true">
        Bot is thinking
        <span className="bot-thinking-dots">
          {Array.from({ length: 3 }, (_, index) => (
            <span
              key={index}
              className={
                index < dotCount
                  ? 'bot-thinking-dot is-visible'
                  : 'bot-thinking-dot'
              }
            >
              .
            </span>
          ))}
        </span>
      </span>
    </span>
  );
}

function ActionButton({
  text,
  onClick,
}: {
  text: string;
  onClick: () => void;
}) {
  return (
    <button type="button" className="action-button" onClick={onClick}>
      <span className="action-text">
        <SuitText text={text} />
      </span>
    </button>
  );
}
