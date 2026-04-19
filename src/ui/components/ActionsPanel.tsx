import type { ReactNode } from 'react';

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
  showBotThinkingDuringIncomeChoiceLock,
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
  showBotThinkingDuringIncomeChoiceLock: boolean;
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
      (item.kind === 'action' &&
        item.action.type === 'choose-income-suit') ||
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
        ) : activePlayerId === humanPlayerId || hasVisibleIncomeChoiceActions ? (
          <div className="actions-human-layout">
            <div className="actions-human-main">
              {humanActionUiBlockedByTurnCycleAnimation ? (
                <p className="empty-note">Resolving income and taxation...</p>
              ) : humanActionUiBlockedByAnimation ? null : visibleActionItems.length ===
                0 ? (
                <p className="empty-note">
                  {isIncomeChoicePhase
                    ? botThinking
                      ? 'Bot is thinking...'
                      : 'Resolving income choices...'
                    : 'No legal actions.'}
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
                        return renderCategorizedAction(
                          `buy-deed-direct-${actionStableKey(onlyOption)}`,
                          <ActionButton
                            text={describeAction(onlyOption, SUIT_TEXT_TOKEN)}
                            onClick={() => onAction(onlyOption)}
                          />
                        );
                      }

                      return renderCategorizedAction(
                        `buy-deed-group-${item.cardId}`,
                        <button
                          type="button"
                          className="action-button has-submenu"
                          onClick={(event) => {
                            if (
                              actionPicker?.kind === 'district' &&
                              actionPicker.actionType === 'buy-deed' &&
                              actionPicker.cardId === item.cardId
                            ) {
                              onClosePicker();
                              return;
                            }
                            onOpenDistrictPicker(
                              {
                                actionType: 'buy-deed',
                                cardId: item.cardId,
                              },
                              event.currentTarget,
                              item.options.length
                            );
                          }}
                        >
                          <span className="action-text">
                            <SuitText
                              text={`Buy deed ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)}`}
                            />
                          </span>
                        </button>
                      );
                    }

                    if (item.kind === 'develop-deed-group') {
                      if (item.options.length === 1) {
                        const [onlyOption] = item.options;
                        return renderCategorizedAction(
                          `develop-deed-direct-${actionStableKey(onlyOption)}`,
                          <ActionButton
                            text={describeAction(onlyOption, SUIT_TEXT_TOKEN)}
                            onClick={() => onAction(onlyOption)}
                          />
                        );
                      }

                      return renderCategorizedAction(
                        `develop-deed-group-${item.cardId}-${item.districtId}`,
                        <button
                          type="button"
                          className="action-button has-submenu"
                          onClick={(event) => {
                            if (
                              actionPicker?.kind === 'deed-payment' &&
                              actionPicker.cardId === item.cardId &&
                              actionPicker.districtId === item.districtId
                            ) {
                              onClosePicker();
                              return;
                            }
                            onOpenDeedPaymentPicker(
                              {
                                cardId: item.cardId,
                                districtId: item.districtId,
                              },
                              event.currentTarget,
                              item.options.length
                            );
                          }}
                        >
                          <span className="action-text">
                            <SuitText
                              text={`Develop deed ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)} in ${item.districtId}`}
                            />
                          </span>
                        </button>
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
                      return renderCategorizedAction(
                        `income-choice-group-${item.playerId}-${item.districtId}-${item.cardId}`,
                        <button
                          type="button"
                          className="action-button has-submenu"
                          onClick={(event) => {
                            if (
                              actionPicker?.kind === 'income-choice' &&
                              actionPicker.playerId === item.playerId &&
                              actionPicker.cardId === item.cardId &&
                              actionPicker.districtId === item.districtId
                            ) {
                              onClosePicker();
                              return;
                            }
                            onOpenIncomeChoicePicker(
                              {
                                playerId: item.playerId,
                                cardId: item.cardId,
                                districtId: item.districtId,
                              },
                              event.currentTarget,
                              item.options.length
                            );
                          }}
                        >
                          <span className="action-text">
                            <SuitText
                              text={`Choose income ${cardSummary(item.cardId, SUIT_TEXT_TOKEN)} in ${item.districtId}`}
                            />
                          </span>
                        </button>
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
            {isIncomeChoicePhase && !botThinking
              ? 'Resolving income choices...'
              : showBotThinkingDuringIncomeChoiceLock || botThinking
              ? 'Bot is thinking...'
              : 'Waiting for bot...'}
          </p>
        )}
      </div>
    </section>
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
