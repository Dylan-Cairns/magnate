import type { Dispatch, RefObject, SetStateAction } from 'react';

import type { GameAction } from '../../engine/types';
import {
  actionPickerTitle,
  buildDevelopOutrightCompositeOptions,
  resolveDevelopOutrightCompositeAction,
  resolveTradeCompositeAction,
  toPickerQuery,
  tradeActionsForPicker,
  tradeReceiveOptions,
  type ActionPickerState,
} from '../actionPickerModel';
import {
  buildPickerOptions,
  formatTokens,
  type TradeSourceGroup,
} from '../actionPresentation';
import { SUIT_TEXT_TOKEN } from '../suitIcons';
import { SuitText } from './SuitText';

export function ActionPicker({
  picker,
  pickerRef,
  legalActions,
  tradeSourceGroups,
  onPickerChange,
  onSelectAction,
  onClose,
}: {
  picker: ActionPickerState;
  pickerRef: RefObject<HTMLElement | null>;
  legalActions: readonly GameAction[];
  tradeSourceGroups: readonly TradeSourceGroup[];
  onPickerChange: Dispatch<SetStateAction<ActionPickerState | null>>;
  onSelectAction: (action: GameAction) => void;
  onClose: () => void;
}) {
  const title = actionPickerTitle(picker, SUIT_TEXT_TOKEN);

  return (
    <section
      ref={pickerRef}
      className="panel trade-popover"
      role="dialog"
      aria-label="Choose follow-up action option"
      style={{
        top: `${picker.top}px`,
        left: `${picker.left}px`,
      }}
    >
      <h2>
        <SuitText text={title} />
      </h2>

      {picker.kind === 'trade-combined' ? (
        <TradeCombinedPicker
          picker={picker}
          legalActions={legalActions}
          tradeSourceGroups={tradeSourceGroups}
          onPickerChange={onPickerChange}
          onSelectAction={onSelectAction}
        />
      ) : picker.kind === 'develop-outright-combined' ? (
        <DevelopOutrightCombinedPicker
          picker={picker}
          legalActions={legalActions}
          onPickerChange={onPickerChange}
          onSelectAction={onSelectAction}
        />
      ) : (
        <StandardPicker
          picker={picker}
          legalActions={legalActions}
          onSelectAction={onSelectAction}
        />
      )}

      <button type="button" className="trade-cancel-button" onClick={onClose}>
        Cancel
      </button>
    </section>
  );
}

function TradeCombinedPicker({
  picker,
  legalActions,
  tradeSourceGroups,
  onPickerChange,
  onSelectAction,
}: {
  picker: Extract<ActionPickerState, { kind: 'trade-combined' }>;
  legalActions: readonly GameAction[];
  tradeSourceGroups: readonly TradeSourceGroup[];
  onPickerChange: Dispatch<SetStateAction<ActionPickerState | null>>;
  onSelectAction: (action: GameAction) => void;
}) {
  const tradeActions = tradeActionsForPicker(legalActions);
  const receiveOptions = tradeReceiveOptions(tradeActions);

  return (
    <>
      <div className="composite-picker-group">
        <p className="composite-picker-label">Give x3</p>
        <div className="trade-choice-list">
          {tradeSourceGroups.map((group) => (
            <button
              key={`trade-combined-source-${group.give}`}
              type="button"
              className={`trade-choice-button${picker.selectedGive === group.give ? ' is-selected' : ''}`}
              onClick={() => {
                const nextGive = group.give;
                if (picker.selectedReceive) {
                  const selectedAction = resolveTradeCompositeAction(
                    tradeActions,
                    {
                      selectedGive: nextGive,
                      selectedReceive: picker.selectedReceive,
                    }
                  );
                  if (selectedAction) {
                    onSelectAction(selectedAction);
                    return;
                  }
                }
                onPickerChange((current) => {
                  if (!current || current.kind !== 'trade-combined') {
                    return current;
                  }
                  return { ...current, selectedGive: nextGive };
                });
              }}
            >
              <SuitText text={`${SUIT_TEXT_TOKEN[group.give]} x3`} />
            </button>
          ))}
        </div>
      </div>

      <div className="composite-picker-group">
        <p className="composite-picker-label">Receive x1</p>
        <div className="trade-choice-list">
          {receiveOptions.map((receiveSuit) => (
            <button
              key={`trade-combined-receive-${receiveSuit}`}
              type="button"
              className={`trade-choice-button${picker.selectedReceive === receiveSuit ? ' is-selected' : ''}`}
              onClick={() => {
                const nextReceive = receiveSuit;
                if (picker.selectedGive) {
                  const selectedAction = resolveTradeCompositeAction(
                    tradeActions,
                    {
                      selectedGive: picker.selectedGive,
                      selectedReceive: nextReceive,
                    }
                  );
                  if (selectedAction) {
                    onSelectAction(selectedAction);
                    return;
                  }
                }
                onPickerChange((current) => {
                  if (!current || current.kind !== 'trade-combined') {
                    return current;
                  }
                  return { ...current, selectedReceive: nextReceive };
                });
              }}
            >
              <SuitText text={`${SUIT_TEXT_TOKEN[receiveSuit]} x1`} />
            </button>
          ))}
        </div>
      </div>
    </>
  );
}

function DevelopOutrightCombinedPicker({
  picker,
  legalActions,
  onPickerChange,
  onSelectAction,
}: {
  picker: Extract<ActionPickerState, { kind: 'develop-outright-combined' }>;
  legalActions: readonly GameAction[];
  onPickerChange: Dispatch<SetStateAction<ActionPickerState | null>>;
  onSelectAction: (action: GameAction) => void;
}) {
  const { outrightOptions, districtOptions, paymentOptions } =
    buildDevelopOutrightCompositeOptions(legalActions, picker.cardId);

  return (
    <>
      <div className="composite-picker-group">
        <p className="composite-picker-label">District</p>
        <div className="trade-choice-list">
          {districtOptions.map((option) => (
            <button
              key={`develop-outright-district-${option.districtId}`}
              type="button"
              className={`trade-choice-button${picker.selectedDistrictId === option.districtId ? ' is-selected' : ''}`}
              onClick={() => {
                const nextDistrictId = option.districtId;
                if (picker.selectedPaymentKey) {
                  const selectedAction = resolveDevelopOutrightCompositeAction(
                    outrightOptions,
                    {
                      cardId: picker.cardId,
                      selectedDistrictId: nextDistrictId,
                      selectedPaymentKey: picker.selectedPaymentKey,
                    }
                  );
                  if (selectedAction) {
                    onSelectAction(selectedAction);
                    return;
                  }
                }
                onPickerChange((current) => {
                  if (
                    !current ||
                    current.kind !== 'develop-outright-combined'
                  ) {
                    return current;
                  }
                  return {
                    ...current,
                    selectedDistrictId: nextDistrictId,
                  };
                });
              }}
            >
              {option.districtId}
            </button>
          ))}
        </div>
      </div>

      <div className="composite-picker-group">
        <p className="composite-picker-label">Payment</p>
        <div className="trade-choice-list single-column">
          {paymentOptions.map(([paymentKey, option]) => (
            <button
              key={`develop-outright-payment-${paymentKey}`}
              type="button"
              className={`trade-choice-button${picker.selectedPaymentKey === paymentKey ? ' is-selected' : ''}`}
              onClick={() => {
                const nextPaymentKey = paymentKey;
                if (picker.selectedDistrictId) {
                  const selectedAction = resolveDevelopOutrightCompositeAction(
                    outrightOptions,
                    {
                      cardId: picker.cardId,
                      selectedDistrictId: picker.selectedDistrictId,
                      selectedPaymentKey: nextPaymentKey,
                    }
                  );
                  if (selectedAction) {
                    onSelectAction(selectedAction);
                    return;
                  }
                }
                onPickerChange((current) => {
                  if (
                    !current ||
                    current.kind !== 'develop-outright-combined'
                  ) {
                    return current;
                  }
                  return {
                    ...current,
                    selectedPaymentKey: nextPaymentKey,
                  };
                });
              }}
            >
              <SuitText text={formatTokens(option.payment, SUIT_TEXT_TOKEN)} />
            </button>
          ))}
        </div>
      </div>
    </>
  );
}

function StandardPicker({
  picker,
  legalActions,
  onSelectAction,
}: {
  picker: Exclude<
    ActionPickerState,
    { kind: 'trade-combined' } | { kind: 'develop-outright-combined' }
  >;
  legalActions: readonly GameAction[];
  onSelectAction: (action: GameAction) => void;
}) {
  const options = buildPickerOptions(
    toPickerQuery(picker),
    legalActions,
    SUIT_TEXT_TOKEN
  );

  return options.length === 0 ? (
    <p className="empty-note">No options available.</p>
  ) : (
    <div className="trade-choice-list">
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          className="trade-choice-button"
          onClick={() => onSelectAction(option.action)}
        >
          <SuitText text={option.label} />
        </button>
      ))}
    </div>
  );
}
