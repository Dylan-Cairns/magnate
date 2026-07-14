export const RESOURCE_FLIGHT_DURATION_MS = 280;
export const RESOURCE_FLIGHT_STAGGER_MS = 75;
export const CARD_FLIGHT_DURATION_MS = 280;
export const ACTION_FLIGHT_COMMIT_BUFFER_MS = 20;

// Dice animation settle durations — must match the CSS transition durations.
export const DICE_D10_SETTLE_MS = 1000;
export const DICE_D6_SETTLE_MS = 1000;
export const DICE_TAX_SETTLE_MS = DICE_D10_SETTLE_MS + DICE_D6_SETTLE_MS;

// Turn-cycle pre-flight timing knobs.
export const TURN_CYCLE_TAX_PRE_FLIGHT_MS = 550;
export const TURN_CYCLE_INCOME_PRE_FLIGHT_MS = 400;
export const TURN_CYCLE_STAGE_GAP_MS = 220;
export const TURN_CYCLE_TAX_FLIGHT_DURATION_MS = 900;
export const TURN_CYCLE_TAX_FLIGHT_STAGGER_MS = 500;
export const TURN_CYCLE_INCOME_FLIGHT_DURATION_MS = 560;
export const TURN_CYCLE_INCOME_FLIGHT_STAGGER_MS = 95;
export const TURN_CYCLE_POST_INCOME_HOLD_MS = 220;
export const DEED_PROGRESS_REVEAL_MS = 420;
