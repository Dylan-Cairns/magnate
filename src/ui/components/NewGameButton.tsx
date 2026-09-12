import type { RefObject } from 'react';

import type { Ruleset } from '../../engine/types';
import { profilesForRuleset, type BotProfileId } from '../../policies/catalog';
import { Tooltip } from './Tooltip';

const RULESET_OPTIONS: readonly { value: Ruleset; label: string }[] = [
  { value: 'standard', label: 'Standard' },
  { value: 'extended', label: 'Extended' },
];

export function NewGameButton({
  expanded,
  panelRef,
  buttonRef,
  seedInputRef,
  botProfileId,
  botStatusText,
  ruleset,
  animationsEnabled,
  onToggle,
  onStart,
  onBotProfileChange,
  onRulesetChange,
  onAnimationsEnabledChange,
}: {
  expanded: boolean;
  panelRef: RefObject<HTMLElement | null>;
  buttonRef: RefObject<HTMLButtonElement | null>;
  seedInputRef: RefObject<HTMLInputElement | null>;
  botProfileId: BotProfileId;
  botStatusText: string;
  ruleset: Ruleset;
  animationsEnabled: boolean;
  onToggle: () => void;
  onStart: () => void;
  onBotProfileChange: (id: BotProfileId) => void;
  onRulesetChange: (ruleset: Ruleset) => void;
  onAnimationsEnabledChange: (enabled: boolean) => void;
}) {
  return (
    <>
      <button
        ref={buttonRef}
        type="button"
        className="new-game-btn tooltip-trigger"
        aria-expanded={expanded}
        aria-controls="new-game-panel"
        onClick={onToggle}
      >
        New Game
        <Tooltip>
          {expanded ? 'Close new game setup' : 'Set up a new game'}
        </Tooltip>
      </button>
      {expanded && (
        <section
          id="new-game-panel"
          ref={panelRef as RefObject<HTMLElement>}
          className="new-game-panel"
          aria-label="New game options"
        >
          <div className="brand-controls">
            <input
              id="seed-input"
              aria-label="Seed"
              className="seed-input"
              ref={seedInputRef}
              autoComplete="off"
              defaultValue=""
              placeholder="seed (blank=random)"
            />
          </div>
          <div className="bot-profile-controls">
            <label htmlFor="ruleset-select">Ruleset</label>
            <select
              id="ruleset-select"
              className="ruleset-select"
              value={ruleset}
              onChange={(e) => onRulesetChange(e.target.value as Ruleset)}
            >
              {RULESET_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className="bot-profile-controls">
            <label htmlFor="bot-profile-select">Bot Profile</label>
            <select
              id="bot-profile-select"
              className="bot-profile-select"
              value={botProfileId}
              onChange={(e) =>
                onBotProfileChange(e.target.value as BotProfileId)
              }
            >
              {profilesForRuleset(ruleset).map((profile) => (
                <option key={profile.id} value={profile.id}>
                  {profile.label}
                </option>
              ))}
            </select>
            <p className="bot-profile-note">{botStatusText}</p>
          </div>
          <div className="bot-profile-controls animation-controls">
            <label className="animation-toggle-row" htmlFor="animations-toggle">
              <span>Animations</span>
              <input
                id="animations-toggle"
                type="checkbox"
                checked={animationsEnabled}
                onChange={(event) =>
                  onAnimationsEnabledChange(event.target.checked)
                }
              />
            </label>
          </div>
          <button
            type="button"
            className="reset-button new-game-start-button"
            onClick={onStart}
          >
            Start Game
          </button>
        </section>
      )}
    </>
  );
}
