import { readFileSync } from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { ACTION_IDS } from '../engine/actionSurface';
import { BRIDGE_COMMANDS, BRIDGE_ERROR_CODES } from './protocol';

interface ContractArtifact {
  contract_name: string;
  bridge: {
    commands: string[];
    error_codes: string[];
  };
  action_surface: {
    action_ids: string[];
  };
}

describe('magnate bridge contract artifact', () => {
  it('stays aligned with runtime command/action/error surfaces', () => {
    const contractPath = path.resolve(
      process.cwd(),
      'contracts',
      'magnate_bridge.v1.json'
    );
    const artifact = JSON.parse(
      readFileSync(contractPath, 'utf-8')
    ) as ContractArtifact;

    expect(artifact.contract_name).toBe('magnate_bridge');
    expect(artifact.bridge.commands).toEqual(BRIDGE_COMMANDS);
    expect(artifact.bridge.error_codes).toEqual(BRIDGE_ERROR_CODES);
    expect(artifact.action_surface.action_ids).toEqual(ACTION_IDS);
  });
});

