import { ALL_CARDS, type CardId, type CardName } from '../engine/cards';

const CARD_IMAGE_MODULES = import.meta.glob('../assets/CardImages/*.png', {
  eager: true,
  import: 'default',
}) as Record<string, string>;

const CARD_IMAGE_FILE_BY_NAME = {
  'Ace of Knots': '1_ace_knots.png',
  'Ace of Leaves': '1_ace_leaves.png',
  'Ace of Moons': '1_ace_moons.png',
  'Ace of Suns': '1_ace_suns.png',
  'Ace of Waves': '1_ace_waves.png',
  'Ace of Wyrms': '1_ace_wyrms.png',
  'The Author': '2_author.png',
  'The Desert': '2_desert.png',
  'The Origin': '2_origin.png',
  'The Journey': '3_journey.png',
  'The Painter': '3_painter.png',
  'The Savage': '3_savage.png',
  'The Battle': '4_battle.png',
  'The Mountain': '4_mountain.png',
  'The Sailor': '4_sailor.png',
  'The Discovery': '5_discovery.png',
  'The Forest': '5_forest.png',
  'The Soldier': '5_soldier.png',
  'The Lunatic': '6_lunactic.png',
  'The Market': '6_market.png',
  'The Penitent': '6_penitent.png',
  'The Castle': '7_castle.png',
  'The Cave': '7_cave.png',
  'The Chance Meeting': '7_chance_meeting.png',
  'The Betrayal': '8_betrayal.png',
  'The Diplomat': '8_diplomat.png',
  'The Mill': '8_mill.png',
  'The Darkness': '9_darkness.png',
  'The Merchant': '9_merchant.png',
  'The Pact': '9_pact.png',
  'The Windfall': 'crown_knots.png',
  'The End': 'crown_leaves.png',
  'The Huntress': 'crown_moons.png',
  'The Bard': 'crown_suns.png',
  'The Sea': 'crown_waves.png',
  'The Calamity': 'crown_wyrms.png',
  'The Excuse': 'excuse.png',
  'The Borderland': 'pawn_borderland.png',
  'The Harvest': 'pawn_harvest.png',
  'The Light Keeper': 'pawn_light_keeper.png',
  'The Watchman': 'pawn_watchman.png',
} as const satisfies Record<CardName, string>;

export const CARD_BACK_IMAGE_FILE = 'back.png';

function resolveCardImageFile(fileName: string): string {
  const moduleKey = `../assets/CardImages/${fileName}`;
  const imageUrl = CARD_IMAGE_MODULES[moduleKey];
  if (!imageUrl) {
    throw new Error(`Missing card image asset: ${fileName}`);
  }
  return imageUrl;
}

function mapByCardId<T>(recordByName: Record<CardName, T>): Record<CardId, T> {
  const record = Object.create(null) as Record<CardId, T>;
  for (const card of ALL_CARDS) {
    record[card.id] = recordByName[card.name];
  }
  return record;
}

const CARD_IMAGE_BY_NAME = Object.freeze(
  Object.fromEntries(
    (Object.entries(CARD_IMAGE_FILE_BY_NAME) as Array<[CardName, string]>).map(
      ([cardName, fileName]) => [cardName, resolveCardImageFile(fileName)]
    )
  ) as Record<CardName, string>
);

export const CARD_IMAGE_FILE_BY_ID = Object.freeze(
  mapByCardId(CARD_IMAGE_FILE_BY_NAME)
);
export const CARD_IMAGE_BY_ID = Object.freeze(mapByCardId(CARD_IMAGE_BY_NAME));
export const CARD_BACK_IMAGE = resolveCardImageFile(CARD_BACK_IMAGE_FILE);
export const ALL_CARD_IMAGE_URLS = Object.freeze(
  Array.from(new Set([...Object.values(CARD_IMAGE_BY_ID), CARD_BACK_IMAGE]))
);
const PRELOADED_CARD_IMAGE_URLS = new Set<string>();
const PRELOADED_CARD_IMAGE_BY_URL = new Map<string, HTMLImageElement>();
const CARD_IMAGE_PRELOAD_PROMISE_BY_URL = new Map<string, Promise<void>>();

export function getCardImageFile(cardId: CardId): string {
  return CARD_IMAGE_FILE_BY_ID[cardId];
}

export function getCardImage(cardId: CardId): string {
  return CARD_IMAGE_BY_ID[cardId];
}

export function isCardImageUrlReady(url: string): boolean {
  return PRELOADED_CARD_IMAGE_URLS.has(url);
}

export function preloadCardImageUrl(url: string): Promise<void> {
  if (PRELOADED_CARD_IMAGE_URLS.has(url)) {
    return Promise.resolve();
  }

  const existing = CARD_IMAGE_PRELOAD_PROMISE_BY_URL.get(url);
  if (existing) {
    return existing;
  }

  if (typeof Image === 'undefined') {
    PRELOADED_CARD_IMAGE_URLS.add(url);
    return Promise.resolve();
  }

  const created = new Promise<void>((resolve, reject) => {
    const image = new Image();
    let settled = false;

    const cleanup = () => {
      image.onload = null;
      image.onerror = null;
    };
    const finish = () => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      PRELOADED_CARD_IMAGE_URLS.add(url);
      PRELOADED_CARD_IMAGE_BY_URL.set(url, image);
      resolve();
    };
    const fail = () => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(new Error(`Failed to preload card image: ${url}`));
    };

    image.onload = () => {
      if (typeof image.decode === 'function') {
        image.decode().catch(() => undefined).finally(finish);
        return;
      }
      finish();
    };
    image.onerror = fail;
    image.src = url;

    if (image.complete) {
      if (image.naturalWidth > 0 || image.naturalHeight > 0) {
        finish();
      } else {
        fail();
      }
    }
  });

  CARD_IMAGE_PRELOAD_PROMISE_BY_URL.set(url, created);
  created.catch(() => {
    if (CARD_IMAGE_PRELOAD_PROMISE_BY_URL.get(url) === created) {
      CARD_IMAGE_PRELOAD_PROMISE_BY_URL.delete(url);
    }
  });
  return created;
}
