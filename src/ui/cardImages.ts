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
  'the AUTHOR': '2_author.png',
  'the DESERT': '2_desert.png',
  'the ORIGIN': '2_origin.png',
  'the JOURNEY': '3_journey.png',
  'the PAINTER': '3_painter.png',
  'the SAVAGE': '3_savage.png',
  'the BATTLE': '4_battle.png',
  'the MOUNTAIN': '4_mountain.png',
  'the SAILOR': '4_sailor.png',
  'the DISCOVERY': '5_discovery.png',
  'the FOREST': '5_forest.png',
  'the SOLDIER': '5_soldier.png',
  'the LUNATIC': '6_lunactic.png',
  'the MARKET': '6_market.png',
  'the PENITENT': '6_penitent.png',
  'the CASTLE': '7_castle.png',
  'the CAVE': '7_cave.png',
  'the CHANCE MEETING': '7_chance_meeting.png',
  'the BETRAYAL': '8_betrayal.png',
  'the DIPLOMAT': '8_diplomat.png',
  'the MILL': '8_mill.png',
  'the DARKNESS': '9_darkness.png',
  'the MERCHANT': '9_merchant.png',
  'the PACT': '9_pact.png',
  'the WINDFALL': 'crown_knots.png',
  'the END': 'crown_leaves.png',
  'the HUNTRESS': 'crown_moons.png',
  'the BARD': 'crown_suns.png',
  'the SEA': 'crown_waves.png',
  'the CALAMITY': 'crown_wyrms.png',
  'the EXCUSE': 'excuse.png',
  'the BORDERLAND': 'pawn_borderland.png',
  'the HARVEST': 'pawn_harvest.png',
  'the LIGHT KEEPER': 'pawn_light_keeper.png',
  'the WATCHMAN': 'pawn_watchman.png',
} as const satisfies Record<CardName, string>;

export const CARD_BACK_IMAGE_FILE = 'back.png';

function resolveCardImageFile(fileName: string): string {
  const moduleKey = `../assets/icons/CardImages/${fileName}`;
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

export function getCardImageFile(cardId: CardId): string {
  return CARD_IMAGE_FILE_BY_ID[cardId];
}

export function getCardImage(cardId: CardId): string {
  return CARD_IMAGE_BY_ID[cardId];
}
