import { ALL_CARDS, type CardId, type CardName } from '../engine/cards';

const CARD_IMAGE_MODULES = import.meta.glob(
  '../assets/decktet-card-art/*.{webp,svg}',
  {
    eager: true,
    import: 'default',
  }
) as Record<string, string>;

const FILE_NAME_PREFIX = 'decktet-card-';

// Card art is raster WebP for the illustrated cards. The Darkness is the one
// exception: its canonical Decktet art is blank, so it uses an adapted pale
// vector illustration (SVG) instead. Vector is preferred when both exist.
const CARD_IMAGE_EXTENSIONS = ['svg', 'webp'] as const;

function cardImageSlug(cardName: string): string {
  const slug = cardName.toLowerCase().replace(/ /g, '-');
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
    throw new Error(`Unsupported card name for asset filename: ${cardName}`);
  }
  return slug;
}

function cardImageFileCandidates(cardName: string): readonly string[] {
  const slug = cardImageSlug(cardName);
  return CARD_IMAGE_EXTENSIONS.map(
    (extension) => `${FILE_NAME_PREFIX}${slug}.${extension}`
  );
}

function moduleKey(fileName: string): string {
  return `../assets/decktet-card-art/${fileName}`;
}

function resolveCardImageUrl(fileName: string): string {
  const imageUrl = CARD_IMAGE_MODULES[moduleKey(fileName)];
  if (!imageUrl) {
    throw new Error(`Missing card image asset: ${fileName}`);
  }
  return imageUrl;
}

function fileNameFromModuleKey(key: string): string {
  return key.slice(key.lastIndexOf('/') + 1);
}

// Resolves the single art file for a card, preferring vector over raster.
export function cardImageFileName(cardName: string): string {
  const candidates = cardImageFileCandidates(cardName);
  const matches = candidates.filter(
    (fileName) => CARD_IMAGE_MODULES[moduleKey(fileName)] !== undefined
  );
  if (matches.length === 0) {
    throw new Error(`Missing card image asset: ${candidates[0]}`);
  }
  if (matches.length > 1) {
    throw new Error(
      `Multiple card image assets for ${cardName}: ${matches.join(', ')}`
    );
  }
  return matches[0];
}

const CANDIDATE_FILE_NAMES = new Set(
  ALL_CARDS.flatMap((card) => cardImageFileCandidates(card.name))
);

if (CANDIDATE_FILE_NAMES.size !== ALL_CARDS.length * CARD_IMAGE_EXTENSIONS.length) {
  throw new Error('Duplicate card image filenames derived from card names.');
}

const AVAILABLE_FILE_NAMES = new Set(
  Object.keys(CARD_IMAGE_MODULES).map(fileNameFromModuleKey)
);

for (const fileName of AVAILABLE_FILE_NAMES) {
  if (!CANDIDATE_FILE_NAMES.has(fileName)) {
    throw new Error(`Unexpected card image asset: ${fileName}`);
  }
}

export const CARD_IMAGE_FILE_BY_ID = Object.freeze(
  Object.fromEntries(
    ALL_CARDS.map((card) => [card.id, cardImageFileName(card.name)])
  ) as Record<CardId, string>
) as Readonly<Record<CardId, string>>;

const CARD_IMAGE_BY_NAME = Object.fromEntries(
  ALL_CARDS.map((card) => [
    card.name,
    resolveCardImageUrl(cardImageFileName(card.name)),
  ])
) as Record<CardName, string>;

export const CARD_IMAGE_BY_ID = Object.freeze(
  Object.fromEntries(
    ALL_CARDS.map((card) => [card.id, CARD_IMAGE_BY_NAME[card.name]])
  ) as Record<CardId, string>
) as Readonly<Record<CardId, string>>;

export const ALL_CARD_IMAGE_URLS = Object.freeze(
  Array.from(new Set(Object.values(CARD_IMAGE_BY_ID)))
);

const PRELOADED_CARD_IMAGE_URLS = new Set<string>();
const PRELOADED_CARD_IMAGE_BY_URL = new Map<string, HTMLImageElement>();
const CARD_IMAGE_PRELOAD_PROMISE_BY_URL = new Map<string, Promise<void>>();
const REPORTED_IMAGE_RENDER_FAILURES = new Set<string>();

export function getCardImageFile(cardId: CardId): string {
  return CARD_IMAGE_FILE_BY_ID[cardId];
}

export function getCardImage(cardId: CardId): string {
  return CARD_IMAGE_BY_ID[cardId];
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
        image.decode().then(finish).catch(fail);
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

export function reportImageRenderFailure(url: string, label: string): void {
  const key = `${label}:${url}`;
  if (REPORTED_IMAGE_RENDER_FAILURES.has(key)) {
    return;
  }
  REPORTED_IMAGE_RENDER_FAILURES.add(key);
  console.error(`[Magnate asset] Failed to render ${label}: ${url}`);
}
