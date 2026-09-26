# Magnate

A single-player, web-playable implementation of the Decktet game
[Magnate](https://decktet.wikidot.com/game:magnate). Rules live in a
deterministic TypeScript engine shared by the browser UI and a Python training
stack that improves the bot through gated self-play.

## Quickstart

Requires [fnm](https://github.com/Schniz/fnm) with shell integration. From the
repo root:

```sh
fnm install
fnm use
corepack enable
corepack install
yarn install
yarn dev
```

`yarn dev` prints the local URL. See
[docs/runbooks/windows-local.md](docs/runbooks/windows-local.md) for setup
troubleshooting and the optional Python environment.

## Where To Look Next

- **Web app and engine:**
  [memoryBank/systemPatterns.md](memoryBank/systemPatterns.md) (architecture),
  [memoryBank/magnateRules.md](memoryBank/magnateRules.md) (rules reference),
  [docs/AGENT_GUIDE.md](docs/AGENT_GUIDE.md) (reading map by surface).
- **Bot training and evaluation:**
  [docs/runbooks/training-loop.md](docs/runbooks/training-loop.md) (Python TD
  loops), [docs/runbooks/bot-eval.md](docs/runbooks/bot-eval.md) (browser-bot
  evaluation), [docs/runbooks/td-browser-benchmarks.md](docs/runbooks/td-browser-benchmarks.md)
  (TD browser inference), [memoryBank/techContext.md](memoryBank/techContext.md)
  (tooling and command index).
- **Agent workflow:** [AGENTS.md](AGENTS.md).

## Credits & License

Magnate is an unofficial, noncommercial fan implementation of the Decktet game
[Magnate](https://decktet.wikidot.com/game:magnate), designed by Cristyn Magnus
with additional development by P.D. Magnus. The Decktet is created by P.D.
Magnus; its card art and game material are used under a
[Creative Commons Attribution-NonCommercial-ShareAlike](https://creativecommons.org/licenses/by-nc-sa/4.0/)
license. Digital version by Dylan Cairns. The same credits appear in the in-app
info menu.

This project is licensed under CC BY-NC-SA 4.0; see [LICENSE](LICENSE). Bundled
third-party software notices are listed in
[public/third-party-notices.txt](public/third-party-notices.txt).
