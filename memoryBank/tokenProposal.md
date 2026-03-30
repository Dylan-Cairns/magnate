# Token Value Proposal

## Purpose

Heuristic v2 currently understands two major action benefits:

- scoring potential: how a move changes district outcomes
- earning potential: how a move improves future resource generation

The missing concept is the value of loose resources in the player's bank. A token should not be treated as generically useful or useless. Its value depends on what that suit can still accomplish in the current game, and how hard that suit is for the player to replace.

The proposed model is a contextual suit-token value:

```text
token value for a suit =
  earning demand for that suit
+ scoring demand for that suit
adjusted by player access / replaceability
```

This gives the bot a principled way to decide whether spending, gaining, trading, or preserving a token is strategically meaningful.

## Concept

Each suit has a current value for each player. That value is not fixed across the game.

A suit becomes more valuable when:

- important visible or information-safe future cards use that suit
- current incomplete deeds need that suit for completion
- those demand sources have strong future earning potential
- those demand sources could make meaningful district-scoring gains
- the player has weak access to that suit through crowns or board income

A suit becomes less valuable when:

- its strongest cards have already been played or are no longer realistically useful
- the game is late enough that new income engines have little time to pay off
- the player already has reliable access to that suit
- extra copies of that suit in the bank have diminishing marginal value
- extra copies above one are exposed to taxation

Three or more copies of a suit also have trade liquidity because they can be converted into another suit. That liquidity is real, but it should be valued separately from direct demand for the suit itself.

This avoids brittle rules like "do not spend the last token of a suit." Spending the last Sun can be acceptable if Suns are easy to replace and remaining Sun demand is low. Spending the last Wyrm can be costly if Wyrms are hard to replace and important Wyrm cards remain.

## Earning Demand

The earning side asks:

```text
What future income infrastructure can this suit still help buy or build?
```

For each suit, look at information-safe demand sources that include that suit. Current incomplete deeds and cards in the player's hand should be weighted most heavily because they are actionable. Publicly known remaining opportunities and unknown-pool expectations can contribute at a lower weight because they are potentialities, not current plans. The model must not inspect hidden opponent hand cards or exact hidden draw order at the root.

Each card contributes based on its income potential, mostly from its rank and expected roll frequency. High-rank cards usually contribute more because they generate on more likely income rolls and can become stronger income infrastructure.

This earning value should decline as the game progresses. A high-rank card still has static income quality, but it has fewer future turns to produce resources. Late in the game, that same card may remain valuable for scoring, while its earning component should fade.

This is related to existing heuristic v2 earning potential, but in the reverse direction:

```text
card earning potential: this card can produce future tokens
token earning value: this token can help unlock future producing cards
```

The same underlying suit-access idea should inform both.

## Scoring Demand

The scoring side asks:

```text
What district-scoring opportunities can cards of this suit still support?
```

For each information-safe demand source that uses a suit, estimate how much scoring value it could create in the current board position. Current incomplete deeds should count strongly because loose tokens can convert them into completed properties. Cards in hand should count next because they can be bought, sold, or developed now. Publicly known remaining opportunities and unknown-pool expectations should count more weakly. A card has higher scoring demand if it could flip, protect, or materially improve a district. It has lower scoring demand if it would only add redundant points to a district that is already secure or irrelevant.

This makes token value board-sensitive. A rank 4 card can be low value in one game state and high value in another, depending on district margins and legal placement paths.

## Access And Replaceability

Raw suit demand should be adjusted by how reliably the player can regain that suit.

Relevant access sources include:

- crown suits
- developed cards that generate the suit
- incomplete deeds that may become future generators
- existing resource-engine strength and game phase

If the player has strong access to a suit, loose tokens of that suit are cheaper to spend. If the player lacks access, each token is more precious because replacing it may require selling, trading, or waiting for unlikely income.

This overlaps intentionally with earning potential. The shared concept is:

```text
suit access = how reliably this player can produce a suit
```

Earning potential uses suit access to value new generators. Token value uses suit access to value loose resources.

## Marginal Bank Value

Token value should be concave within each suit. A flat value per token would overvalue hoarding, especially because taxation only protects the first loose token of each suit.

A simple shape:

```text
1st token of a suit: full contextual suit value
2nd token: reduced value
3rd token: further reduced value, with possible trade liquidity
4th+ token: low surplus value
```

The exact multipliers should be calibrated, but the important property is:

```text
value(token 1) > value(token 2) > value(token 3) > value(token 4+)
```

This captures two game realities:

- the first token of a suit is safe from taxation and can preserve access to that suit
- surplus tokens are useful but fragile because taxation discards that suit down to one

For action scoring, this should be a local marginal curve, not a long-horizon tax projection. Each rollout decision re-scores the current state after the previous action, so tax risk does not need to be pre-simulated over many future turns. The current bank can simply treat surplus tokens as discounted because they are tax-exposed right now.

Trade liquidity should be an additional local adjustment. If a player has at least three tokens of a suit, that pile can become one token of a missing high-demand suit. This should add some value to the third token and beyond, but not enough to make large piles look as good as naturally holding the demanded suit.

## Move Scoring

For a candidate move, compare the player's resource-bank value before and after the move:

```text
token delta =
  value of resource bank after the move
- value of resource bank before the move
```

That delta becomes an additional heuristic v2 term alongside scoring and earning deltas.

The token delta should represent discounted option value, not fully realized board value. A token that could support a strong future card is valuable, but the unfinished possibility should be worth less than actually building the card. When a move spends a scarce token to create real scoring or earning value, the scoring and earning deltas should carry the main reward, while the token delta charges only the lost liquidity and remaining opportunity cost.

Examples:

- Selling a low-impact Wyrms/Knots card can be good if it gains scarce Wyrms/Knots tokens while stronger Wyrms/Knots opportunities remain.
- Buying a mediocre Wyrms/Knots deed can be bad if it spends rare tokens needed for more important future cards.
- Buying a modest Suns/Leaves deed can be acceptable if Suns and Leaves are easy for the player to replace and the move adds some scoring value.
- Spending the only copy of a scarce suit can be costly; spending from four copies down to three should usually be much cheaper because both states still have surplus tax-exposed tokens.
- Trading three low-demand surplus tokens into one high-demand missing suit can be good even though total token count decreases.

The goal is not to reward hoarding. Tokens matter because they preserve or create useful future options.

## Leaf-State Evaluation

The same concept should improve rollout leaf evaluation.

A non-terminal leaf state should not only ask:

```text
Who has more resources?
Who has broader suit coverage?
```

It should also ask:

```text
Whose resource bank matches the remaining earning and scoring demands of the game?
```

This helps rollout v2 evaluate unfinished simulations more accurately. A player with fewer tokens may still have the better bank if those tokens match scarce, high-value remaining suits. A player with many tokens may have a weaker bank if those tokens are easy to replace or no longer support meaningful cards.

Terminal scoring remains rules-based. Token value is mainly for action scoring and non-terminal leaf evaluation, where strategic resource quality is otherwise under-modeled.

## Relevance

This model targets a real weakness in rollout v2. The bot currently values moves that improve scoring or income infrastructure, but it weakly understands the opportunity cost of the resources spent to make those moves.

Human players naturally see that some tokens are precious in a given position and others are expendable. A contextual suit-token value gives the bot a clean mathematical version of that judgment without one-off tactical patches.
