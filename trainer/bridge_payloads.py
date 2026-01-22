from __future__ import annotations

from typing import Literal, NotRequired, TypeAlias, TypedDict

Suit = Literal["Moons", "Suns", "Waves", "Leaves", "Wyrms", "Knots"]
PlayerId = Literal["PlayerA", "PlayerB"]
Winner = Literal["PlayerA", "PlayerB", "Draw"]
WinnerDecider = Literal["districts", "rank-total", "resources", "draw"]
GamePhase = Literal[
    "StartTurn",
    "TaxCheck",
    "CollectIncome",
    "ActionWindow",
    "DrawCard",
    "GameOver",
]
ActionId = Literal[
    "buy-deed",
    "choose-income-suit",
    "develop-deed",
    "develop-outright",
    "end-turn",
    "sell-card",
    "trade",
]
BridgeCommandName = Literal["metadata", "reset", "step", "legalActions", "observation", "serialize"]


class ResourcePoolPayload(TypedDict):
    Moons: int
    Suns: int
    Waves: int
    Leaves: int
    Wyrms: int
    Knots: int


class SuitCountsPayload(TypedDict, total=False):
    Moons: int
    Suns: int
    Waves: int
    Leaves: int
    Wyrms: int
    Knots: int


class IncomeRollPayload(TypedDict):
    die1: int
    die2: int


class IncomeChoicePayload(TypedDict):
    playerId: PlayerId
    districtId: str
    cardId: str
    suits: list[Suit]


class GameLogEntryPayload(TypedDict):
    turn: int
    player: PlayerId
    phase: GamePhase
    summary: str
    details: NotRequired[dict[str, object]]


class PlayerTotalsPayload(TypedDict):
    PlayerA: int
    PlayerB: int


class FinalScorePayload(TypedDict):
    districtPoints: PlayerTotalsPayload
    rankTotals: PlayerTotalsPayload
    resourceTotals: PlayerTotalsPayload
    winner: Winner
    decidedBy: WinnerDecider


class DeedPayload(TypedDict):
    cardId: str
    progress: int
    tokens: SuitCountsPayload


class DistrictStackPayload(TypedDict):
    developed: list[str]
    deed: NotRequired[DeedPayload]


class DistrictStacksPayload(TypedDict):
    PlayerA: DistrictStackPayload
    PlayerB: DistrictStackPayload


class DistrictPayload(TypedDict):
    id: str
    markerSuitMask: list[Suit]
    stacks: DistrictStacksPayload


class DeckStatePayload(TypedDict):
    draw: list[str]
    discard: list[str]
    reshuffles: int


class DeckViewPayload(TypedDict):
    drawCount: int
    discard: list[str]
    reshuffles: int


class PlayerStatePayload(TypedDict):
    id: PlayerId
    hand: list[str]
    crowns: list[str]
    resources: ResourcePoolPayload


class ObservedPlayerPayload(TypedDict):
    id: PlayerId
    crowns: list[str]
    resources: ResourcePoolPayload
    hand: list[str]
    handCount: int
    handHidden: bool


class SerializedStatePayload(TypedDict):
    schemaVersion: int
    seed: str
    rngCursor: int
    deck: DeckStatePayload
    players: list[PlayerStatePayload]
    activePlayerIndex: int
    turn: int
    phase: GamePhase
    districts: list[DistrictPayload]
    cardPlayedThisTurn: bool
    log: list[GameLogEntryPayload]
    finalTurnsRemaining: NotRequired[int]
    lastIncomeRoll: NotRequired[IncomeRollPayload]
    lastTaxSuit: NotRequired[Suit]
    pendingIncomeChoices: NotRequired[list[IncomeChoicePayload]]
    incomeChoiceReturnPlayerId: NotRequired[PlayerId]
    finalScore: NotRequired[FinalScorePayload]


class PlayerViewPayload(TypedDict):
    viewerId: PlayerId
    activePlayerId: PlayerId
    turn: int
    phase: GamePhase
    districts: list[DistrictPayload]
    players: list[ObservedPlayerPayload]
    deck: DeckViewPayload
    cardPlayedThisTurn: bool
    log: list[GameLogEntryPayload]
    finalTurnsRemaining: NotRequired[int]
    lastIncomeRoll: NotRequired[IncomeRollPayload]
    lastTaxSuit: NotRequired[Suit]
    pendingIncomeChoices: NotRequired[list[IncomeChoicePayload]]
    incomeChoiceReturnPlayerId: NotRequired[PlayerId]
    finalScore: NotRequired[FinalScorePayload]


class BuyDeedActionPayload(TypedDict):
    type: Literal["buy-deed"]
    cardId: str
    districtId: str


class ChooseIncomeSuitActionPayload(TypedDict):
    type: Literal["choose-income-suit"]
    playerId: PlayerId
    districtId: str
    cardId: str
    suit: Suit


class DevelopDeedActionPayload(TypedDict):
    type: Literal["develop-deed"]
    districtId: str
    cardId: str
    tokens: SuitCountsPayload


class DevelopOutrightActionPayload(TypedDict):
    type: Literal["develop-outright"]
    cardId: str
    districtId: str
    payment: SuitCountsPayload


class EndTurnActionPayload(TypedDict):
    type: Literal["end-turn"]


class SellCardActionPayload(TypedDict):
    type: Literal["sell-card"]
    cardId: str


class TradeActionPayload(TypedDict):
    type: Literal["trade"]
    give: Suit
    receive: Suit


GameActionPayload: TypeAlias = (
    BuyDeedActionPayload
    | ChooseIncomeSuitActionPayload
    | DevelopDeedActionPayload
    | DevelopOutrightActionPayload
    | EndTurnActionPayload
    | SellCardActionPayload
    | TradeActionPayload
)


class BridgeActionSurfacePayload(TypedDict):
    stableKey: Literal["actionKey"]
    canonicalOrder: Literal["ascending_lexicographic_action_key"]


class BridgeObservationSpecPayload(TypedDict):
    name: Literal["player_view_v1"]
    defaultViewer: Literal["active-player"]
    optionalMask: Literal["legal action keys"]


class BridgeModelIOInputsPayload(TypedDict):
    observation: Literal["observation"]
    actionMask: Literal["action_mask"]


class BridgeModelIOOutputsPayload(TypedDict):
    maskedLogits: Literal["masked_logits"]
    value: Literal["value"]


class BridgeModelIOPayload(TypedDict):
    inputs: BridgeModelIOInputsPayload
    outputs: BridgeModelIOOutputsPayload


class BridgeMetadataPayload(TypedDict):
    contractName: Literal["magnate_bridge"]
    contractVersion: Literal["v1"]
    schemaVersion: int
    commands: list[BridgeCommandName]
    actionIds: list[ActionId]
    actionSurface: BridgeActionSurfacePayload
    observationSpec: BridgeObservationSpecPayload
    modelIO: BridgeModelIOPayload
