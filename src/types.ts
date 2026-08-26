export type Champion = {
  id: string
  name: string
  cost: number
  role: string
  traits: string[]
  traitNames: string[]
  stats: Record<string, number>
  ability: { name: string; description: string }
  image: string
}

export type Trait = {
  id: string
  name: string
  nameEn: string
  description: string
  searchable: boolean
  unique: boolean
  breakpoints: number[]
  image: string
}

export type Item = {
  id: string
  name: string
  nameEn: string
  description: string
  category: string
  composition: string[]
  tags: string[]
  effects: Record<string, number>
  traits: string[]
  image: string
}

export type Set18Data = {
  generatedAt: string
  set: number
  patch: string
  setName: string
  setNameEn: string
  dataVersion: string
  sources: Array<{ id: string; name: string; url: string; type: string }>
  champions: Champion[]
  traits: Trait[]
  items: Item[]
}

export type MetaComp = {
  id: string
  name: string
  tier: string
  carries: string[]
  leveling: string
  metaScore: number
  sourceIds: string[]
}

export type MetaData = {
  generatedAt: string
  patch: string
  sources: Array<{ id: string; name: string; url: string; note: string }>
  comps: Array<MetaComp & { avgPlacement?: number; games?: number; pickRate?: number }>
}

export type CoachInput = {
  level: number
  ownedChampionIds: string[]
  components: string[]
}

export type ActiveTrait = {
  trait: Trait
  count: number
  activeBreakpoint: number
  nextBreakpoint?: number
}

export type ItemSuggestion = {
  item: Item
  score: number
  holder?: Champion
  reason: string
}

export type CompRecommendation = {
  comp: MetaComp
  score: number
  board: Champion[]
  activeTraits: ActiveTrait[]
  matchReasons: string[]
}

export type CoachResult = {
  earlyBoard: Champion[]
  earlyTraits: ActiveTrait[]
  itemSuggestions: ItemSuggestion[]
  buyNext: Champion[]
  comps: CompRecommendation[]
}

export type ApiActiveTrait = {
  traitId: string
  count: number
  activeBreakpoint: number
  nextBreakpoint?: number
}

export type ApiCompRecommendation = {
  id: string
  rank: number
  name: string
  tier: string
  score: number
  confidence?: number
  uncertainty?: number
  crossSource?: boolean
  modelDisagreement?: number
  componentFit?: number
  transitionFit?: number
  avgPlacement?: number | null
  games: number
  pickRate: number
  earlyBoardIds: string[]
  boardIds: string[]
  carryIds: string[]
  activeTraits: ApiActiveTrait[]
  matchReasons: string[]
  leveling: string
  transitionPath?: Array<{
    level: number
    boardIds: string[]
    avgPlacement?: number | null
    games: number
  }>
}

export type ApiStageItem = {
  stage: 'opener' | 'mid' | 'late'
  stageLabel: string
  itemId: string
  holderId: string
  finalHolderId?: string | null
  score: number
  reason: string
  transferReason?: string
  sampleCount: number
}

export type ApiBisBuild = {
  stage: 'bis'
  stageLabel: string
  holderId: string
  itemIds: string[]
  score: number
  sampleCount: number
  avgPlacement: number
}

export type ApiCoachResult = {
  earlyBoardIds: string[]
  earlyTraits: ApiActiveTrait[]
  buyNextIds: string[]
  comps: ApiCompRecommendation[]
  itemPlan: Array<ApiStageItem | ApiBisBuild>
  model: {
    boardAvailable: boolean
    itemAvailable: boolean
    board?: { evaluation?: { calibratedMAE?: number; rankingAccuracy?: number }; samples?: number; ensembleSize?: number }
    item?: { evaluation?: { calibratedMAE?: number; rankingAccuracy?: number }; samples?: number; ensembleSize?: number }
  }
  data: {
    set?: number
    patch?: string
    queue?: string
    generatedAt?: string
    clusterId?: number
    clusters?: number
    trainingSamples?: number
    crossSourceRows?: number
    opggGames24h?: number
    highEloUnitPriors?: number
    highEloItemHolderPriors?: number
  }
}
