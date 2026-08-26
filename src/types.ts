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
  comps: MetaComp[]
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
