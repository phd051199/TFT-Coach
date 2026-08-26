import type {
  ActiveTrait,
  Champion,
  CoachInput,
  CoachResult,
  CompRecommendation,
  Item,
  ItemSuggestion,
  MetaComp,
  Set18Data,
  Trait,
} from '../types'

type Indexes = {
  championById: Map<string, Champion>
  championByName: Map<string, Champion>
  traitById: Map<string, Trait>
}

const roleIsTank = (role: string) => role.includes('Tank') || role.includes('Fighter')
const roleIsAd = (role: string) => role.startsWith('AD')
const roleIsAp = (role: string) => role.startsWith('AP')

function makeIndexes(data: Set18Data): Indexes {
  return {
    championById: new Map(data.champions.map((champion) => [champion.id, champion])),
    championByName: new Map(data.champions.map((champion) => [champion.name, champion])),
    traitById: new Map(data.traits.map((trait) => [trait.id, trait])),
  }
}

export function activeTraits(board: Champion[], data: Set18Data): ActiveTrait[] {
  const traitMap = new Map<string, number>()
  for (const champion of board) {
    for (const traitId of champion.traits) traitMap.set(traitId, (traitMap.get(traitId) ?? 0) + 1)
  }
  const traitsById = new Map(data.traits.map((trait) => [trait.id, trait]))
  const output: ActiveTrait[] = []
  for (const [id, count] of traitMap.entries()) {
    const trait = traitsById.get(id)
    if (!trait) continue
    const breakpoints = [...trait.breakpoints].sort((a, b) => a - b)
    const activeBreakpoint = breakpoints.filter((point) => point <= count).at(-1) ?? 0
    const nextBreakpoint = breakpoints.find((point) => point > count)
    output.push({ trait, count, activeBreakpoint, ...(nextBreakpoint === undefined ? {} : { nextBreakpoint }) })
  }
  return output
    .sort((a, b) => {
      if (Boolean(a.activeBreakpoint) !== Boolean(b.activeBreakpoint)) return b.activeBreakpoint - a.activeBreakpoint
      return b.count - a.count
    })
}

function boardScore(
  board: Champion[],
  data: Set18Data,
  owned: Set<string>,
  pathCore: Set<string> = new Set(),
  early = false,
) {
  const traits = activeTraits(board, data)
  let score = 0
  for (const trait of traits) {
    if (trait.activeBreakpoint > 0) score += 7 + trait.activeBreakpoint * 0.75
    else if (trait.nextBreakpoint && trait.nextBreakpoint - trait.count === 1) score += 2.1
    if (trait.nextBreakpoint && trait.nextBreakpoint - trait.count === 1) score += 1.2
  }

  const tanks = board.filter((champion) => roleIsTank(champion.role)).length
  const damage = board.filter((champion) => champion.role.includes('Carry') || champion.role.includes('Caster') || champion.role.includes('Reaper')).length
  score += Math.min(tanks, 3) * 1.9 + Math.min(damage, 3) * 1.5

  for (const champion of board) {
    if (owned.has(champion.id)) score += 5.2
    if (pathCore.has(champion.name)) score += 3.8
    const hp = champion.stats.hp ?? 0
    const damageStat = champion.stats.damage ?? 0
    const attackSpeed = champion.stats.attackSpeed ?? 0
    score += (hp / 1000 + damageStat / 100 + attackSpeed) * (early ? 0.9 : 0.35)
    if (early) score -= Math.max(0, champion.cost - 2) * 1.6
  }

  return score
}

function beamBestBoard(
  pool: Champion[],
  size: number,
  data: Set18Data,
  owned: Set<string>,
  pathCore: Set<string>,
  early: boolean,
) {
  type State = { board: Champion[]; score: number }
  let beam: State[] = [{ board: [], score: 0 }]
  const width = early ? 170 : 110

  for (let slot = 0; slot < size; slot += 1) {
    const next = new Map<string, State>()
    for (const state of beam) {
      const used = new Set(state.board.map((champion) => champion.id))
      for (const champion of pool) {
        if (used.has(champion.id)) continue
        const board = [...state.board, champion]
        const key = board.map((unit) => unit.id).sort().join('|')
        const score = boardScore(board, data, owned, pathCore, early)
        const existing = next.get(key)
        if (!existing || score > existing.score) next.set(key, { board, score })
      }
    }
    beam = [...next.values()].sort((a, b) => b.score - a.score).slice(0, width)
  }
  return beam[0]?.board ?? []
}

function buildCompBoard(comp: MetaComp, data: Set18Data, indexes: Indexes, owned: Set<string>) {
  const core = new Set(comp.carries)
  const locked = comp.carries
    .map((name) => indexes.championByName.get(name))
    .filter((champion): champion is Champion => Boolean(champion))
  const targetSize = comp.leveling.toLowerCase().includes('9') ? 9 : 8
  const remaining = data.champions.filter((champion) => !locked.some((unit) => unit.id === champion.id))
  const fillerCount = Math.max(0, targetSize - locked.length)
  if (!fillerCount) return locked.slice(0, targetSize)

  const filler = beamBestBoard(
    [...locked, ...remaining],
    targetSize,
    data,
    owned,
    core,
    false,
  )
  // Beam search is free to drop a carry; enforce source core after search and refill greedily.
  const withCore = [...locked]
  for (const champion of filler) {
    if (withCore.length >= targetSize) break
    if (!withCore.some((unit) => unit.id === champion.id)) withCore.push(champion)
  }
  return withCore
}

function compScore(comp: MetaComp, board: Champion[], input: CoachInput, data: Set18Data, indexes: Indexes) {
  const owned = new Set(input.ownedChampionIds)
  const componentTags = new Set(
    input.components.flatMap((id) => {
      const component = data.items.find((item) => item.id === id)
      const name = component?.nameEn.toLowerCase() ?? ''
      if (/sword|bow|glove/.test(name)) return ['ad']
      if (/rod|tear/.test(name)) return ['ap']
      if (/vest|cloak|belt/.test(name)) return ['tank']
      return ['flex']
    }),
  )
  let score = comp.metaScore * 50
  const reasons: string[] = []

  const ownedNames = input.ownedChampionIds.map((id) => indexes.championById.get(id)?.name).filter(Boolean)
  const coreHits = comp.carries.filter((name) => ownedNames.includes(name)).length
  const boardHits = board.filter((champion) => owned.has(champion.id)).length
  score += coreHits * 17 + boardHits * 4.5
  if (coreHits) reasons.push(`Có sẵn ${coreHits} tướng core`)
  else if (boardHits >= 2) reasons.push(`Giữ được ${boardHits} tướng đang có khi chuyển bài`)

  const carryUnits = comp.carries.map((name) => indexes.championByName.get(name)).filter((unit): unit is Champion => Boolean(unit))
  const adCarries = carryUnits.filter((unit) => roleIsAd(unit.role)).length
  const apCarries = carryUnits.filter((unit) => roleIsAp(unit.role)).length
  const tanks = carryUnits.filter((unit) => roleIsTank(unit.role)).length
  if (componentTags.has('ad') && adCarries) {
    score += 7 + adCarries * 1.5
    reasons.push('Đồ rơi hợp carry AD')
  }
  if (componentTags.has('ap') && apCarries) {
    score += 7 + apCarries * 1.5
    reasons.push('Đồ rơi hợp carry AP')
  }
  if (componentTags.has('tank') && tanks) {
    score += 4.5
    reasons.push('Có hướng ghép đồ thủ cho frontline')
  }

  const active = activeTraits(board, data).filter((trait) => trait.activeBreakpoint > 0)
  score += active.length * 1.3
  return { score, reasons, active }
}

function canCraft(composition: string[], components: string[]) {
  const available = new Map<string, number>()
  for (const id of components) available.set(id, (available.get(id) ?? 0) + 1)
  for (const id of composition) {
    const left = available.get(id) ?? 0
    if (left <= 0) return false
    available.set(id, left - 1)
  }
  return true
}

function itemRoleScore(item: Item, holder: Champion) {
  const tags = new Set(item.tags)
  let score = 0
  if (tags.has('tank') && roleIsTank(holder.role)) score += 8
  if (tags.has('attack') && roleIsAd(holder.role)) score += 8
  if (tags.has('magic') && roleIsAp(holder.role)) score += 8
  if (tags.has('stacking') && (holder.role.includes('Carry') || holder.role.includes('Caster'))) score += 2

  const name = item.nameEn.toLowerCase()
  if (/guinsoo|red buff|kraken/.test(name) && (holder.role.includes('Carry') || holder.traitNames.includes('Liên Kích'))) score += 4.5
  if (/spear of shojin|blue buff|archangel/.test(name) && (roleIsAp(holder.role) || holder.traitNames.includes('Thuật Sĩ'))) score += 4.2
  if (/gargoyle|warmog|bramble|dragon's claw/.test(name) && roleIsTank(holder.role)) score += 4.2
  if (/infinity edge|deathblade|last whisper/.test(name) && roleIsAd(holder.role)) score += 3.5
  if (/ionic spark|sunfire|evenshroud/.test(name) && roleIsTank(holder.role)) score += 2.8
  return score
}

function suggestItems(input: CoachInput, data: Set18Data, targetBoard: Champion[]): ItemSuggestion[] {
  if (input.components.length < 2) return []
  const craftable = data.items.filter(
    (item) => item.category === 'completed' && item.composition.length === 2 && canCraft(item.composition, input.components),
  )
  const holders = targetBoard.length ? targetBoard : data.champions.filter((champion) => champion.cost <= 3)
  return craftable
    .map((item) => {
      let bestHolder = holders[0]
      let best = -Infinity
      for (const holder of holders) {
        const score = itemRoleScore(item, holder)
        if (score > best) {
          best = score
          bestHolder = holder
        }
      }
      const tempo = ['Sunfire Cape', 'Gargoyle Stoneplate', 'Guinsoo\'s Rageblade', 'Spear of Shojin', 'Infinity Edge', 'Warmog\'s Armor']
        .some((name) => item.nameEn.includes(name))
        ? 2.5
        : 0
      const score = best + tempo
      const reason = bestHolder
        ? `${bestHolder.name} dùng tốt (${roleIsTank(bestHolder.role) ? 'frontline' : roleIsAd(bestHolder.role) ? 'AD' : 'AP'}); ghép được ngay từ đồ đang có.`
        : 'Ghép được ngay từ đồ đang có.'
      return { item, holder: bestHolder, score, reason }
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
}

export function recommend(input: CoachInput, data: Set18Data, comps: MetaComp[]): CoachResult {
  const indexes = makeIndexes(data)
  const owned = new Set(input.ownedChampionIds)

  const compRecommendations: CompRecommendation[] = comps
    .map((comp) => {
      const board = buildCompBoard(comp, data, indexes, owned)
      const { score, reasons, active } = compScore(comp, board, input, data, indexes)
      return { comp, score, board, activeTraits: active, matchReasons: reasons }
    })
    .sort((a, b) => b.score - a.score)

  const topComp = compRecommendations[0]
  const pathCore = new Set(topComp?.comp.carries ?? [])
  const earlyPool = data.champions.filter((champion) => champion.cost <= 3 || owned.has(champion.id))
  const earlyBoard = beamBestBoard(earlyPool, Math.min(Math.max(input.level, 3), 6), data, owned, pathCore, true)
  const earlyTraits = activeTraits(earlyBoard, data).filter((trait) => trait.activeBreakpoint > 0)
  const itemSuggestions = suggestItems(input, data, topComp?.board ?? earlyBoard)

  const buyNext = [...earlyBoard, ...(topComp?.board ?? [])]
    .filter((champion, index, list) => list.findIndex((item) => item.id === champion.id) === index)
    .filter((champion) => !owned.has(champion.id))
    .sort((a, b) => {
      const aCore = pathCore.has(a.name) ? 1 : 0
      const bCore = pathCore.has(b.name) ? 1 : 0
      if (aCore !== bCore) return bCore - aCore
      return a.cost - b.cost
    })
    .slice(0, 8)

  return { earlyBoard, earlyTraits, itemSuggestions, buyNext, comps: compRecommendations.slice(0, 5) }
}
