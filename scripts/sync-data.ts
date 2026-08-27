import { mkdir, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { cleanHtml, fetchJson, fetchText, USER_AGENT, writeJson } from './lib'

type TfTraitRef = { name: string; normalizedName: string; id: string; amount: number; isSearchable: boolean }
type TfChampion = {
  displayName: string
  characterId: string
  cost: number
  teamPlannerCode: number
  role: string
  traits: TfTraitRef[]
  stats: Record<string, number>
  ability: { name: string; descriptionTemplate: string }
  icons: { square: string; cdragonSplash?: string | null }
}
type TfTrait = {
  displayName: string
  normalizedName: string
  traitId: string
  isSearchable: boolean
  isUniqueDisplayOnly: boolean
  tooltipTemplate: string
  iconPath: string
  conditionalSets: Array<{ min_units: number; max_units?: number; style_name: string }>
}
type TfItem = {
  apiName: string
  name: string
  descriptionTemplate: string
  iconPath: string
  effects: Record<string, number>
  composition: string[]
  tags: string[]
  category: string
  associatedTraits?: string[]
}
type CDragon = {
  items: Array<{
    apiName: string
    name: string
    desc?: string | null
    composition?: string[]
    effects?: Record<string, number>
    tags?: string[]
  }>
  sets: Record<string, { traits?: Array<{ apiName: string; name: string; desc?: string }> }>
}

type LiveCluster = {
  cluster_info: {
    tft_set: string
    cluster_id: number
    cluster_details: {
      unit_lookup: Record<string, {
        assetName: string
        name: string
        unit_cost: number
        trait_bonus: Array<{ apiName: string; value: number }>
      }>
    }
  }
}

type GamesPayload = {
  games: Array<{ srq: string[]; patch: string[]; count: number }>
}

const TFTRAITS = 'https://tftraits.com'
const CDRAGON_VI = 'https://raw.communitydragon.org/latest/cdragon/tft/vi_vn.json'

async function getVersion() {
  const html = await fetchText(`${TFTRAITS}/explorer/`)
  const version = html.match(/<meta name="tft-set" content="([^"]+)"/)?.[1]
  if (!version) throw new Error('Không tìm thấy version Set 18 từ TFTraits')
  return version
}

async function getLivePatch() {
  const payload = await fetchJson<GamesPayload>('https://api-hc.metatft.com/tft-stat-api/games?days=1')
  const patches = payload.games
    .filter((row) => row.srq.at(-1) === '1100')
    .map((row) => row.patch[0] ?? '')
    .filter((patch) => /^18\.\d+$/.test(patch))
    .map((patch) => patch.split('.').map(Number) as [number, number])
  if (!patches.length) throw new Error('Không thấy trận ranked Set 18 live trong feed hiện tại')
  patches.sort((a, b) => b[0] - a[0] || b[1] - a[1])
  return `${patches[0][0]}.${patches[0][1]}`
}

function assetUrl(path: string) {
  return `${TFTRAITS}/${path.replace(/^\//, '')}`
}

async function downloadAsset(url: string, output: string) {
  const response = await fetch(url, { headers: { 'user-agent': USER_AGENT } })
  if (!response.ok) throw new Error(`${response.status}: ${url}`)
  await mkdir(join(output, '..'), { recursive: true })
  await writeFile(output, Buffer.from(await response.arrayBuffer()))
}

async function main() {
  const [version, livePatch] = await Promise.all([getVersion(), getLivePatch()])
  const query = `v=${encodeURIComponent(version)}`
  const [championsRaw, traits, items, cdragon, liveCluster] = await Promise.all([
    fetchJson<TfChampion[]>(`${TFTRAITS}/api/display/champions?${query}`),
    fetchJson<TfTrait[]>(`${TFTRAITS}/api/display/traits?${query}`),
    fetchJson<TfItem[]>(`${TFTRAITS}/api/display/items?${query}`),
    fetchJson<CDragon>(CDRAGON_VI),
    fetchJson<LiveCluster>('https://api-hc.metatft.com/tft-comps-api/latest_cluster_info'),
  ])

  if (liveCluster.cluster_info.tft_set !== 'TFTSet18') {
    throw new Error(`Meta live không phải Set 18: ${liveCluster.cluster_info.tft_set}`)
  }
  const liveUnits = liveCluster.cluster_info.cluster_details.unit_lookup
  const champions = championsRaw.filter((champion) => {
    const live = liveUnits[champion.characterId]
    if (!live || live.unit_cost < 1 || live.unit_cost > 5) return false
    const sourceTraits = champion.traits.map((trait) => trait.id).sort()
    const liveTraits = live.trait_bonus.filter((trait) => trait.value > 0).map((trait) => trait.apiName).sort()
    return champion.cost === live.unit_cost && sourceTraits.join('|') === liveTraits.join('|')
  })

  if (champions.length < 60 || traits.length < 30) {
    throw new Error(`Set 18 live validation chưa đủ: ${champions.length} tướng, ${traits.length} tộc/hệ`)
  }

  const viItems = new Map(cdragon.items.map((item) => [item.apiName, item]))
  const viTraits = new Map((cdragon.sets['18']?.traits ?? []).map((trait) => [trait.apiName, trait]))

  const selectedItems = items.filter((item) => ['component', 'completed', 'artifact', 'radiant', 'emblem'].includes(item.category))

  const payload = {
    generatedAt: new Date().toISOString(),
    set: 18,
    patch: livePatch,
    setName: 'Đại Ngàn Kỳ Bí',
    setNameEn: 'Enchanted Wilds',
    dataVersion: version,
    sources: [
      { id: 'riot-cdragon', name: 'CommunityDragon / Riot live client data', url: CDRAGON_VI, type: 'game-data' },
      { id: 'metatft-live', name: 'MetaTFT live Set 18 validation', url: 'https://www.metatft.com/', type: 'live-validation' },
      { id: 'tftraits', name: 'TFTraits normalized catalog (live-ID validated)', url: `${TFTRAITS}/explorer/`, type: 'normalized-enrichment' },
    ],
    champions: champions.map((champion) => ({
      id: champion.characterId,
      name: champion.displayName,
      cost: champion.cost,
      role: champion.role,
      traits: liveUnits[champion.characterId].trait_bonus.filter((trait) => trait.value > 0).map((trait) => trait.apiName),
      traitNames: liveUnits[champion.characterId].trait_bonus
        .filter((trait) => trait.value > 0)
        .map((trait) => viTraits.get(trait.apiName)?.name ?? champion.traits.find((entry) => entry.id === trait.apiName)?.name ?? trait.apiName),
      stats: champion.stats,
      ability: {
        name: champion.ability?.name ?? '',
        description: cleanHtml(champion.ability?.descriptionTemplate ?? ''),
      },
      image: `/${champion.icons.square}`,
    })),
    traits: traits.map((trait) => ({
      id: trait.traitId,
      name: viTraits.get(trait.traitId)?.name ?? trait.displayName,
      nameEn: trait.displayName,
      description: cleanHtml(viTraits.get(trait.traitId)?.desc ?? trait.tooltipTemplate),
      searchable: trait.isSearchable,
      unique: trait.isUniqueDisplayOnly,
      breakpoints: trait.conditionalSets.map((set) => set.min_units),
      image: `/${trait.iconPath.replace(/^\//, '')}`,
    })),
    items: selectedItems.map((item) => {
      const localized = viItems.get(item.apiName)
      return {
        id: item.apiName,
        name: localized?.name ?? item.name,
        nameEn: item.name,
        description: cleanHtml(localized?.desc ?? item.descriptionTemplate),
        category: item.category,
        composition: localized?.composition?.length ? localized.composition : item.composition,
        tags: localized?.tags?.length ? localized.tags : item.tags,
        effects: localized?.effects && Object.keys(localized.effects).length ? localized.effects : item.effects,
        traits: item.associatedTraits ?? [],
        image: `/${item.iconPath.replace(/^\//, '')}`,
      }
    }),
  }

  await writeJson('src/data/set18.generated.json', payload)

  const assets: Array<{ url: string; target: string }> = []
  for (const champion of champions) {
    assets.push({ url: assetUrl(champion.icons.square), target: join('public', champion.icons.square) })
  }
  for (const trait of traits) {
    assets.push({ url: assetUrl(trait.iconPath), target: join('public', trait.iconPath.replace(/^\//, '')) })
  }
  for (const item of selectedItems) {
    assets.push({ url: assetUrl(item.iconPath), target: join('public', item.iconPath.replace(/^\//, '')) })
  }

  let downloaded = 0
  for (const asset of assets) {
    try {
      await downloadAsset(asset.url, asset.target)
      downloaded += 1
    } catch (error) {
      console.warn('asset failed', asset.url, error instanceof Error ? error.message : error)
    }
  }

  console.log(`Set 18: ${champions.length} tướng, ${traits.length} tộc/hệ, ${selectedItems.length} trang bị; ${downloaded}/${assets.length} ảnh.`)
  console.log(`Version: ${version}; live patch: ${livePatch}; MetaTFT cluster: ${liveCluster.cluster_info.cluster_id}`)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
