import { readJson, writeJson } from './lib'

type SetData = {
  champions: Array<{ id: string; name: string }>
  traits: Array<{ id: string; name: string }>
}

type Snapshot = {
  generatedAt: string
  patch: string
  queue: string
  clusterId: number
  clusters: Array<{
    id: number
    nameParts: Array<{ name: string; score?: number; type?: string }>
    centroidUnits: string[]
    overall: { avg: number; count: number; pick: number }
    unitStats: Array<{ unit: string; count: number; avg: number }>
  }>
}

function tier(avg: number) {
  if (avg > 0 && avg <= 3.45) return 'S'
  if (avg > 0 && avg <= 4.05) return 'A'
  return 'B'
}

function score(avg: number, games: number) {
  const placement = avg >= 1 && avg <= 8 ? (8.5 - avg) / 7.5 : 0.5
  const evidence = Math.min(1, Math.log1p(Math.max(0, games)) / Math.log(501))
  return Math.max(0, Math.min(1, placement * 0.82 + evidence * 0.18))
}

async function main() {
  const set = await readJson<SetData>('src/data/set18.generated.json')
  const snapshot = await readJson<Snapshot>('backend/data/metatft.snapshot.json')
  if (snapshot.queue !== 'LIVE' || !snapshot.patch.startsWith('18.')) {
    throw new Error(`Refuse non-live meta snapshot: queue=${snapshot.queue}, patch=${snapshot.patch}`)
  }

  const championById = new Map(set.champions.map((champion) => [champion.id, champion]))
  const traitById = new Map(set.traits.map((trait) => [trait.id, trait]))
  const comps = snapshot.clusters
    .map((cluster) => {
      const labels = cluster.nameParts
        .map((part) => championById.get(part.name)?.name ?? traitById.get(part.name)?.name)
        .filter((name): name is string => Boolean(name))
      const carries = cluster.nameParts
        .filter((part) => part.type === 'unit')
        .map((part) => championById.get(part.name)?.name)
        .filter((name): name is string => Boolean(name))
      const fallbackCarries = cluster.unitStats
        .filter((row) => championById.has(row.unit))
        .sort((a, b) => b.count - a.count)
        .slice(0, 3)
        .map((row) => championById.get(row.unit)!.name)
      const avg = cluster.overall.avg
      const games = cluster.overall.count
      return {
        id: String(cluster.id),
        name: labels.slice(0, 3).join(' · ') || `Comp ${cluster.id}`,
        tier: tier(avg),
        carries: carries.length ? carries : fallbackCarries,
        leveling: 'Theo board live tối ưu',
        metaScore: score(avg, games),
        avgPlacement: avg,
        games,
        pickRate: cluster.overall.pick,
        sourceIds: ['metatft-live'],
      }
    })
    .filter((comp) => comp.carries.length > 0)
    .sort((a, b) => b.metaScore - a.metaScore)

  await writeJson('src/data/meta.generated.json', {
    generatedAt: snapshot.generatedAt,
    patch: snapshot.patch,
    sources: [
      {
        id: 'metatft-live',
        name: 'MetaTFT live/current statistics',
        url: 'https://www.metatft.com/',
        note: `Set 18 live patch ${snapshot.patch}, cluster ${snapshot.clusterId}; không dùng queue PBE.`,
      },
      {
        id: 'riot-match',
        name: 'Riot TFT Match-V1',
        url: 'https://developer.riotgames.com/apis#tft-match-v1',
        note: 'Ground truth match history khi cấu hình RIOT_API_KEY; collector tự lọc tft_set_number=18.',
      },
    ],
    comps,
  })
  console.log(`Live meta: ${comps.length} clusters, patch ${snapshot.patch}.`)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
