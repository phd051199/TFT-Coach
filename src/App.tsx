import {
  BarChart3,
  BookOpen,
  Database,
  ExternalLink,
  Flame,
  LayoutDashboard,
  RefreshCw,
  Shield,
  Sparkles,
  Sword,
  Target,
  TrendingUp,
} from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import setRaw from './data/set18.generated.json'
import metaRaw from './data/meta.generated.json'
import { recommend } from './engine/recommend'
import { requestCoach } from './api/coach'
import { BoardCanvas } from './components/BoardCanvas'
import { ChampionPicker } from './components/ChampionPicker'
import { ItemPicker } from './components/ItemPicker'
import type {
  ActiveTrait,
  ApiBisBuild,
  ApiCoachResult,
  ApiStageItem,
  Champion,
  CoachInput,
  Item,
  ItemSuggestion,
  MetaData,
  Set18Data,
} from './types'
import './App.css'

const setData = setRaw as unknown as Set18Data
const metaData = metaRaw as unknown as MetaData

type View = 'coach' | 'meta' | 'library' | 'items' | 'sources'

const costLabels: Record<number, string> = {
  1: '1 vàng', 2: '2 vàng', 3: '3 vàng', 4: '4 vàng', 5: '5 vàng',
}

function formatAge(iso: string) {
  const date = new Date(iso)
  return new Intl.DateTimeFormat('vi-VN', { day: '2-digit', month: '2-digit', hour: '2-digit', minute: '2-digit' }).format(date)
}

function ChampionMini({ champion, label }: { champion: Champion; label?: string }) {
  return (
    <div className={`champion-mini cost-${champion.cost}`} title={`${champion.name} · ${champion.traitNames.join(' / ')}`}>
      <img src={champion.image} alt="" />
      <div><strong>{champion.name}</strong><span>{label ?? champion.traitNames.slice(0, 2).join(' · ')}</span></div>
    </div>
  )
}

function ItemMini({ item, components }: { item: Item; components: Item[] }) {
  return (
    <div className="item-mini">
      <img className="item-icon-lg" src={item.image} alt="" />
      <div className="item-copy">
        <strong>{item.name}</strong>
        <span>{item.nameEn}</span>
        {item.composition.length === 2 && (
          <div className="recipe-row">
            {item.composition.map((componentId, index) => {
              const component = components.find((entry) => entry.id === componentId)
              return component ? <img key={`${componentId}-${index}`} src={component.image} title={component.name} alt="" /> : null
            })}
          </div>
        )}
      </div>
    </div>
  )
}

type DisplayComp = {
  id: string
  name: string
  tier: string
  score: number
  confidence?: number
  uncertainty?: number
  crossSource?: boolean
  modelDisagreement?: number
  componentFit?: number
  transitionFit?: number
  transitionPath?: Array<{ level: number; board: Champion[]; avgPlacement?: number | null; games: number }>
  board: Champion[]
  carries: Champion[]
  activeTraits: ActiveTrait[]
  matchReasons: string[]
  leveling: string
  avgPlacement?: number | null
  games?: number
  pickRate?: number
}

function StageItemPlan({
  rows,
  bis,
  itemById,
  championById,
  components,
}: {
  rows: ApiStageItem[]
  bis: ApiBisBuild[]
  itemById: Map<string, Item>
  championById: Map<string, Champion>
  components: Item[]
}) {
  const stages: Array<{ id: ApiStageItem['stage']; title: string; note: string }> = [
    { id: 'opener', title: 'Đầu game', note: '2-1 → 2-5 · ưu tiên tempo/giữ máu' },
    { id: 'mid', title: 'Giữa game', note: '3-2 → 4-1 · giữ đường pivot' },
    { id: 'late', title: 'Cuối game', note: '4-2+ · chuyển đồ về carry/tank chính' },
  ]
  return (
    <div className="stage-item-plan">
      {stages.map((stage) => {
        const stageRows = rows.filter((row) => row.stage === stage.id)
        return (
          <article className="stage-column" key={stage.id}>
            <header><span>{stage.title}</span><small>{stage.note}</small></header>
            {stageRows.length ? stageRows.map((row) => {
              const item = itemById.get(row.itemId)
              const holder = championById.get(row.holderId)
              const finalHolder = row.finalHolderId ? championById.get(row.finalHolderId) : undefined
              if (!item || !holder) return null
              return (
                <div className="stage-item-row" key={`${stage.id}-${row.itemId}-${row.holderId}`}>
                  <ItemMini item={item} components={components} />
                  <div className="holder-route">
                    <span>Cầm ngay</span><strong>{holder.name}</strong>
                    {finalHolder && finalHolder.id !== holder.id && <small>→ chuyển cho {finalHolder.name}</small>}
                    <p>{row.reason}{row.sampleCount ? ` · ${row.sampleCount} mẫu live` : ''}</p>
                  </div>
                  <b className="fit-chip">{Math.round(row.score)}</b>
                </div>
              )
            }) : <div className="stage-empty">Chưa có combo 2 mảnh hợp lệ cho giai đoạn này.</div>}
          </article>
        )
      })}
      {bis.length > 0 && (
        <article className="stage-column bis-column">
          <header><span>BIS board cuối</span><small>Build 3 món quan sát được trên live</small></header>
          {bis.slice(0, 4).map((row) => {
            const holder = championById.get(row.holderId)
            if (!holder) return null
            return (
              <div className="bis-row" key={`${row.holderId}-${row.itemIds.join('-')}`}>
                <img className="bis-holder" src={holder.image} alt="" />
                <div><strong>{holder.name}</strong><span>avg {row.avgPlacement.toFixed(2)} · {row.sampleCount} mẫu</span></div>
                <div className="bis-items">
                  {row.itemIds.map((itemId) => {
                    const item = itemById.get(itemId)
                    return item ? <img src={item.image} title={item.name} alt={item.name} key={itemId} /> : null
                  })}
                </div>
              </div>
            )
          })}
        </article>
      )}
    </div>
  )
}

function CoachView() {
  const [input, setInput] = useState<CoachInput>({ level: 4, ownedChampionIds: [], components: [] })
  const components = useMemo(() => setData.items.filter((item) => item.category === 'component' && !/spatula|pan/i.test(item.nameEn)), [])
  const championById = useMemo(() => new Map(setData.champions.map((champion) => [champion.id, champion])), [])
  const traitById = useMemo(() => new Map(setData.traits.map((trait) => [trait.id, trait])), [])
  const itemById = useMemo(() => new Map(setData.items.map((item) => [item.id, item])), [])
  const fallback = useMemo(() => recommend(input, setData, metaData.comps), [input])
  const [apiResult, setApiResult] = useState<ApiCoachResult | null>(null)
  const [apiState, setApiState] = useState<'loading' | 'live' | 'fallback'>('loading')

  useEffect(() => {
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      setApiState('loading')
      void requestCoach(input, controller.signal)
        .then((result) => {
          setApiResult(result)
          setApiState('live')
        })
        .catch((error: unknown) => {
          if (error instanceof DOMException && error.name === 'AbortError') return
          setApiResult(null)
          setApiState('fallback')
        })
    }, 100)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [input])

  const usingApi = apiState === 'live' && Boolean(apiResult)
  const earlyBoard = usingApi
    ? apiResult!.earlyBoardIds.map((id) => championById.get(id)).filter((unit): unit is Champion => Boolean(unit))
    : fallback.earlyBoard
  const earlyTraits: ActiveTrait[] = usingApi
    ? apiResult!.earlyTraits.map((row) => {
      const trait = traitById.get(row.traitId)
      return trait ? { trait, count: row.count, activeBreakpoint: row.activeBreakpoint, ...(row.nextBreakpoint === undefined ? {} : { nextBreakpoint: row.nextBreakpoint }) } : null
    }).filter((row): row is ActiveTrait => Boolean(row))
    : fallback.earlyTraits
  const buyNext = usingApi
    ? apiResult!.buyNextIds.map((id) => championById.get(id)).filter((unit): unit is Champion => Boolean(unit))
    : fallback.buyNext
  const stageRows = (usingApi ? apiResult!.itemPlan.filter((row): row is ApiStageItem => row.stage !== 'bis') : [])
  const bisRows = (usingApi ? apiResult!.itemPlan.filter((row): row is ApiBisBuild => row.stage === 'bis') : [])

  const openerBoardItems: ItemSuggestion[] = usingApi
    ? stageRows.filter((row) => row.stage === 'opener').slice(0, 3).map((row) => ({
      item: itemById.get(row.itemId)!,
      holder: championById.get(row.holderId),
      score: row.score,
      reason: row.reason,
    })).filter((row) => Boolean(row.item))
    : fallback.itemSuggestions.slice(0, 3)

  const displayComps: DisplayComp[] = usingApi
    ? apiResult!.comps.map((comp) => ({
      id: comp.id,
      name: comp.name,
      tier: comp.tier,
      score: comp.score,
      confidence: comp.confidence,
      uncertainty: comp.uncertainty,
      crossSource: comp.crossSource,
      modelDisagreement: comp.modelDisagreement,
      componentFit: comp.componentFit,
      transitionFit: comp.transitionFit,
      transitionPath: comp.transitionPath?.map((row) => ({
        level: row.level,
        board: row.boardIds.map((id) => championById.get(id)).filter((unit): unit is Champion => Boolean(unit)),
        avgPlacement: row.avgPlacement,
        games: row.games,
      })),
      board: comp.boardIds.map((id) => championById.get(id)).filter((unit): unit is Champion => Boolean(unit)),
      carries: comp.carryIds.map((id) => championById.get(id)).filter((unit): unit is Champion => Boolean(unit)),
      activeTraits: comp.activeTraits.map((row) => {
        const trait = traitById.get(row.traitId)
        return trait ? { trait, count: row.count, activeBreakpoint: row.activeBreakpoint, ...(row.nextBreakpoint === undefined ? {} : { nextBreakpoint: row.nextBreakpoint }) } : null
      }).filter((row): row is ActiveTrait => Boolean(row)),
      matchReasons: comp.matchReasons,
      leveling: comp.leveling,
      avgPlacement: comp.avgPlacement,
      games: comp.games,
      pickRate: comp.pickRate,
    }))
    : fallback.comps.map((recommendation) => ({
      id: recommendation.comp.id,
      name: recommendation.comp.name,
      tier: recommendation.comp.tier,
      score: recommendation.score,
      board: recommendation.board,
      carries: recommendation.comp.carries
        .map((name) => setData.champions.find((unit) => unit.name === name))
        .filter((unit): unit is Champion => Boolean(unit)),
      activeTraits: recommendation.activeTraits,
      matchReasons: recommendation.matchReasons,
      leveling: recommendation.comp.leveling,
    }))
  const top = displayComps[0]
  const firstSlam = stageRows.find((row) => row.stage === 'opener')
  const firstSlamItem = firstSlam ? itemById.get(firstSlam.itemId) : fallback.itemSuggestions[0]?.item
  const firstSlamHolder = firstSlam ? championById.get(firstSlam.holderId) : fallback.itemSuggestions[0]?.holder

  function loadExample() {
    const names = ['Rakan', 'Xayah', 'Kobuko']
    const owned = setData.champions.filter((champion) => names.includes(champion.name)).map((champion) => champion.id)
    const sword = components.find((item) => item.nameEn === 'B.F. Sword')?.id
    const bow = components.find((item) => item.nameEn === 'Recurve Bow')?.id
    const belt = components.find((item) => item.nameEn === "Giant's Belt")?.id
    setInput({ level: 4, ownedChampionIds: owned, components: [sword, bow, belt].filter((id): id is string => Boolean(id)) })
  }

  return (
    <div className="coach-layout">
      <aside className="coach-inputs">
        <div className="input-intro">
          <div className="live-dot"><i /> COACH INPUT</div>
          <h2>Bạn đang có gì?</h2>
          <p>Chọn đúng đồ và tướng rơi. Engine sẽ ưu tiên giữ máu trước, rồi mới tính đường pivot vào bài meta.</p>
          <button type="button" className="preset-button" onClick={loadExample}><Sparkles size={15} /> Nạp ví dụ đầu game</button>
        </div>

        <section className="input-section level-section">
          <div className="section-heading-row"><div><span className="step-badge">00</span><h3>Level hiện tại</h3></div></div>
          <div className="level-selector">
            {[3, 4, 5, 6].map((level) => (
              <button key={level} type="button" className={input.level === level ? 'active' : ''} onClick={() => setInput((state) => ({ ...state, level }))}>
                <span>Lv.</span>{level}
              </button>
            ))}
          </div>
        </section>

        <ItemPicker components={components} selectedIds={input.components} onChange={(ids) => setInput((state) => ({ ...state, components: ids }))} />
        <ChampionPicker champions={setData.champions} selectedIds={input.ownedChampionIds} onChange={(ids) => setInput((state) => ({ ...state, ownedChampionIds: ids }))} />
      </aside>

      <main className="coach-output">
        <div className="coach-hero">
          <div>
            <span className="eyebrow">QUYẾT ĐỊNH NGAY · SET 18 · PATCH {apiResult?.data.patch ?? metaData.patch}</span>
            <h1>Def khỏe bây giờ, <em>pivot đúng</em> về sau.</h1>
            <p>Board live + TensorFlow xếp hạng đường chuyển bài, đồng thời chỉ rõ món nào nên slam, ai cầm tạm và carry nào nhận lại đồ ở từng giai đoạn.</p>
            <div className={`engine-status ${apiState}`}>
              <i />
              {apiState === 'live'
                ? `LIVE · ${apiResult?.data.trainingSamples?.toLocaleString('vi-VN') ?? 0} samples · ${apiResult?.data.crossSourceRows?.toLocaleString('vi-VN') ?? 0} cross-source · ${apiResult?.data.clusters ?? 0} clusters`
                : apiState === 'loading' ? 'Đang tính bằng backend ML…' : 'Backend offline · dùng solver local fallback'}
            </div>
          </div>
          <div className="confidence-card">
            <span>Hướng tốt nhất</span>
            <strong>{top?.name ?? 'Đang tính...'}</strong>
            <div className="confidence-meter"><i style={{ width: `${Math.min(100, Math.round(top?.confidence ?? top?.score ?? 0))}%` }} /></div>
            <small>
              {top?.confidence !== undefined
                ? `Tin cậy ${Math.round(top.confidence)}%${top.crossSource ? ' · cross-check nhiều nguồn' : ''}${top.componentFit !== undefined ? ` · đồ fit ${Math.round(top.componentFit)}%` : ''}`
                : top?.matchReasons[0] ?? 'Dựa trên meta score + trait graph'}
            </small>
          </div>
        </div>

        <section className="decision-strip">
          <article>
            <Shield size={18} />
            <div><span>Def hiện tại</span><strong>{earlyTraits.slice(0, 3).map((trait) => `${trait.trait.name} ${trait.activeBreakpoint}`).join(' · ') || 'Board cân bằng'}</strong></div>
          </article>
          <article>
            <Sword size={18} />
            <div><span>Ghép ngay</span><strong>{firstSlamItem ? `${firstSlamItem.name}${firstSlamHolder ? ` → ${firstSlamHolder.name}` : ''}` : 'Giữ đồ linh hoạt'}</strong></div>
          </article>
          <article>
            <Target size={18} />
            <div><span>Bắt trong shop</span><strong>{buyNext.slice(0, 3).map((champion) => champion.name).join(' · ') || 'Tướng nâng cấp'}</strong></div>
          </article>
          <article>
            <TrendingUp size={18} />
            <div><span>Đích đến</span><strong>{top?.leveling ?? 'Flex theo lobby'}</strong></div>
          </article>
        </section>

        {top?.transitionPath && top.transitionPath.length > 1 && (
          <section className="transition-route" aria-label="Lộ trình chuyển đội hình">
            <div className="transition-title"><span>TRANSITION PATH</span><strong>Giữ board mạnh, giảm số lần thay quân</strong></div>
            <div className="transition-steps">
              {top.transitionPath.slice(0, 7).map((step, index) => (
                <div className="transition-step" key={`${step.level}-${index}`}>
                  <b>Lv.{step.level}</b>
                  <span>{step.board.slice(0, 4).map((unit) => unit.name).join(' · ')}{step.board.length > 4 ? '…' : ''}</span>
                  {step.games > 0 && <small>{step.games} mẫu{step.avgPlacement ? ` · avg ${step.avgPlacement.toFixed(2)}` : ''}</small>}
                </div>
              ))}
            </div>
          </section>
        )}

        <BoardCanvas champions={earlyBoard} title={`Board def level ${input.level}`} items={openerBoardItems} />

        <section className="panel item-roadmap-panel">
          <div className="panel-title">
            <div><span className="eyebrow">ITEM ROUTE · LIVE HOLDER DATA</span><h2>Ghép món nào, cho ai cầm, lúc nào chuyển?</h2></div>
            <Flame size={20} />
          </div>
          {usingApi && (stageRows.length || bisRows.length) ? (
            <StageItemPlan rows={stageRows} bis={bisRows} itemById={itemById} championById={championById} components={components} />
          ) : fallback.itemSuggestions.length ? (
            <div className="item-suggestions">
              {fallback.itemSuggestions.slice(0, 4).map((suggestion, index) => (
                <article className="ranked-item" key={suggestion.item.id}>
                  <span className="rank-number">0{index + 1}</span>
                  <ItemMini item={suggestion.item} components={components} />
                  <p>{suggestion.reason}</p>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty-state"><Sword size={24} /><strong>Chọn ít nhất 2 mảnh đồ</strong><span>Engine sẽ tối ưu recipe không tranh mảnh, holder đầu game và đích chuyển đồ cuối game.</span></div>
          )}
        </section>

        <section className="panel shop-plan-panel">
          <div className="panel-title"><div><span className="eyebrow">SHOP PLAN</span><h2>Tướng nên bắt tiếp</h2></div><Target size={20} /></div>
          <div className="buy-list">
            {buyNext.map((champion, index) => <ChampionMini key={champion.id} champion={champion} label={index < 3 ? 'Ưu tiên cao' : 'Giữ nếu dư bench'} />)}
          </div>
        </section>

        <section className="panel pivot-panel">
          <div className="panel-title">
            <div><span className="eyebrow">PIVOT RANKING</span><h2>Đội hình hướng tới</h2></div>
            <span className="source-pill">Live placement + opener fit + TensorFlow</span>
          </div>
          <div className="pivot-list">
            {displayComps.slice(0, 5).map((recommendation, index) => (
              <article key={recommendation.id} className={`pivot-card ${index === 0 ? 'best' : ''}`}>
                <div className="pivot-rank"><b>#{index + 1}</b><span className={`tier tier-${recommendation.tier.toLowerCase()}`}>{recommendation.tier}</span></div>
                <div className="pivot-main">
                  <h3>{recommendation.name}</h3>
                  <p>{recommendation.matchReasons.slice(0, 2).join(' · ') || 'Meta score cao, có đường ghép trait ổn định.'}</p>
                  <div className="carry-row">
                    {recommendation.carries.map((champion) => <img key={champion.id} src={champion.image} title={champion.name} alt={champion.name} />)}
                  </div>
                </div>
                <div className="pivot-traits">
                  {recommendation.activeTraits.slice(0, 5).map((trait) => (
                    <span key={trait.trait.id}><img src={trait.trait.image} alt="" />{trait.trait.name} {trait.activeBreakpoint}</span>
                  ))}
                </div>
                <div className="pivot-score"><strong>{Math.round(recommendation.score)}</strong><span>fit score</span><small>{recommendation.confidence !== undefined ? `tin cậy ${Math.round(recommendation.confidence)}%${recommendation.componentFit !== undefined ? ` · đồ ${Math.round(recommendation.componentFit)}%` : ''}` : recommendation.avgPlacement ? `avg ${recommendation.avgPlacement.toFixed(2)} · ${recommendation.games ?? 0} mẫu` : recommendation.leveling}</small></div>
              </article>
            ))}
          </div>
        </section>

        {top && <BoardCanvas champions={top.board} title={`Đích đến · ${top.name}`} compact />}
      </main>
    </div>
  )
}

function MetaView() {
  const championsByName = useMemo(() => new Map(setData.champions.map((champion) => [champion.name, champion])), [])
  return (
    <main className="page-shell">
      <header className="page-hero"><span className="eyebrow">META LIVE · PATCH {metaData.patch}</span><h1>Đội hình mạnh mùa 18</h1><p>Cluster và placement lấy từ Set 18 live/current. Snapshot PBE cũ không còn tham gia meta ranking hoặc TensorFlow training.</p></header>
      <div className="meta-grid">
        {metaData.comps.map((comp, index) => (
          <article className="meta-card" key={comp.id}>
            <div className="meta-card-top"><span className={`tier tier-${comp.tier.toLowerCase()}`}>{comp.tier}</span><small>#{index + 1}</small></div>
            <h2>{comp.name}</h2>
            <p>{comp.leveling}</p>
            <div className="meta-carries">
              {comp.carries.map((name) => {
                const champion = championsByName.get(name)
                return champion ? <ChampionMini key={name} champion={champion} /> : null
              })}
            </div>
            <footer><span>{comp.games ? `${comp.games} mẫu · avg ${comp.avgPlacement?.toFixed(2)}` : 'Live score'}</span><strong>{Math.round(comp.metaScore * 100)}%</strong></footer>
          </article>
        ))}
      </div>
    </main>
  )
}

function LibraryView() {
  const [query, setQuery] = useState('')
  const [tab, setTab] = useState<'champions' | 'traits'>('champions')
  const normalized = query.trim().toLowerCase()
  const champions = setData.champions.filter((champion) => `${champion.name} ${champion.traitNames.join(' ')}`.toLowerCase().includes(normalized))
  const traits = setData.traits.filter((trait) => `${trait.name} ${trait.nameEn}`.toLowerCase().includes(normalized))
  return (
    <main className="page-shell">
      <header className="page-hero"><span className="eyebrow">SET 18 DATABASE</span><h1>Tướng & tộc/hệ</h1><p>{setData.champions.length} tướng · {setData.traits.length} tộc/hệ, kèm ảnh, chỉ số và mốc kích hoạt.</p></header>
      <div className="library-toolbar">
        <div className="segmented"><button type="button" className={tab === 'champions' ? 'active' : ''} onClick={() => setTab('champions')}>Tướng</button><button type="button" className={tab === 'traits' ? 'active' : ''} onClick={() => setTab('traits')}>Tộc/Hệ</button></div>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Tìm tên, tộc/hệ..." />
      </div>
      {tab === 'champions' ? (
        <div className="champion-library">
          {champions.map((champion) => (
            <article key={champion.id} className={`unit-card cost-${champion.cost}`}>
              <div className="unit-art"><img src={champion.image} alt={champion.name} /><span>{costLabels[champion.cost]}</span></div>
              <div className="unit-info"><h3>{champion.name}</h3><p>{champion.traitNames.join(' · ')}</p><div className="stat-line"><span>HP <b>{champion.stats.hp}</b></span><span>AD <b>{champion.stats.damage}</b></span><span>AS <b>{champion.stats.attackSpeed}</b></span><span>Mana <b>{champion.stats.initialMana}/{champion.stats.mana}</b></span></div><small>{champion.role}</small></div>
            </article>
          ))}
        </div>
      ) : (
        <div className="trait-library">
          {traits.map((trait) => (
            <article key={trait.id} className="trait-card"><img src={trait.image} alt="" /><div><h3>{trait.name} <small>{trait.nameEn}</small></h3><div className="breakpoint-row">{trait.breakpoints.map((point) => <b key={point}>{point}</b>)}</div><p>{trait.description.slice(0, 260)}{trait.description.length > 260 ? '…' : ''}</p></div></article>
          ))}
        </div>
      )}
    </main>
  )
}

function ItemsView() {
  const components = setData.items.filter((item) => item.category === 'component')
  const completed = setData.items.filter((item) => item.category === 'completed')
  return (
    <main className="page-shell">
      <header className="page-hero"><span className="eyebrow">ITEM DATABASE</span><h1>Đồ ghép mùa 18</h1><p>Hiển thị recipe trực tiếp để bạn nhìn một lần là biết hai mảnh nào tạo ra món gì.</p></header>
      <div className="items-library">
        {completed.map((item) => <article className="item-card" key={item.id}><ItemMini item={item} components={components} /><div className="tag-row">{item.tags.filter((tag) => tag !== 'craftable').map((tag) => <span key={tag}>{tag}</span>)}</div></article>)}
      </div>
    </main>
  )
}

function SourcesView() {
  return (
    <main className="page-shell sources-page">
      <header className="page-hero"><span className="eyebrow">DATA PIPELINE</span><h1>Nguồn dữ liệu & cách chấm điểm</h1><p>Không trộn tất cả nguồn thành “một con số bí mật”. Mỗi nguồn có vai trò riêng: game data là sự thật cấu trúc; match history là thống kê; guide/high-Elo là tín hiệu định tính.</p></header>
      <div className="source-grid">
        {[...setData.sources.map((source) => ({ ...source, note: source.type === 'game-data' ? 'Tên tiếng Việt, item ID, trait ID và dữ liệu client.' : 'Chuẩn hóa dữ liệu Unreal Set 18 và asset.' })), ...metaData.sources].map((source) => (
          <article className="source-card" key={`${source.id}-${source.name}`}><Database size={22} /><div><h3>{source.name}</h3><p>{source.note}</p><a href={source.url} target="_blank" rel="noreferrer">Mở nguồn <ExternalLink size={13} /></a></div></article>
        ))}
      </div>
      <section className="algorithm-panel">
        <h2>Recommendation engine hybrid</h2>
        <div className="algorithm-grid">
          <article><b>1</b><h3>Observed live candidates</h3><p>Ưu tiên board opener/final đã xuất hiện ở patch live; beam search chỉ làm fallback khi level hoặc opener quá lạ.</p></article>
          <article><b>2</b><h3>TensorFlow ensemble</h3><p>3 model độc lập cùng rank board/item; bất đồng giữa model làm giảm confidence thay vì che giấu uncertainty.</p></article>
          <article><b>3</b><h3>Stage-aware item solver</h3><p>Tối ưu multiset mảnh đồ không dùng trùng nguyên liệu, chấm holder live ở opener/mid/late và chỉ rõ đường chuyển đồ về carry cuối.</p></article>
          <article><b>4</b><h3>Cross-source + high Elo</h3><p>MetaTFT aggregate, pro live nhiều region và OP.GG được cân theo freshness/evidence; board cùng xuất hiện ở nhiều nguồn được tăng reliability.</p></article>
        </div>
      </section>
    </main>
  )
}

export default function App() {
  const [view, setView] = useState<View>('coach')
  const nav = [
    { id: 'coach' as const, label: 'Coach', icon: LayoutDashboard },
    { id: 'meta' as const, label: 'Meta', icon: BarChart3 },
    { id: 'library' as const, label: 'Tướng & Tộc/Hệ', icon: BookOpen },
    { id: 'items' as const, label: 'Trang bị', icon: Sword },
    { id: 'sources' as const, label: 'Dữ liệu', icon: Database },
  ]
  return (
    <div className="app-shell">
      <header className="topbar">
        <button type="button" className="brand" onClick={() => setView('coach')}>
          <span className="brand-mark"><Sparkles size={18} /></span>
          <span><strong>HexCoach</strong><small>TFT Mùa 18</small></span>
        </button>
        <nav>
          {nav.map((item) => { const Icon = item.icon; return <button key={item.id} type="button" className={view === item.id ? 'active' : ''} onClick={() => setView(item.id)}><Icon size={16} />{item.label}</button> })}
        </nav>
        <div className="patch-status"><RefreshCw size={14} /><div><b>{metaData.patch} · Set 18</b><span>data {formatAge(setData.generatedAt)}</span></div></div>
      </header>
      {view === 'coach' && <CoachView />}
      {view === 'meta' && <MetaView />}
      {view === 'library' && <LibraryView />}
      {view === 'items' && <ItemsView />}
      {view === 'sources' && <SourcesView />}
      <footer className="site-footer">HexCoach là công cụ fan-made, không được Riot Games bảo trợ. Recommendation mặc định chỉ dùng Set 18 live/current; dữ liệu PBE không tham gia training.</footer>
    </div>
  )
}
