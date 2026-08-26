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
import { useMemo, useState } from 'react'
import setRaw from './data/set18.generated.json'
import metaRaw from './data/meta.generated.json'
import { recommend } from './engine/recommend'
import { BoardCanvas } from './components/BoardCanvas'
import { ChampionPicker } from './components/ChampionPicker'
import { ItemPicker } from './components/ItemPicker'
import type { Champion, CoachInput, Item, MetaData, Set18Data } from './types'
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

function CoachView() {
  const [input, setInput] = useState<CoachInput>({ level: 4, ownedChampionIds: [], components: [] })
  const components = useMemo(() => setData.items.filter((item) => item.category === 'component' && !/spatula|pan/i.test(item.nameEn)), [])
  const result = useMemo(() => recommend(input, setData, metaData.comps), [input])
  const top = result.comps[0]

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
            <span className="eyebrow">QUYẾT ĐỊNH NGAY · SET 18</span>
            <h1>Def khỏe bây giờ, <em>pivot đúng</em> về sau.</h1>
            <p>Gợi ý kết hợp tộc/hệ, sức mạnh tướng đầu game, đồ ghép được và hướng meta thay vì bắt bạn force một bài từ 2-1.</p>
          </div>
          <div className="confidence-card">
            <span>Hướng tốt nhất</span>
            <strong>{top?.comp.name ?? 'Đang tính...'}</strong>
            <div className="confidence-meter"><i style={{ width: `${Math.min(100, Math.round((top?.score ?? 0) * 0.85))}%` }} /></div>
            <small>{top?.matchReasons[0] ?? 'Dựa trên meta score + trait graph'}</small>
          </div>
        </div>

        <section className="decision-strip">
          <article>
            <Shield size={18} />
            <div><span>Def hiện tại</span><strong>{result.earlyTraits.slice(0, 3).map((trait) => `${trait.trait.name} ${trait.activeBreakpoint}`).join(' · ') || 'Board cân bằng'}</strong></div>
          </article>
          <article>
            <Sword size={18} />
            <div><span>Ghép ngay</span><strong>{result.itemSuggestions[0]?.item.name ?? 'Giữ đồ linh hoạt'}</strong></div>
          </article>
          <article>
            <Target size={18} />
            <div><span>Bắt trong shop</span><strong>{result.buyNext.slice(0, 3).map((champion) => champion.name).join(' · ') || 'Tướng nâng cấp'}</strong></div>
          </article>
          <article>
            <TrendingUp size={18} />
            <div><span>Đích đến</span><strong>{top?.comp.leveling ?? 'Flex theo lobby'}</strong></div>
          </article>
        </section>

        <BoardCanvas champions={result.earlyBoard} title={`Board def level ${input.level}`} items={result.itemSuggestions.slice(0, 2)} />

        <div className="coach-grid two-col">
          <section className="panel">
            <div className="panel-title"><div><span className="eyebrow">ITEM ENGINE</span><h2>Đồ nên ghép</h2></div><Flame size={20} /></div>
            {result.itemSuggestions.length ? (
              <div className="item-suggestions">
                {result.itemSuggestions.slice(0, 4).map((suggestion, index) => (
                  <article className="ranked-item" key={suggestion.item.id}>
                    <span className="rank-number">0{index + 1}</span>
                    <ItemMini item={suggestion.item} components={components} />
                    <p>{suggestion.reason}</p>
                  </article>
                ))}
              </div>
            ) : (
              <div className="empty-state"><Sword size={24} /><strong>Chọn ít nhất 2 mảnh đồ</strong><span>Tôi sẽ liệt kê toàn bộ món có thể ghép và holder hợp nhất.</span></div>
            )}
          </section>

          <section className="panel">
            <div className="panel-title"><div><span className="eyebrow">SHOP PLAN</span><h2>Tướng nên bắt</h2></div><Target size={20} /></div>
            <div className="buy-list">
              {result.buyNext.map((champion, index) => <ChampionMini key={champion.id} champion={champion} label={index < 3 ? 'Ưu tiên cao' : 'Giữ nếu dư bench'} />)}
            </div>
          </section>
        </div>

        <section className="panel pivot-panel">
          <div className="panel-title">
            <div><span className="eyebrow">PIVOT RANKING</span><h2>Đội hình hướng tới</h2></div>
            <span className="source-pill">Meta + board fit + item fit</span>
          </div>
          <div className="pivot-list">
            {result.comps.slice(0, 4).map((recommendation, index) => (
              <article key={recommendation.comp.id} className={`pivot-card ${index === 0 ? 'best' : ''}`}>
                <div className="pivot-rank"><b>#{index + 1}</b><span className={`tier tier-${recommendation.comp.tier.toLowerCase()}`}>{recommendation.comp.tier}</span></div>
                <div className="pivot-main">
                  <h3>{recommendation.comp.name}</h3>
                  <p>{recommendation.matchReasons.slice(0, 2).join(' · ') || 'Meta score cao, có đường ghép trait ổn định.'}</p>
                  <div className="carry-row">
                    {recommendation.comp.carries.map((name) => {
                      const champion = setData.champions.find((unit) => unit.name === name)
                      return champion ? <img key={name} src={champion.image} title={name} alt={name} /> : null
                    })}
                  </div>
                </div>
                <div className="pivot-traits">
                  {recommendation.activeTraits.slice(0, 5).map((trait) => (
                    <span key={trait.trait.id}><img src={trait.trait.image} alt="" />{trait.trait.name} {trait.activeBreakpoint}</span>
                  ))}
                </div>
                <div className="pivot-score"><strong>{Math.round(recommendation.score)}</strong><span>fit score</span><small>{recommendation.comp.leveling}</small></div>
              </article>
            ))}
          </div>
        </section>

        {top && <BoardCanvas champions={top.board} title={`Đích đến · ${top.comp.name}`} compact />}
      </main>
    </div>
  )
}

function MetaView() {
  const championsByName = useMemo(() => new Map(setData.champions.map((champion) => [champion.name, champion])), [])
  return (
    <main className="page-shell">
      <header className="page-hero"><span className="eyebrow">META SNAPSHOT · PATCH 18.1</span><h1>Đội hình mạnh mùa 18</h1><p>Snapshot khởi đầu tổng hợp từ tier list PBE trước ngày live. Khi Riot API có dữ liệu đủ lớn, collector high-Elo có thể thay trọng số này bằng match history thật.</p></header>
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
            <footer><span>Meta prior</span><strong>{Math.round(comp.metaScore * 100)}%</strong></footer>
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
        <h2>Recommendation engine hiện tại</h2>
        <div className="algorithm-grid">
          <article><b>1</b><h3>Beam search board</h3><p>Duyệt nhiều board ứng viên theo từng slot, thưởng khi chạm mốc trait, đủ frontline/carry và giữ được tướng bạn đã có.</p></article>
          <article><b>2</b><h3>Item compatibility</h3><p>Enumerate toàn bộ món ghép được từ multiset mảnh đồ, sau đó chấm theo role AD/AP/Tank, trait và tempo item.</p></article>
          <article><b>3</b><h3>Meta prior</h3><p>Tier meta chỉ là prior. Nếu opener của bạn lệch mạnh, engine có thể xếp một bài tier thấp hơn lên trên vì chi phí pivot thấp hơn.</p></article>
          <article><b>4</b><h3>High-Elo ready</h3><p>Pipeline đã tách nguồn để thêm Riot Match-V1: placement, top4, item-holder, unit frequency và transition graph theo rank/patch.</p></article>
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
        <div className="patch-status"><RefreshCw size={14} /><div><b>18.1 · Set 18</b><span>data {formatAge(setData.generatedAt)}</span></div></div>
      </header>
      {view === 'coach' && <CoachView />}
      {view === 'meta' && <MetaView />}
      {view === 'library' && <LibraryView />}
      {view === 'items' && <ItemsView />}
      {view === 'sources' && <SourcesView />}
      <footer className="site-footer">HexCoach là công cụ fan-made, không được Riot Games bảo trợ. Dữ liệu Set 18 tự đồng bộ từ nguồn công khai; meta ngày đầu mùa có thể đổi rất nhanh.</footer>
    </div>
  )
}
