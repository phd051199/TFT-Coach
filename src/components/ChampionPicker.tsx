import { Search, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { Champion } from '../types'

type Props = {
  champions: Champion[]
  selectedIds: string[]
  onChange: (ids: string[]) => void
}

export function ChampionPicker({ champions, selectedIds, onChange }: Props) {
  const [query, setQuery] = useState('')
  const [cost, setCost] = useState(0)
  const selected = new Set(selectedIds)
  const visible = useMemo(() => champions.filter((champion) => {
    const matchesCost = cost === 0 || champion.cost === cost
    const haystack = `${champion.name} ${champion.traitNames.join(' ')}`.toLowerCase()
    return matchesCost && haystack.includes(query.trim().toLowerCase())
  }), [champions, cost, query])

  function toggle(id: string) {
    onChange(selected.has(id) ? selectedIds.filter((value) => value !== id) : [...selectedIds, id])
  }

  return (
    <section className="input-section">
      <div className="section-heading-row">
        <div>
          <span className="step-badge">02</span>
          <h3>Tướng đang có</h3>
        </div>
        {selectedIds.length > 0 && (
          <button className="text-button" type="button" onClick={() => onChange([])}><X size={14} /> Xóa</button>
        )}
      </div>
      <div className="search-box"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Tìm tướng / tộc hệ..." /></div>
      <div className="cost-tabs">
        {[0, 1, 2, 3, 4, 5].map((value) => (
          <button key={value} className={cost === value ? 'active' : ''} type="button" onClick={() => setCost(value)}>{value || 'Tất cả'}</button>
        ))}
      </div>
      <div className="champion-grid picker-grid">
        {visible.map((champion) => (
          <button
            key={champion.id}
            type="button"
            className={`champion-tile cost-${champion.cost} ${selected.has(champion.id) ? 'selected' : ''}`}
            onClick={() => toggle(champion.id)}
            title={`${champion.name} · ${champion.traitNames.join(' / ')}`}
          >
            <img src={champion.image} alt="" />
            <span>{champion.name}</span>
            {selected.has(champion.id) && <b>✓</b>}
          </button>
        ))}
      </div>
    </section>
  )
}
