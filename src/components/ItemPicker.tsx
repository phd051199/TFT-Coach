import { Minus, Plus } from 'lucide-react'
import type { Item } from '../types'

type Props = {
  components: Item[]
  selectedIds: string[]
  onChange: (ids: string[]) => void
}

export function ItemPicker({ components, selectedIds, onChange }: Props) {
  function count(id: string) { return selectedIds.filter((value) => value === id).length }
  function add(id: string) { onChange([...selectedIds, id]) }
  function remove(id: string) {
    const index = selectedIds.lastIndexOf(id)
    if (index < 0) return
    onChange(selectedIds.filter((_, itemIndex) => itemIndex !== index))
  }

  return (
    <section className="input-section">
      <div className="section-heading-row">
        <div><span className="step-badge">01</span><h3>Đồ rơi đầu game</h3></div>
        <span className="muted-mini">Bấm + nhiều lần nếu trùng</span>
      </div>
      <div className="component-grid">
        {components.map((item) => {
          const amount = count(item.id)
          return (
            <div className={`component-card ${amount ? 'selected' : ''}`} key={item.id}>
              <button type="button" className="item-main" onClick={() => add(item.id)}>
                <img src={item.image} alt="" />
                <span>{item.name}</span>
              </button>
              <div className="counter">
                <button type="button" onClick={() => remove(item.id)} disabled={!amount}><Minus size={12} /></button>
                <b>{amount}</b>
                <button type="button" onClick={() => add(item.id)}><Plus size={12} /></button>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
