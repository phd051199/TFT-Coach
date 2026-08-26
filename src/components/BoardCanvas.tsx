import { Copy, Check } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Champion, ItemSuggestion } from '../types'

type Props = {
  champions: Champion[]
  title?: string
  items?: ItemSuggestion[]
  compact?: boolean
}

const W = 760
const H = 440
const HEX_R = 48

function hexCenter(row: number, col: number) {
  const xGap = HEX_R * 1.72
  const yGap = HEX_R * 1.48
  return {
    x: 80 + col * xGap + (row % 2 ? xGap / 2 : 0),
    y: 66 + row * yGap,
  }
}

function polygon(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.beginPath()
  for (let i = 0; i < 6; i += 1) {
    const angle = Math.PI / 3 * i + Math.PI / 6
    const px = x + Math.cos(angle) * r
    const py = y + Math.sin(angle) * r
    if (i === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  }
  ctx.closePath()
}

function positions(count: number) {
  const preferred = [
    [3, 1], [3, 3], [3, 5], [2, 2], [2, 4], [1, 1], [1, 3], [1, 5], [0, 2], [0, 4],
  ]
  return preferred.slice(0, count)
}

export function BoardCanvas({ champions, title = 'Đội hình', items = [], compact = false }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [copied, setCopied] = useState(false)
  const itemByHolder = useMemo(() => new Map(items.filter((item) => item.holder).map((item) => [item.holder!.id, item.item])), [items])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const ratio = Math.max(1, window.devicePixelRatio || 1)
    canvas.width = W * ratio
    canvas.height = H * ratio
    canvas.style.aspectRatio = `${W} / ${H}`
    ctx.scale(ratio, ratio)

    const gradient = ctx.createLinearGradient(0, 0, W, H)
    gradient.addColorStop(0, '#0d2728')
    gradient.addColorStop(0.55, '#101c25')
    gradient.addColorStop(1, '#151826')
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, W, H)

    for (let row = 0; row < 4; row += 1) {
      for (let col = 0; col < 7; col += 1) {
        const { x, y } = hexCenter(row, col)
        polygon(ctx, x, y, HEX_R - 3)
        ctx.fillStyle = row >= 2 ? 'rgba(32, 108, 92, .15)' : 'rgba(255,255,255,.025)'
        ctx.fill()
        ctx.strokeStyle = row >= 2 ? 'rgba(84, 196, 156, .24)' : 'rgba(255,255,255,.08)'
        ctx.lineWidth = 1
        ctx.stroke()
      }
    }

    let cancelled = false
    const drawUnit = async (champion: Champion, index: number) => {
      const [row, col] = positions(champions.length)[index] ?? [3, index]
      const { x, y } = hexCenter(row, col)
      const image = new Image()
      image.src = champion.image
      try { await image.decode() } catch { return }
      if (cancelled) return
      ctx.save()
      polygon(ctx, x, y, HEX_R - 7)
      ctx.clip()
      ctx.drawImage(image, x - HEX_R, y - HEX_R, HEX_R * 2, HEX_R * 2)
      const fade = ctx.createLinearGradient(0, y, 0, y + HEX_R)
      fade.addColorStop(0.3, 'rgba(0,0,0,0)')
      fade.addColorStop(1, 'rgba(0,0,0,.85)')
      ctx.fillStyle = fade
      ctx.fillRect(x - HEX_R, y, HEX_R * 2, HEX_R)
      ctx.restore()

      const costColors: Record<number, string> = { 1: '#9299a8', 2: '#46be78', 3: '#4a92e3', 4: '#b76ae7', 5: '#e7b955' }
      polygon(ctx, x, y, HEX_R - 6)
      ctx.strokeStyle = costColors[champion.cost] ?? '#ccc'
      ctx.lineWidth = 3
      ctx.stroke()
      ctx.font = '700 13px Inter, system-ui, sans-serif'
      ctx.fillStyle = '#fff'
      ctx.textAlign = 'center'
      ctx.fillText(champion.name, x, y + 31)

      const item = itemByHolder.get(champion.id)
      if (item) {
        const itemImage = new Image()
        itemImage.src = item.image
        try { await itemImage.decode() } catch { return }
        if (cancelled) return
        ctx.drawImage(itemImage, x + 18, y - 40, 24, 24)
        ctx.strokeStyle = '#e7b955'
        ctx.lineWidth = 1
        ctx.strokeRect(x + 18, y - 40, 24, 24)
      }
    }

    champions.forEach((champion, index) => { void drawUnit(champion, index) })
    return () => { cancelled = true }
  }, [champions, itemByHolder])

  async function copyComp() {
    const units = champions.map((champion) => champion.name).join(', ')
    const itemText = items
      .filter((entry) => entry.holder)
      .map((entry) => `${entry.holder!.name}: ${entry.item.name}`)
      .join(' | ')
    await navigator.clipboard.writeText(`${title}\n${units}${itemText ? `\n${itemText}` : ''}`)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1400)
  }

  return (
    <div className={`board-shell ${compact ? 'compact' : ''}`}>
      <div className="board-toolbar">
        <div>
          <span className="eyebrow">BOARD CANVAS</span>
          <strong>{title}</strong>
        </div>
        <button className="ghost-button" onClick={copyComp} type="button">
          {copied ? <Check size={16} /> : <Copy size={16} />}
          {copied ? 'Đã copy' : 'Copy đội hình'}
        </button>
      </div>
      <canvas ref={canvasRef} className="board-canvas" aria-label={`${title}: ${champions.map((champion) => champion.name).join(', ')}`} />
      <div className="sr-only">{champions.map((champion) => champion.name).join(', ')}</div>
    </div>
  )
}
