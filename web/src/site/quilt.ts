// The 4,096 outputs as one halftone sheet: a board of boards. Big blocks = from-square,
// dots inside = to-square, rank 8 at the top of both. Dot area follows the byte (log scale).
import { quiltBytes, percent, type Ply } from '../data'
import { parseFen, FILES } from '../world/board3d'
import { halftone, inkLayer, overprint, sizeCanvas, stampedRing } from './ink'

const NAMES: Record<string, string> = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' }
const square = (i: number): string => `${FILES[i % 8]}${Math.floor(i / 8) + 1}`
const index = (s: string): number => FILES.indexOf(s[0]) + 8 * (Number(s[1]) - 1)
const prob = (v: number): number => Math.pow(10, -7 + (7 * v) / 255)

function odds(v: number): string {
  if (v === 0) return 'one in ten million, or less'
  const p = prob(v)
  return p < 0.001 ? `about one in ${Math.round(1 / p).toLocaleString('en')}` : percent(p)
}

export type Quilt = { draw: () => void; show: (ply: Ply) => void }

export function mountQuilt(fig: HTMLElement, readout: HTMLElement, first: Ply): Quilt {
  let ply = first
  let bytes = quiltBytes(ply)
  let pieces = parseFen(ply.fen)
  const base = fig.querySelector<HTMLCanvasElement>('canvas.ink')!
  const over = fig.querySelector<HTMLCanvasElement>('canvas.hover')!
  const idle = readout.textContent ?? ''
  let gap = 4, pitch = 8, block = 64
  const corner = (from: number): [number, number] => [(from % 8) * (block + gap), (7 - Math.floor(from / 8)) * (block + gap)]
  const dot = (from: number, to: number): [number, number] => {
    const [bx, by] = corner(from)
    return [bx + ((to % 8) + 0.5) * pitch, by + (7 - Math.floor(to / 8) + 0.5) * pitch]
  }

  function draw(): void {
    const W = Math.floor(fig.clientWidth)
    if (!W) return
    gap = W < 480 ? 2 : 4; pitch = (W - 7 * gap) / 64; block = pitch * 8
    const g = sizeCanvas(base, W, W); sizeCanvas(over, W, W)
    const tint = inkLayer(W, W, 'orange', 31, (t) => {
      for (let f = 0; f < 64; f++) {
        if ((f % 8 + Math.floor(f / 8)) % 2) continue // light from-squares stay paper
        const [x, y] = corner(f); halftone(t, x, y, block, block, 0.22, Math.max(2.4, pitch * 0.45), 15)
      }
    })
    const dots = inkLayer(W, W, 'blue', 32, (t) => {
      t.beginPath()
      for (let i = 0; i < bytes.length; i++) {
        const r = pitch * 0.5 * Math.sqrt(bytes[i] / 255)
        if (r < 0.3) continue
        const [x, y] = dot(i >> 6, i & 63); t.moveTo(x + r, y); t.arc(x, y, r, 0, Math.PI * 2)
      }
      t.fill()
    })
    const ring = inkLayer(W, W, 'purple', 33, (t) => {
      const [x, y] = dot(index(ply.uci.slice(0, 2)), index(ply.uci.slice(2, 4)))
      stampedRing(t, x, y, pitch * 2, pitch * 1.8, 5, Math.max(1.6, pitch * 0.3))
      t.font = `700 ${Math.max(12, pitch * 1.7)}px "Courier Prime"`
      // Keep the label on the sheet when the move is near its right edge.
      const label = `${ply.san} ${percent(ply.share)}`
      const right = x + pitch * 2.4 + t.measureText(label).width > W
      t.textAlign = right ? 'right' : 'left'
      t.fillText(label, right ? x - pitch * 2.4 : x + pitch * 2.4, y - pitch * 1.6)
    })
    overprint(g, [[tint, 'orange'], [dots, 'blue'], [ring, 'purple']])
  }

  function hover(e: PointerEvent): void {
    // offsetX/Y are in the canvas's own (unrotated) space, so the card's tilt doesn't throw the aim off.
    const bx = Math.floor(e.offsetX / (block + gap)), by = Math.floor(e.offsetY / (block + gap))
    const ix = Math.floor((e.offsetX - bx * (block + gap)) / pitch), iy = Math.floor((e.offsetY - by * (block + gap)) / pitch)
    const g = over.getContext('2d')!
    g.clearRect(0, 0, over.width, over.height)
    if (bx < 0 || bx > 7 || by < 0 || by > 7 || ix < 0 || ix > 7 || iy < 0 || iy > 7) { readout.textContent = idle; return }
    const from = bx + 8 * (7 - by), to = ix + 8 * (7 - iy)
    const [x, y] = dot(from, to), [cx, cy] = corner(from)
    g.strokeStyle = '#111'; g.lineWidth = 1
    g.strokeRect(cx - 1.5, cy - 1.5, block + 3, block + 3)
    g.beginPath(); g.arc(x, y, pitch * 0.85, 0, 7); g.stroke()
    const p = pieces.get(square(from))
    const who = p ? `${p.color === 'w' ? 'white' : 'black'} ${NAMES[p.type]}` : 'empty square'
    readout.textContent = `${who}, ${square(from)} → ${square(to)}: ${odds(bytes[from * 64 + to])}`
  }
  over.addEventListener('pointermove', hover)
  over.addEventListener('pointerdown', hover)
  over.addEventListener('pointerleave', () => { over.getContext('2d')!.clearRect(0, 0, over.width, over.height); readout.textContent = idle })
  return {
    draw,
    show(next) { ply = next; bytes = quiltBytes(ply); pieces = parseFen(ply.fen); draw() },
  }
}
