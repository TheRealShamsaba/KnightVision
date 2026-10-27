// 2D riso printing: every drawing is split into ink layers, each layer gets the specks a rubber
// stamp never inks, and the layers are overprinted (multiply) a pixel or two out of register.
import type { Candidate } from '../data'
import { parseFen, FILES } from '../world/board3d'

export const INK = { orange: '#FF6C2F', purple: '#765BA7', blue: '#3D5588', black: '#111111' } as const
export type Ink = keyof typeof INK
// Fixed per-drum misregistration in CSS px, like a real riso that has drifted.
const REG: Record<Ink, [number, number]> = { orange: [0, 0], purple: [1.3, -0.9], blue: [-1, 1.2], black: [0.5, 0.7] }

export function rng(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export const dpr = (): number => Math.min(window.devicePixelRatio || 1, 2)

export function sizeCanvas(c: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D {
  const d = dpr()
  c.width = Math.round(w * d); c.height = Math.round(h * d)
  c.style.width = `${w}px`; c.style.height = `${h}px`
  const g = c.getContext('2d')!
  g.setTransform(d, 0, 0, d, 0, 0)
  return g
}

// Piece silhouettes in a 100-unit box, base on y = 86. Hand-drawn so they read like carved rubber.
const PATHS: Record<string, string> = {
  p: 'M50 15a13 13 0 1 1 0 26a13 13 0 1 1 0-26Z M41 44h18l5 30h8v12H28V74h8Z',
  n: 'M22 86V74h12c0-12 6-20 14-26l-12 3-10-8 18-20 3-9 5 8c14 5 22 19 22 40v12h4v12Z M42 30a3 3 0 1 0 6 0a3 3 0 1 0-6 0Z',
  b: 'M50 5a6 6 0 1 1 0 12a6 6 0 1 1 0-12Z M50 19C64 29 68 45 59 58H41C32 45 36 29 50 19Z M53 29l4 3-9 13-4-3Z M41 61h18l4 13h9v12H28V74h9Z',
  r: 'M26 14h10v8h8v-8h12v8h8v-8h10v20l-8 6v34h8v12H26V74h8V40l-8-6Z',
  q: 'M20 28l12 36h36l12-36-15 17-7-27-8 24-8-24-7 27Z M30 67h40l4 7h6v12H20V74h6Z' +
    [[20, 20], [42, 10], [58, 10], [80, 20]].map(([x, y]) => ` M${x - 4} ${y}a4 4 0 1 0 8 0a4 4 0 1 0-8 0Z`).join(''),
  k: 'M47 4h6v6h6v6h-6v8h-6v-8h-6v-6h6Z M28 32c12-7 32-7 44 0l-7 32H35Z M31 67h38l4 7h6v12H21V74h6Z',
}
const paths = new Map<string, Path2D>()

/** Draw a piece with its box's top-left at (x, y). `outline` = carved outline; `grow` fattens it (for knock-outs). */
export function drawPiece(g: CanvasRenderingContext2D, type: string, x: number, y: number, size: number, outline = false, grow = 0): void {
  let p = paths.get(type)
  if (!p) paths.set(type, (p = new Path2D(PATHS[type])))
  g.save(); g.translate(x, y); g.scale(size / 100, size / 100); g.lineJoin = 'round'
  if (grow) { g.lineWidth = grow; g.stroke(p) }
  if (outline) { g.lineWidth = 7; g.stroke(p) } else g.fill(p, 'evenodd')
  g.restore()
}

/** One drum's worth of ink on its own transparent sheet. */
export function inkLayer(w: number, h: number, ink: Ink, seed: number, draw: (g: CanvasRenderingContext2D) => void): HTMLCanvasElement {
  const c = document.createElement('canvas')
  const g = sizeCanvas(c, w, h)
  g.fillStyle = g.strokeStyle = INK[ink]
  draw(g)
  const r = rng(seed)
  g.globalCompositeOperation = 'destination-out'
  for (let i = 0, n = Math.round((w * h) / 45); i < n; i++) {
    g.globalAlpha = 0.25 + r() * 0.75
    g.beginPath(); g.arc(r() * w, r() * h, 0.3 + r() * r() * 1.3, 0, 7); g.fill()
  }
  // Uneven pressure: broad, faint patches where the stamp rocked.
  g.globalAlpha = 1
  for (let i = 0; i < 5; i++) {
    const x = r() * w, y = r() * h, rad = (0.15 + r() * 0.3) * Math.max(w, h)
    const grd = g.createRadialGradient(x, y, 0, x, y, rad)
    grd.addColorStop(0, 'rgba(0,0,0,.22)'); grd.addColorStop(1, 'rgba(0,0,0,0)')
    g.fillStyle = grd; g.fillRect(0, 0, w, h)
  }
  return c
}

export function overprint(g: CanvasRenderingContext2D, layers: [HTMLCanvasElement, Ink][]): void {
  const d = dpr()
  g.save(); g.setTransform(1, 0, 0, 1, 0, 0)
  g.clearRect(0, 0, g.canvas.width, g.canvas.height)
  g.globalCompositeOperation = 'multiply'
  for (const [layer, ink] of layers) g.drawImage(layer, REG[ink][0] * d, REG[ink][1] * d)
  g.restore()
}

/** A riso tint: a rotated screen of dots (area = coverage) clipped to a rectangle. */
export function halftone(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, coverage: number, cell: number, deg: number): void {
  const a = (deg * Math.PI) / 180, cos = Math.cos(a), sin = Math.sin(a)
  const r = cell * Math.sqrt(coverage / Math.PI)
  const us: number[] = [], vs: number[] = []
  for (const [px, py] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) { us.push(px * cos + py * sin); vs.push(-px * sin + py * cos) }
  g.save(); g.beginPath(); g.rect(x, y, w, h); g.clip(); g.beginPath()
  for (let u = Math.floor(Math.min(...us) / cell) * cell; u <= Math.max(...us) + cell; u += cell)
    for (let v = Math.floor(Math.min(...vs) / cell) * cell; v <= Math.max(...vs) + cell; v += cell) {
      const px = u * cos - v * sin, py = u * sin + v * cos
      g.moveTo(px + r, py); g.arc(px, py, r, 0, Math.PI * 2)
    }
  g.fill(); g.restore()
}

/** A hand-cut ring: never quite round, with a gap where the stamp lifted. */
export function stampedRing(g: CanvasRenderingContext2D, cx: number, cy: number, rx: number, ry: number, seed: number, width: number): void {
  const r = rng(seed), p1 = r() * 6, p2 = r() * 6, start = r() * 6.28, gap = 0.3 + r() * 0.3
  g.lineCap = 'round'
  for (const [lw, k0] of [[width, 1], [width * 0.55, 1.03]]) {
    g.lineWidth = lw; g.beginPath()
    for (let i = 0; i <= 90; i++) {
      const t = start + (i / 90) * (Math.PI * 2 - gap)
      const k = k0 + 0.05 * Math.sin(t * 2 + p1) + 0.03 * Math.sin(t * 3 + p2)
      const px = cx + Math.cos(t) * rx * k, py = cy + Math.sin(t) * ry * k
      if (i) g.lineTo(px, py); else g.moveTo(px, py)
    }
    g.stroke()
  }
}

function arrow(g: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number, w: number): void {
  const len = Math.hypot(x2 - x1, y2 - y1), head = w * 2.4
  g.save(); g.translate(x1, y1); g.rotate(Math.atan2(y2 - y1, x2 - x1)); g.beginPath()
  g.moveTo(0, -w / 2); g.lineTo(len - head, -w / 2); g.lineTo(len - head, -w * 1.3); g.lineTo(len, 0)
  g.lineTo(len - head, w * 1.3); g.lineTo(len - head, w / 2); g.lineTo(0, w / 2); g.closePath(); g.fill(); g.restore()
}

/** v1's mix-up: planes sent K Q R B N P, read as P N B R Q K. */
export const SWAP: Record<string, string> = { k: 'p', q: 'n', r: 'b', b: 'r', n: 'q', p: 'k' }

/** Duchamp-style board: squares in orange, white pieces purple, black pieces black, arrows in blue. */
export function stampBoard(c: HTMLCanvasElement, fen: string, size: number, seed: number, arrows: Candidate[] = [], swap?: Record<string, string>, flip = false): void {
  const g = sizeCanvas(c, size, size)
  const pad = size * 0.065, sq = (size - 2 * pad) / 8
  // Column and row on the page; `flip` prints the board from Black's side.
  const col = (s: string) => (flip ? 7 - FILES.indexOf(s[0]) : FILES.indexOf(s[0]))
  const row = (s: string) => (flip ? Number(s[1]) - 1 : 8 - Number(s[1]))
  const pieces = [...parseFen(fen)].map(([s, p]) => ({
    x: pad + col(s) * sq, y: pad + row(s) * sq, type: swap ? swap[p.type] : p.type, color: p.color,
  }))
  const centre = (s: string): [number, number] => [pad + (col(s) + 0.5) * sq, pad + (row(s) + 0.5) * sq]
  const orange = inkLayer(size, size, 'orange', seed, (t) => {
    t.lineWidth = Math.max(1.5, size / 220); t.strokeRect(pad - 4, pad - 4, 8 * sq + 8, 8 * sq + 8)
    for (let f = 0; f < 8; f++) for (let r = 0; r < 8; r++) if ((f + r) % 2 === 0) t.fillRect(pad + f * sq, pad + (7 - r) * sq, sq, sq)
    // Knock the orange out under each piece, as a real separation would.
    t.globalCompositeOperation = 'destination-out'
    for (const p of pieces) drawPiece(t, p.type, p.x + sq * 0.09, p.y + sq * 0.08, sq * 0.82, false, 12)
  })
  const piecesOf = (color: 'w' | 'b') => (t: CanvasRenderingContext2D) => {
    for (const p of pieces) if (p.color === color) drawPiece(t, p.type, p.x + sq * 0.09, p.y + sq * 0.08, sq * 0.82)
  }
  const purple = inkLayer(size, size, 'purple', seed + 1, piecesOf('w'))
  const black = inkLayer(size, size, 'black', seed + 2, (t) => {
    piecesOf('b')(t)
    t.font = `${Math.max(9, sq * 0.2)}px "Courier Prime"`; t.textAlign = 'center'
    for (let i = 0; i < 8; i++) {
      t.fillText(FILES[flip ? 7 - i : i], pad + (i + 0.5) * sq, size - pad * 0.25)
      t.fillText(String(flip ? i + 1 : 8 - i), pad * 0.4, pad + (i + 0.58) * sq)
    }
  })
  const blue = inkLayer(size, size, 'blue', seed + 3, (t) => {
    arrows.forEach((m, i) => {
      const [x1, y1] = centre(m.uci.slice(0, 2)), [x2, y2] = centre(m.uci.slice(2, 4))
      t.globalAlpha = i ? 0.8 : 1
      arrow(t, x1, y1, x2, y2, sq * (0.07 + 0.2 * Math.sqrt(m.p)))
    })
  })
  overprint(g, [[orange, 'orange'], [blue, 'blue'], [purple, 'purple'], [black, 'black']])
}

/** One input plane: an 8×8 stamp, inked only where `type` stands (uppercase = white). */
export function pieceSheet(c: HTMLCanvasElement, fen: string, type: string, size: number, seed: number): void {
  const g = sizeCanvas(c, size, size)
  const pad = size * 0.06, sq = (size - 2 * pad) / 8
  const white = type === type.toUpperCase(), t0 = type.toLowerCase()
  const grid = inkLayer(size, size, 'orange', seed, (t) => {
    t.lineWidth = Math.max(0.5, size / 220)
    t.beginPath()
    for (let i = 1; i < 8; i++) { t.moveTo(pad + i * sq, pad); t.lineTo(pad + i * sq, size - pad); t.moveTo(pad, pad + i * sq); t.lineTo(size - pad, pad + i * sq) }
    t.stroke(); t.lineWidth *= 2.4; t.strokeRect(pad, pad, 8 * sq, 8 * sq)
  })
  const marks = inkLayer(size, size, white ? 'purple' : 'blue', seed + 1, (t) => {
    for (const [s, p] of parseFen(fen)) {
      if (p.type !== t0 || (p.color === 'w') !== white) continue
      const x = pad + FILES.indexOf(s[0]) * sq, y = pad + (8 - Number(s[1])) * sq
      t.fillRect(x + 0.5, y + 0.5, sq - 1, sq - 1)
      t.globalCompositeOperation = 'destination-out'
      drawPiece(t, t0, x + sq * 0.17, y + sq * 0.14, sq * 0.66)
      t.globalCompositeOperation = 'source-over'
    }
  })
  overprint(g, [[grid, 'orange'], [marks, white ? 'purple' : 'blue']])
}

/** A bar made of halftone dots: length = probability, dots thinning out at the end like a riso ramp. */
export function dotBar(c: HTMLCanvasElement, p: number, w: number, h: number): void {
  const g = sizeCanvas(c, w, h)
  const pitch = h * 0.5, n = Math.max(1, Math.round((p * w) / pitch))
  g.fillStyle = INK.blue
  for (let i = 0; i < n; i++) {
    const fade = Math.min(1, (n - i) / Math.max(2, n * 0.35))
    for (const row of [0.27, 0.73]) {
      g.beginPath(); g.arc((i + 0.5) * pitch + (row > 0.5 ? pitch / 2 : 0), h * row, pitch * 0.45 * Math.sqrt(fade), 0, 7); g.fill()
    }
  }
}

/** A perforated postage stamp, printed in three inks. */
export function postage(c: HTMLCanvasElement, value: string, unit: string, piece: string, seed: number): void {
  const w = 92, h = 112, g = sizeCanvas(c, w, h)
  g.fillStyle = '#FBF8F1'; g.fillRect(0, 0, w, h)
  const tint = inkLayer(w, h, 'orange', seed, (t) => { halftone(t, 9, 9, w - 18, h - 18, 0.32, 3.4, 15); t.lineWidth = 1.2; t.strokeRect(9, 9, w - 18, h - 18) })
  const fig = inkLayer(w, h, 'purple', seed + 1, (t) => drawPiece(t, piece, 30, 14, 52))
  const type = inkLayer(w, h, 'black', seed + 2, (t) => {
    t.font = '11px "Courier Prime"'; t.fillText(unit, 14, 82)
    t.font = '800 21px "Bricolage Grotesque"'; t.fillText(value, 13, 99)
  })
  g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = 'multiply'
  for (const [l, ink] of [[tint, 'orange'], [fig, 'purple'], [type, 'black']] as [HTMLCanvasElement, Ink][]) g.drawImage(l, REG[ink][0], REG[ink][1])
  g.restore()
  // Perforations last, so they punch through every ink.
  g.globalCompositeOperation = 'destination-out'; g.beginPath()
  for (let x = 4; x < w; x += 8) { g.moveTo(x + 3, 0); g.arc(x, 0, 3, 0, 7); g.moveTo(x + 3, h); g.arc(x, h, 3, 0, 7) }
  for (let y = 4; y < h; y += 8) { g.moveTo(3, y); g.arc(0, y, 3, 0, 7); g.moveTo(w + 3, y); g.arc(w, y, 3, 0, 7) }
  g.fill(); g.globalCompositeOperation = 'source-over'
}
