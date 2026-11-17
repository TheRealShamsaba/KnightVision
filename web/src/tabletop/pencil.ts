// Everything drawn by hand: red-pencil arrows on the board, and the tracing-paper sheet with the
// Bauhaus pieces sketched flat (white in outline, black filled with ink).
import type { Candidate } from '../data'
import { INK, RED, PAPER } from './textures'

type Ctx = CanvasRenderingContext2D
export type Placed = { square: string; type: string; white: boolean }

export function parseFen(fen: string): Placed[] {
  const out: Placed[] = []
  fen.split(' ')[0].split('/').forEach((row, r) => {
    let file = 0
    for (const ch of row) {
      if (/\d/.test(ch)) { file += Number(ch); continue }
      out.push({ square: `${'abcdefgh'[file]}${8 - r}`, type: ch.toLowerCase(), white: ch !== ch.toLowerCase() })
      file++
    }
  })
  return out
}
type At = (square: string) => [number, number]

// Seeded, so the wobble in a line is the same on every load.
let seed = 11
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)

/** A slightly bowed line laid down in a few soft passes, as a pencil would. */
function pencilLine(ctx: Ctx, x1: number, y1: number, x2: number, y2: number, width: number) {
  const len = Math.hypot(x2 - x1, y2 - y1), nx = -(y2 - y1) / len, ny = (x2 - x1) / len
  const bow = (rand() - 0.5) * len * 0.05
  ctx.lineCap = 'round'
  for (let pass = 0; pass < 3; pass++) {
    ctx.lineWidth = width * (1 - pass * 0.22)
    ctx.globalAlpha = 0.45
    ctx.beginPath()
    for (let i = 0; i <= 10; i++) {
      const t = i / 10, j = (rand() - 0.5) * width * 0.25, b = Math.sin(Math.PI * t) * bow
      ctx.lineTo(x1 + (x2 - x1) * t + nx * (b + j), y1 + (y2 - y1) * t + ny * (b + j))
    }
    ctx.stroke()
  }
  ctx.globalAlpha = 1
}

export function pencilArrow(ctx: Ctx, from: [number, number], to: [number, number], width: number, label?: string) {
  const a = Math.atan2(to[1] - from[1], to[0] - from[0])
  const head = Math.max(30, width * 2.4)
  ctx.strokeStyle = RED
  pencilLine(ctx, from[0] + Math.cos(a) * 18, from[1] + Math.sin(a) * 18, to[0], to[1], width)
  for (const side of [-1, 1]) {
    const b = a + Math.PI + side * 0.5
    pencilLine(ctx, to[0], to[1], to[0] + Math.cos(b) * head, to[1] + Math.sin(b) * head, Math.max(3, width * 0.8))
  }
  if (!label) return
  ctx.fillStyle = RED
  ctx.font = 'italic 28px Literata, serif'
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  ctx.fillText(label, to[0] + Math.cos(a) * 46, to[1] + Math.sin(a) * 46 - 8)
}

// Each glyph is a front view of the wooden piece: [kind, x, y, w, h, turn] in square units, y up from the base.
type Op = ['rect' | 'ball', number, number, number, number, number?]
const GLYPHS: Record<string, Op[]> = {
  p: [['rect', 0, 0.15, 0.3, 0.3]],
  r: [['rect', 0, 0.25, 0.5, 0.5]],
  b: [['rect', 0, 0.05, 0.28, 0.1], ['rect', 0, 0.38, 0.12, 0.66, Math.PI / 4], ['rect', 0, 0.38, 0.12, 0.66, -Math.PI / 4]],
  n: [['rect', 0, 0.09, 0.5, 0.18], ['rect', -0.16, 0.43, 0.18, 0.5]],
  q: [['rect', 0, 0.2, 0.4, 0.4], ['ball', 0, 0.6, 0.2, 0.2]],
  k: [['rect', 0, 0.22, 0.44, 0.44], ['rect', 0, 0.58, 0.26, 0.26, Math.PI / 4]],
}

function glyph(ctx: Ctx, type: string, white: boolean, cx: number, base: number, s: number) {
  ctx.lineWidth = s * 0.03
  ctx.strokeStyle = INK
  ctx.fillStyle = white ? PAPER : INK
  for (const [kind, x, y, w, h, turn = 0] of GLYPHS[type]) {
    ctx.save()
    ctx.translate(cx + x * s * (type === 'n' && !white ? -1 : 1), base - y * s)
    ctx.rotate(turn)
    ctx.beginPath()
    if (kind === 'ball') ctx.arc(0, 0, w * s, 0, Math.PI * 2)
    else ctx.rect(-w * s / 2, -h * s / 2, w * s, h * s)
    ctx.fill(); ctx.stroke()
    ctx.restore()
  }
}

/** The board in graphite: grid, hatched dark squares, coordinates, pieces. Returns a square locator. */
function drawBoard(ctx: Ctx, fen: string, ox: number, oy: number, s: number, swap?: Record<string, string>): At {
  ctx.strokeStyle = 'rgba(30, 27, 23, 0.6)'
  ctx.lineWidth = 2
  for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
    if ((r + f) % 2 === 0) continue
    ctx.save()
    ctx.beginPath(); ctx.rect(ox + f * s, oy + r * s, s, s); ctx.clip()
    ctx.globalAlpha = 0.22
    for (let d = -s; d < s; d += 9) { ctx.beginPath(); ctx.moveTo(ox + f * s + d, oy + r * s + s); ctx.lineTo(ox + f * s + d + s, oy + r * s); ctx.stroke() }
    ctx.restore()
  }
  ctx.strokeRect(ox, oy, s * 8, s * 8)
  ctx.fillStyle = 'rgba(30, 27, 23, 0.55)'
  ctx.font = 'italic 22px Literata, serif'
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  for (let i = 0; i < 8; i++) {
    ctx.fillText('abcdefgh'[i], ox + (i + 0.5) * s, oy + 8 * s + 22)
    ctx.fillText(String(8 - i), ox - 22, oy + (i + 0.5) * s)
  }
  const at: At = sq => [ox + ('abcdefgh'.indexOf(sq[0]) + 0.5) * s, oy + (8 - Number(sq[1]) + 0.5) * s]
  for (const p of parseFen(fen)) {
    const [x, y] = at(p.square)
    glyph(ctx, swap ? swap[p.type] : p.type, p.white, x, y + s * 0.36, s * 0.92)
  }
  return at
}

function canvas(size: number): [HTMLCanvasElement, Ctx] {
  const c = document.createElement('canvas')
  c.width = c.height = size
  return [c, c.getContext('2d')!]
}

/** Red-pencil candidate arrows on a transparent sheet the size of the board (rank 8 at the top). */
export function arrowSheet(candidates: Candidate[]): HTMLCanvasElement {
  const [c, ctx] = canvas(1024)
  const at = (sq: string): [number, number] => [('abcdefgh'.indexOf(sq[0]) + 0.5) * 128, (8 - Number(sq[1]) + 0.5) * 128]
  // Weakest first, so the strongest belief is drawn on top.
  for (const m of [...candidates].reverse()) {
    pencilArrow(ctx, at(m.uci.slice(0, 2)), at(m.uci.slice(2, 4)), 9 + m.p * 56)
  }
  return c
}

/** One side of the tracing paper: the sheet's own drawing plus the other side faintly showing through, upside down. */
export function tracingSides(
  front: { fen: string; title: string; move: Candidate },
  back: { fen: string; title: string; move: Candidate; swap: Record<string, string> },
): [HTMLCanvasElement, HTMLCanvasElement] {
  const size = 1024, s = size / 9.6, o = s * 0.8
  const ink = (side: typeof front, swap?: Record<string, string>) => {
    const [c, ctx] = canvas(size)
    const at = drawBoard(ctx, side.fen, o, o, s, swap)
    ctx.fillStyle = INK
    ctx.font = '500 34px Jost, sans-serif'
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle'
    ctx.fillText(side.title, o - 6, o * 0.45)
    pencilArrow(ctx, at(side.move.uci.slice(0, 2)), at(side.move.uci.slice(2, 4)), 8 + side.move.p * 40,
      `${side.move.san} ${Math.round(side.move.p * 100)}%`)
    return c
  }
  const layers = [ink(front), ink(back, back.swap)]
  return layers.map((mine, i) => {
    const [c, ctx] = canvas(size)
    ctx.fillStyle = '#F6F2EA'
    ctx.fillRect(0, 0, size, size)
    ctx.save()
    ctx.globalAlpha = 0.12
    ctx.translate(0, size); ctx.scale(1, -1)
    ctx.drawImage(layers[1 - i], 0, 0)
    ctx.restore()
    ctx.drawImage(mine, 0, 0)
    return c
  }) as [HTMLCanvasElement, HTMLCanvasElement]
}
