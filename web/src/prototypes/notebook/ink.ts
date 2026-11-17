// Shared drawing kit: palette, tiny DOM/SVG builders, glyph pieces, arrows, and the one-second clock.
import { FILES } from '../../world/board3d'

export const INK = '#111111'
export const PAPER = '#F6F4EF'
export const GRAPHITE = '#8A8A86'
export const BLUE = '#1F4FD8' // data ink: the network's choice / right
export const RED = '#D9481E' // data ink: the move played / wrong

const NS = 'http://www.w3.org/2000/svg'
type Attrs = Record<string, string | number>

export function s<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Attrs = {}, parent?: Element): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag)
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v))
  parent?.appendChild(node)
  return node
}

export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs = {}, parent?: Element, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v))
  if (text !== undefined) node.textContent = text
  parent?.appendChild(node)
  return node
}

export const reducedMotion = (): boolean => matchMedia('(prefers-reduced-motion: reduce)').matches

/** Square index (a1 = 0) to SVG top-left, rank 8 at the top. */
export const sqX = (sq: number, S: number): number => (sq % 8) * S
export const sqY = (sq: number, S: number): number => (7 - Math.floor(sq / 8)) * S
export const sqIndex = (name: string): number => FILES.indexOf(name[0]) + (Number(name[1]) - 1) * 8
export const sqName = (sq: number): string => `${FILES[sq % 8]}${Math.floor(sq / 8) + 1}`
export const uciSquares = (uci: string): [number, number] => [sqIndex(uci.slice(0, 2)), sqIndex(uci.slice(2, 4))]

// The solid Unicode glyphs (U+265A..F) with a text-style selector; fill and stroke do the colouring,
// so a "white" piece is the same shape filled with paper and outlined in ink.
const GLYPH: Record<string, string> = { k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟' }
export const glyph = (type: string): string => GLYPH[type] + '︎'

export function piece(parent: Element, type: string, white: boolean, cx: number, cy: number, size: number): SVGTextElement {
  const t = s('text', {
    x: cx, y: cy + size * 0.34, 'font-size': size, 'text-anchor': 'middle', class: 'glyph',
    fill: white ? PAPER : INK, stroke: white ? INK : PAPER,
    'stroke-width': size * (white ? 0.035 : 0.09), 'paint-order': white ? 'normal' : 'stroke', 'stroke-linejoin': 'round',
  }, parent)
  t.textContent = glyph(type)
  return t
}

/** A flat arrow as one polygon, so its thickness can carry a number. */
export function arrow(parent: Element, from: number, to: number, width: number, S: number, colour = INK, halo = true): SVGPolygonElement {
  const x1 = sqX(from, S) + S / 2, y1 = sqY(from, S) + S / 2
  const x2 = sqX(to, S) + S / 2, y2 = sqY(to, S) + S / 2
  const len = Math.hypot(x2 - x1, y2 - y1)
  const dx = (x2 - x1) / len, dy = (y2 - y1) / len, nx = -dy, ny = dx
  const head = Math.max(width * 2, S * 0.3), hw = Math.max(width * 1.5, S * 0.17), w = width / 2
  const ex = x2 - dx * S * 0.12, ey = y2 - dy * S * 0.12 // stop just short of the centre
  const bx = ex - dx * head, by = ey - dy * head
  const sx = x1 + dx * S * 0.22, sy = y1 + dy * S * 0.22
  const pts = [[sx + nx * w, sy + ny * w], [bx + nx * w, by + ny * w], [bx + nx * hw, by + ny * hw], [ex, ey],
    [bx - nx * hw, by - ny * hw], [bx - nx * w, by - ny * w], [sx - nx * w, sy - ny * w]]
  return s('polygon', {
    points: pts.map(p => p.map(n => n.toFixed(2)).join(',')).join(' '),
    fill: colour, stroke: halo ? PAPER : 'none', 'stroke-width': halo ? 2.5 : 0, 'paint-order': 'stroke', 'stroke-linejoin': 'round',
  }, parent)
}

// The clock: every animation in the notebook advances on the same one-second tick, like a chess clock.
const listeners = new Set<(n: number) => void>()
let count = 0
let sound = false
let audio: AudioContext | null = null
setInterval(() => {
  if (document.hidden) return
  count++
  if (sound) click()
  listeners.forEach(fn => fn(count))
}, 1000)
export const onTick = (fn: (n: number) => void): (() => void) => { listeners.add(fn); return () => listeners.delete(fn) }

export function setSound(on: boolean): void {
  sound = on
  if (on && !audio) audio = new AudioContext()
}

// A short filtered noise burst: close enough to a clock's escapement.
function click(): void {
  if (!audio) return
  const len = Math.floor(audio.sampleRate * 0.025)
  const buf = audio.createBuffer(1, len, audio.sampleRate)
  const ch = buf.getChannelData(0)
  for (let i = 0; i < len; i++) ch[i] = (Math.random() * 2 - 1) * (1 - i / len) ** 4
  const src = audio.createBufferSource()
  const band = audio.createBiquadFilter()
  const gain = audio.createGain()
  band.type = 'bandpass'; band.frequency.value = 3200; band.Q.value = 4
  gain.gain.value = 0.35
  src.buffer = buf
  src.connect(band).connect(gain).connect(audio.destination)
  src.start()
}

export function caption(fig: Element, n: number, text: string): HTMLElement {
  const cap = h('figcaption', {}, fig)
  // Numbered again in page order by main.ts, so figures can be added or moved freely.
  h('b', { class: 'fig-no' }, cap, `Figure ${n}. `)
  cap.append(text)
  return cap
}

export function fileRankLabels(parent: Element, S: number, offset = 10): void {
  for (let i = 0; i < 8; i++) {
    s('text', { x: i * S + S / 2, y: 8 * S + offset + 4, class: 'axis', 'text-anchor': 'middle' }, parent).textContent = FILES[i]
    s('text', { x: -offset, y: i * S + S / 2 + 4, class: 'axis', 'text-anchor': 'middle' }, parent).textContent = String(8 - i)
  }
}
