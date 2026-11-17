// Figure 3: all 4,096 outputs as a board of boards. Big block = from-square, cell inside = to-square,
// both with rank 8 at the top. Dot radius grows with the logarithm of probability.
import { data, quiltBytes, percent } from '../../data'
import { parseFen, FILES } from '../../world/board3d'
import { INK, PAPER, GRAPHITE, BLUE, RED, s, h, sqName, uciSquares, caption } from './ink'
import { LAST, onSeek, current, scrubber } from './game'

const ML = 14, MT = 14 // room for the from-square axis labels
const byteToP = (v: number): number => 10 ** (-7 + (7 * v) / 255)
const pToByte = (p: number): number => ((Math.log10(p) + 7) * 255) / 7
// A gentle curve so one-in-a-thousand moves still leave a visible mark.
const radius = (v: number, c: number): number => c * 0.5 * (v / 255) ** 1.35
// Decode each ply's base64 once; hovering redraws often.
const decoded = new Map<number, Uint8Array>()
const bytesAt = (i: number): Uint8Array => decoded.get(i) ?? decoded.set(i, quiltBytes(data.opera.plies[i])).get(i)!

function odds(v: number): string {
  if (v === 0) return 'one in ten million or less'
  const p = byteToP(v)
  if (p >= 0.001) return percent(p)
  const n = Number((1 / p).toPrecision(2))
  return `1 in ${n.toLocaleString('en-GB')}`
}

export function quilt(fig: HTMLElement): void {
  const stats = h('div', { class: 'quilt-stats' }, fig)
  const wrap = h('div', { class: 'quilt-wrap' }, fig)
  const box = h('div', { class: 'quilt' }, wrap)
  const loupe = h('div', { class: 'loupe', 'aria-live': 'polite' }, wrap)
  const canvas = h('canvas', { 'aria-label': 'The 4,096 move scores as a 64 by 64 grid' }, box)
  const ctx = canvas.getContext('2d')!
  const readout = h('p', { class: 'readout' }, fig, '')
  scrubber(fig, 'scrub-3')

  // Dot-size key, so the log scale can be read off the page.
  const key = h('div', { class: 'quilt-key' }, fig)
  for (const p of [1e-6, 1e-4, 1e-2, 1]) {
    const item = h('span', {}, key)
    const k = s('svg', { viewBox: '0 0 16 16', 'aria-hidden': 'true' }, item)
    s('circle', { cx: 8, cy: 8, r: Math.max(0.6, radius(pToByte(p), 16)), fill: INK }, k)
    h('span', { class: 'num' }, item, p === 1 ? '1' : `10⁻${String(-Math.log10(p)).replace(/\d/g, d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d])}`)
  }

  let width = 0, c = 0, g = 0, hover: [number, number] | null = null
  const pitch = (): number => 8 * c + g

  const draw = (): void => {
    const i = current()
    const dpr = devicePixelRatio || 1
    const H = MT + 64 * c + 7 * g
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(H * dpr)
    canvas.style.width = `${width}px`; canvas.style.height = `${H}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = PAPER; ctx.fillRect(0, 0, width, H)
    ctx.font = `10px "Fragment Mono", monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = GRAPHITE
    for (let k = 0; k < 8; k++) {
      ctx.fillText(FILES[k], ML + k * pitch() + 4 * c, MT / 2)
      ctx.fillText(String(8 - k), ML / 2, MT + k * pitch() + 4 * c)
    }
    if (i >= LAST) {
      ctx.fillStyle = INK; ctx.font = `15px "Schibsted Grotesk", sans-serif`
      ctx.fillText('Checkmate. Nothing left to score.', ML + (width - ML) / 2, MT + (H - MT) / 2)
      return
    }
    const ply = data.opera.plies[i]
    const bytes = bytesAt(i)
    const board = parseFen(ply.fen)
    const side = ply.fen.split(' ')[1]
    const cellXY = (from: number, to: number): [number, number] => [
      ML + (from % 8) * pitch() + (to % 8) * c,
      MT + (7 - Math.floor(from / 8)) * pitch() + (7 - Math.floor(to / 8)) * c,
    ]
    for (let from = 0; from < 64; from++) {
      const [bx, by] = cellXY(from, 56) // to = a8 is the block's top-left cell
      const pc = board.get(sqName(from))
      // Grey blocks: squares holding a piece of the side to move, where legal moves can start.
      if (pc && pc.color === side) { ctx.fillStyle = '#E4E1D8'; ctx.fillRect(bx, by, 8 * c, 8 * c) }
      ctx.strokeStyle = GRAPHITE; ctx.lineWidth = 0.5; ctx.strokeRect(bx + 0.25, by + 0.25, 8 * c - 0.5, 8 * c - 0.5)
      ctx.fillStyle = INK
      for (let to = 0; to < 64; to++) {
        const r = radius(bytes[from * 64 + to], c)
        if (r < 0.35) continue
        const [x, y] = cellXY(from, to)
        ctx.beginPath(); ctx.arc(x + c / 2, y + c / 2, r, 0, Math.PI * 2); ctx.fill()
      }
    }
    const mark = (uci: string, colour: string, pad: number): void => {
      const [x, y] = cellXY(...uciSquares(uci))
      ctx.strokeStyle = colour; ctx.lineWidth = 1.5
      ctx.strokeRect(x - pad, y - pad, c + 2 * pad, c + 2 * pad)
    }
    mark(ply.top[0].uci, BLUE, 1)
    mark(ply.uci, RED, ply.uci === ply.top[0].uci ? 3 : 1)
    if (hover) {
      const [x, y] = cellXY(...hover)
      ctx.strokeStyle = INK; ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(ML, y + c / 2); ctx.lineTo(x - 2, y + c / 2)
      ctx.moveTo(x + c / 2, MT); ctx.lineTo(x + c / 2, y - 2)
      ctx.stroke()
      ctx.strokeRect(x - 0.5, y - 0.5, c + 1, c + 1)
    }
  }

  const describe = (): void => {
    const i = current()
    stats.innerHTML = ''
    readout.innerHTML = ''
    if (i >= LAST) { readout.textContent = 'The game is over.'; return }
    const ply = data.opera.plies[i]
    statLine(stats, BLUE, 'Its favourite legal move', ply.top[0].san, percent(ply.top[0].p))
    statLine(stats, RED, 'The move played', ply.san, percent(ply.share))
    statLine(stats, null, 'Belief on legal moves at all', '', percent(ply.legalMass))
    if (!hover) { readout.textContent = 'Hover or tap a cell to read it.'; return }
    const [from, to] = hover
    const v = bytesAt(i)[from * 64 + to]
    h('span', { class: 'num' }, readout, `${sqName(from)} → ${sqName(to)}`)
    readout.append('  ')
    readout.append('  raw score ')
    h('span', { class: 'num' }, readout, odds(v))
    const named = ply.top.find(t => t.uci.slice(0, 4) === sqName(from) + sqName(to))
    if (named) readout.append(`  (${named.san})`)
  }

  // The magnifier: one from-square's 64 destinations, large enough to read the numbers.
  const NAMES: Record<string, string> = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' }
  const magnify = (): void => {
    loupe.innerHTML = ''
    const i = current()
    if (i >= LAST) return
    const ply = data.opera.plies[i]
    const [playedFrom, playedTo] = uciSquares(ply.uci)
    const [favFrom, favTo] = uciSquares(ply.top[0].uci)
    const from = hover?.[0] ?? playedFrom
    const pc = parseFen(ply.fen).get(sqName(from))
    const title = h('p', { class: 'loupe-title' }, loupe)
    title.append(`From ${sqName(from)}`)
    h('span', { class: 'dim' }, title, pc ? `, ${pc.color === 'w' ? 'white' : 'black'} ${NAMES[pc.type]}` : ', an empty square')
    const C = 30
    const svg = s('svg', { viewBox: `-14 -4 ${8 * C + 18} ${8 * C + 20}`, role: 'img', 'aria-label': `Where the network would move from ${sqName(from)}` }, loupe)
    const bytes = bytesAt(i)
    for (let to = 0; to < 64; to++) {
      const x = (to % 8) * C, y = (7 - Math.floor(to / 8)) * C
      s('rect', { x, y, width: C, height: C, fill: (to % 8 + Math.floor(to / 8)) % 2 ? PAPER : '#ECE9E1', stroke: '#D6D3CB', 'stroke-width': 0.5 }, svg)
      const v = bytes[from * 64 + to]
      const r = radius(v, C)
      // Dots sit high in the cell so the percentage underneath stays readable.
      if (r >= 0.35) s('circle', { cx: x + C / 2, cy: y + C / 2 - 4, r: Math.min(r * 0.85, C * 0.3), fill: INK }, svg)
      const p = byteToP(v)
      if (p >= 0.005) s('text', { x: x + C / 2, y: y + C - 3, 'text-anchor': 'middle', class: 'loupe-num' }, svg).textContent = p >= 0.095 ? `${Math.round(p * 100)}%` : `${(p * 100).toFixed(1)}`
      const ring = (colour: string, pad: number) => s('rect', { x: x + pad, y: y + pad, width: C - 2 * pad, height: C - 2 * pad, fill: 'none', stroke: colour, 'stroke-width': 1.5 }, svg)
      if (from === favFrom && to === favTo) ring(BLUE, 1.5)
      if (from === playedFrom && to === playedTo) ring(RED, from === favFrom && to === favTo ? 4 : 1.5)
      if (hover && hover[0] === from && hover[1] === to) ring(INK, 0.5)
    }
    s('rect', { x: 0, y: 0, width: 8 * C, height: 8 * C, fill: 'none', stroke: INK, 'stroke-width': 1 }, svg)
    for (let k = 0; k < 8; k++) {
      s('text', { x: k * C + C / 2, y: 8 * C + 13, 'text-anchor': 'middle', class: 'axis' }, svg).textContent = FILES[k]
      s('text', { x: -7, y: k * C + C / 2 + 4, 'text-anchor': 'middle', class: 'axis' }, svg).textContent = String(8 - k)
    }
    h('p', { class: 'loupe-note dim' }, loupe, hover ? 'Numbers are raw percentages of all its belief.' : 'Showing the move played. Hover or tap a block to look inside it.')
  }

  const update = (): void => { draw(); describe(); magnify() }
  const pick = (e: PointerEvent): void => {
    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left - ML, y = e.clientY - rect.top - MT
    const bx = Math.floor(x / pitch()), by = Math.floor(y / pitch())
    const ix = x - bx * pitch(), iy = y - by * pitch()
    const inside = bx >= 0 && bx < 8 && by >= 0 && by < 8 && ix < 8 * c && iy < 8 * c
    const next: [number, number] | null = inside
      ? [(7 - by) * 8 + bx, (7 - Math.floor(iy / c)) * 8 + Math.floor(ix / c)] : null
    if (next?.[0] === hover?.[0] && next?.[1] === hover?.[1]) return
    hover = next
    update()
  }
  canvas.addEventListener('pointermove', pick)
  canvas.addEventListener('pointerdown', pick)
  canvas.addEventListener('pointerleave', e => { if (e.pointerType === 'mouse') { hover = null; update() } })

  new ResizeObserver(() => {
    width = box.clientWidth
    g = width > 480 ? 4 : 2
    c = (width - ML - 7 * g) / 64
    draw()
  }).observe(box)
  onSeek(update)
  document.fonts?.ready.then(draw)
  describe()
  magnify()

  caption(fig, 3, `All 4,096 scores for the same position. Each large block is the square a piece moves from; each cell inside it is the square it moves to, rank 8 at the top. Dot size follows the logarithm of probability, from one in ten million (nothing) to certain (a full dot). Shaded blocks hold a piece of the side to move. Blue marks the network's favourite legal move, red the move actually played; the percentages beside them are shares of the belief it put on legal moves, while hovering reads a cell's raw score.`)
}

function statLine(parent: Element, colour: string | null, label: string, san: string, value: string): void {
  const row = h('div', { class: 'stat' }, parent)
  const sw = h('span', { class: 'swatch' }, row)
  if (colour) sw.style.borderColor = colour
  else sw.classList.add('none')
  h('span', { class: 'stat-label' }, row, label)
  if (san) h('span', { class: 'san' }, row, san)
  h('span', { class: 'num' }, row, value)
}
