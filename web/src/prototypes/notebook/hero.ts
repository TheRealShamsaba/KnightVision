// Figure 1: the op-art board. Each dark square is a nest of square rings that steps inward once per
// clock tick; the step is delayed by distance from the centre, so a slow wave walks out across the board.
// It runs for a few ticks as an overture, then settles into grey so the pieces and arrows can be read.
import { data, percent } from '../../data'
import { parseFen } from '../../world/board3d'
import { INK, PAPER, s, h, piece, arrow, sqX, sqY, sqIndex, uciSquares, onTick, setSound, reducedMotion, caption, fileRankLabels } from './ink'

const S = 60
const RINGS = 4
const GAP = S / 2 / RINGS
const STROKE = GAP * 0.5

export function hero(fig: HTMLElement): void {
  const ply = data.opera.plies[10]
  const top = ply.top.slice(0, 6)
  const width = (p: number): number => 1.5 + p * 20

  const wrap = h('div', { class: 'hero' }, fig)
  const svg = s('svg', { viewBox: `-34 -34 ${8 * S + 68} ${8 * S + 68}`, role: 'img', 'aria-label': `The Opera Game before 6.Bc4, with the network's six favourite moves drawn as arrows` }, wrap)
  // Three ruled lines frame the board, like the edge of a printed diagram.
  for (const d of [6, 12, 18]) s('rect', { x: -d, y: -d, width: 8 * S + 2 * d, height: 8 * S + 2 * d, fill: 'none', stroke: INK, 'stroke-width': d === 6 ? 1.5 : 0.75 }, svg)
  s('rect', { x: 0, y: 0, width: 8 * S, height: 8 * S, fill: PAPER }, svg)

  const rings: { rect: SVGRectElement; sq: number; k: number; delay: number }[] = []
  for (let sq = 0; sq < 64; sq++) {
    const f = sq % 8, r = Math.floor(sq / 8)
    if ((f + r) % 2 !== 0) continue // light squares stay paper
    const delay = Math.round(Math.max(Math.abs(f - 3.5), Math.abs(r - 3.5)) - 0.5)
    for (let k = 0; k < RINGS; k++) {
      rings.push({ rect: s('rect', { fill: 'none', stroke: INK, 'stroke-width': STROKE, class: 'ring' }, svg), sq, k, delay })
    }
  }
  const step = (tick: number): void => {
    for (const ring of rings) {
      const phase = (((tick - ring.delay) % 4) + 4) % 4
      const inset = STROKE / 2 + (ring.k + phase / 4) * GAP
      const size = S - 2 * inset
      if (size < 2) { ring.rect.setAttribute('display', 'none'); continue }
      ring.rect.removeAttribute('display')
      ring.rect.setAttribute('x', String(sqX(ring.sq, S) + inset))
      ring.rect.setAttribute('y', String(sqY(ring.sq, S) + inset))
      ring.rect.setAttribute('width', String(size))
      ring.rect.setAttribute('height', String(size))
    }
  }
  step(0)
  s('rect', { x: 0, y: 0, width: 8 * S, height: 8 * S, fill: 'none', stroke: INK, 'stroke-width': 1.5 }, svg)

  for (const [name, pc] of parseFen(ply.fen)) {
    const sq = sqIndex(name)
    piece(svg, pc.type, pc.color === 'w', sqX(sq, S) + S / 2, sqY(sq, S) + S / 2, S * 0.78)
  }
  // Thinnest first, so the favourite is drawn on top.
  for (const c of [...top].reverse()) {
    const [from, to] = uciSquares(c.uci)
    arrow(svg, from, to, width(c.p), S)
  }
  fileRankLabels(svg, S, 26)

  const legend = h('div', { class: 'hero-legend' }, wrap)
  h('p', { class: 'legend-title' }, legend, 'Its six favourite moves')
  const list = h('ol', {}, legend)
  for (const c of top) {
    const li = h('li', {}, list)
    const bar = s('svg', { viewBox: '0 0 60 24', class: 'bar', 'aria-hidden': 'true' }, li)
    s('rect', { x: 0, y: 12 - width(c.p) / 2, width: 60, height: width(c.p), fill: INK }, bar)
    h('span', { class: 'san' }, li, c.san)
    h('span', { class: 'num' }, li, percent(c.p))
  }
  const controls = h('div', { class: 'controls quiet' }, legend)
  const clockBtn = h('button', { type: 'button' }, controls, 'Stop the clock')
  const soundBtn = h('button', { type: 'button', 'aria-pressed': 'false' }, controls, 'Tick sound: off')

  const OVERTURE = 8
  let stop: (() => void) | null = null
  const settle = (): void => {
    stop?.(); stop = null
    svg.classList.add('settled')
    clockBtn.textContent = 'Run the op-art again'
  }
  const start = (): void => {
    svg.classList.remove('settled')
    let ticks = 0
    stop = onTick(t => { step(t); if (++ticks >= OVERTURE) settle() })
    clockBtn.textContent = 'Stop'
  }
  clockBtn.addEventListener('click', () => (stop ? settle() : start()))
  if (reducedMotion()) settle()
  else start()
  soundBtn.addEventListener('click', () => {
    const on = soundBtn.getAttribute('aria-pressed') !== 'true'
    setSound(on)
    soundBtn.setAttribute('aria-pressed', String(on))
    soundBtn.textContent = `Tick sound: ${on ? 'on' : 'off'}`
  })

  caption(fig, 1, `The Opera Game, just before 6.Bc4. The arrows are the network's six favourite moves, each as thick as its probability. Bc4, the move played, gets ${percent(ply.share)}. The dark squares step inward once a second for a few seconds, like a clock, then settle.`)
}
