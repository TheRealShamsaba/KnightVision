// The figures that explain the machinery: the training loss, one real convolution, how far a square
// can see, and the Elo estimate. Each one is driven by the reader's hand, and each uses real numbers.
import { data, percent } from '../../data'
import { parseFen, FILES } from '../../world/board3d'
import { INK, PAPER, GRAPHITE, BLUE, RED, s, h, piece, sqX, sqY, sqIndex, sqName, caption, fileRankLabels } from './ink'
import { current, fenAt, onSeek } from './game'

/** Pointer position in an SVG's own coordinates. */
function svgPoint(svg: SVGSVGElement, e: PointerEvent): [number, number] {
  const m = svg.getScreenCTM()!.inverse()
  const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m)
  return [p.x, p.y]
}

/** Lets a pointer drag set a value: pointerdown + pointermove while pressed (touch works the same way). */
function draggable(el: Element, fn: (e: PointerEvent) => void): void {
  let down = false
  el.addEventListener('pointerdown', (e) => { down = true; (el as HTMLElement).setPointerCapture?.((e as PointerEvent).pointerId); fn(e as PointerEvent) })
  el.addEventListener('pointermove', (e) => { if (down || (e as PointerEvent).pointerType === 'mouse') fn(e as PointerEvent) })
  el.addEventListener('pointerup', () => { down = false })
  el.addEventListener('pointercancel', () => { down = false })
}

// ---------- The loss ----------

export function lossFig(fig: HTMLElement): void {
  const l = data.learning
  const W = 640, H = 300, L = 46, B = 36, T = 14, R = 14
  const xMin = -5, xMax = 0, yMax = 12
  const x = (log10p: number) => L + ((log10p - xMin) / (xMax - xMin)) * (W - L - R)
  const y = (loss: number) => T + (1 - Math.min(loss, yMax) / yMax) * (H - T - B)
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'The loss, minus the log of the probability given to the right move' }, h('div', { class: 'plot' }, fig))
  // Axes and grid.
  for (let k = xMin; k <= xMax; k++) {
    s('line', { x1: x(k), x2: x(k), y1: T, y2: H - B, stroke: '#E4E1D8' }, svg)
    s('text', { x: x(k), y: H - B + 16, 'text-anchor': 'middle', class: 'axis' }, svg).textContent = k === 0 ? '100%' : k === -1 ? '10%' : k === -2 ? '1%' : `10${'⁻'}${'⁰¹²³⁴⁵⁶⁷⁸⁹'[-k]}`
  }
  for (let v = 0; v <= yMax; v += 2) {
    s('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: v ? '#E4E1D8' : INK }, svg)
    s('text', { x: L - 8, y: y(v) + 4, 'text-anchor': 'end', class: 'axis' }, svg).textContent = String(v)
  }
  s('text', { x: (L + W - R) / 2, y: H - 4, 'text-anchor': 'middle', class: 'axis' }, svg).textContent = 'probability it gave the move the person played (log scale)'
  s('text', { x: 12, y: (T + H - B) / 2, 'text-anchor': 'middle', class: 'axis', transform: `rotate(-90 12 ${(T + H - B) / 2})` }, svg).textContent = 'loss'
  // The curve: loss = −ln p.
  let d = ''
  for (let k = 0; k <= 100; k++) {
    const lg = xMin + (k / 100) * (xMax - xMin)
    d += `${k ? 'L' : 'M'}${x(lg).toFixed(1)} ${y(-Math.log(10 ** lg)).toFixed(1)}`
  }
  s('path', { d, fill: 'none', stroke: INK, 'stroke-width': 2 }, svg)
  // Fixed markers: blind guessing, the untrained network, the trained network.
  const mark = (loss: number, label: string, colour: string, dy: number, left = false) => {
    const lg = Math.log10(Math.exp(-loss))
    s('circle', { cx: x(lg), cy: y(loss), r: 5, fill: PAPER, stroke: colour, 'stroke-width': 2 }, svg)
    s('text', { x: x(lg) + (left ? -9 : 9), y: y(loss) + dy, class: 'plot-label', fill: colour, 'text-anchor': left ? 'end' : 'start' }, svg).textContent = label
  }
  mark(Math.log(4096), 'blind guess, 1 in 4,096', GRAPHITE, 18)
  mark(l.untrained.loss, `untrained: ${l.untrained.loss.toFixed(2)}`, RED, -10)
  mark(l.trained.loss, `trained: ${l.trained.loss.toFixed(2)}`, BLUE, -12, true)
  // The reader's handle.
  const handle = s('g', {}, svg)
  const guide = s('line', { stroke: INK, 'stroke-dasharray': '3 3' }, handle)
  const dot = s('circle', { r: 7, fill: INK }, handle)
  const readout = h('p', { class: 'readout' }, fig)
  const set = (lg: number) => {
    lg = Math.max(xMin, Math.min(xMax, lg))
    const p = 10 ** lg, loss = -Math.log(p)
    guide.setAttribute('x1', String(x(lg))); guide.setAttribute('x2', String(x(lg)))
    guide.setAttribute('y1', String(H - B)); guide.setAttribute('y2', String(y(loss)))
    dot.setAttribute('cx', String(x(lg))); dot.setAttribute('cy', String(y(loss)))
    readout.innerHTML = ''
    readout.append('If it gives the right move ')
    h('span', { class: 'num' }, readout, p >= 0.001 ? percent(p) : `1 in ${Math.round(1 / p).toLocaleString('en-GB')}`)
    readout.append(', the loss is ')
    h('span', { class: 'num' }, readout, loss.toFixed(2))
    readout.append('. Drag along the curve.')
  }
  draggable(svg, (e) => { const [px] = svgPoint(svg, e); set(xMin + ((px - L) / (W - L - R)) * (xMax - xMin)) })
  set(Math.log10(0.1))
  caption(fig, 2, `The loss training minimises: minus the natural log of the probability the network gave the move the person actually played. Certain and right costs 0; a blind guess among all 4,096 moves costs ${Math.log(4096).toFixed(2)}. For the position after 1.e4 e5, where the person played ${l.answer}, the untrained network scores ${l.untrained.loss.toFixed(2)}, barely better than blind; the trained one ${l.trained.loss.toFixed(2)}. Training is five million small nudges down this curve.`)
}

// ---------- One real convolution ----------

const ORDER = 'PNBRQKpnbrqk'

/** The twelve sheets as numbers: planes[k][row][col], row 0 = rank 8, the trained layout. */
function planes(fen: string): number[][][] {
  const out = ORDER.split('').map(() => Array.from({ length: 8 }, () => new Array(8).fill(0)))
  for (const [name, pc] of parseFen(fen)) {
    const k = ORDER.indexOf(pc.color === 'w' ? pc.type.toUpperCase() : pc.type)
    out[k][8 - Number(name[1])][FILES.indexOf(name[0])] = 1
  }
  return out
}

export function convFig(fig: HTMLElement): void {
  const filters = data.filters
  let which = 0
  let centre = sqIndex('e2')
  const picker = h('div', { class: 'segmented' }, fig)
  const buttons = filters.map((f, i) => {
    const b = h('button', { type: 'button', 'aria-pressed': String(i === 0) }, picker, `Channel ${f.channel}`)
    b.addEventListener('click', () => { which = i; buttons.forEach((x, k) => x.setAttribute('aria-pressed', String(k === i))); draw() })
    return b
  })
  const row = h('div', { class: 'conv' }, fig)
  const S = 34
  const boardSvg = s('svg', { viewBox: `-16 -4 ${8 * S + 20} ${8 * S + 22}`, role: 'img', 'aria-label': 'The board with a 3 by 3 window you can drag' }, h('div', { class: 'conv-board' }, row))
  const boardLayer = s('g', {}, boardSvg)
  const filterBox = h('div', { class: 'conv-filter' }, row)
  const outSvg = s('svg', { viewBox: `-16 -4 ${8 * S + 20} ${8 * S + 22}`, role: 'img', 'aria-label': "That channel's output for every square" }, h('div', { class: 'conv-out' }, row))
  const outLayer = s('g', {}, outSvg)
  fileRankLabels(boardSvg, S, 9)
  fileRankLabels(outSvg, S, 9)
  const readout = h('p', { class: 'readout' }, fig)

  const draw = () => {
    const fen = fenAt(current())
    const x = planes(fen)
    const f = filters[which]
    const w = (k: number, dr: number, dc: number) => f.weights[k * 9 + (dr + 1) * 3 + (dc + 1)]
    // Output for every square: the same 3×3 × 12 dot product, slid across the board, then ReLU.
    const out: number[][] = []
    for (let r = 0; r < 8; r++) {
      out.push([])
      for (let c = 0; c < 8; c++) {
        let acc = f.bias
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
          const rr = r + dr, cc = c + dc
          if (rr < 0 || rr > 7 || cc < 0 || cc > 7) continue
          for (let k = 0; k < 12; k++) acc += w(k, dr, dc) * x[k][rr][cc]
        }
        out[r].push(acc)
      }
    }
    const peak = Math.max(1e-6, ...out.flat().map((v) => Math.max(0, v)))
    // Board with the window.
    boardLayer.innerHTML = ''
    for (let sq = 0; sq < 64; sq++) {
      const dark = (sq % 8 + Math.floor(sq / 8)) % 2 === 0
      s('rect', { x: sqX(sq, S), y: sqY(sq, S), width: S, height: S, fill: dark ? '#E4E1D8' : PAPER }, boardLayer)
    }
    for (const [name, pc] of parseFen(fen)) {
      const sq = sqIndex(name)
      piece(boardLayer, pc.type, pc.color === 'w', sqX(sq, S) + S / 2, sqY(sq, S) + S / 2, S * 0.78)
    }
    const cr = 7 - Math.floor(centre / 8), cc = centre % 8
    s('rect', { x: (cc - 1) * S, y: (cr - 1) * S, width: 3 * S, height: 3 * S, fill: 'none', stroke: INK, 'stroke-width': 3 }, boardLayer)
    s('rect', { x: 0, y: 0, width: 8 * S, height: 8 * S, fill: 'none', stroke: INK, 'stroke-width': 1.25 }, boardLayer)
    // The filter: twelve 3×3 grids of weights, blue positive, red negative, square size = strength.
    const maxW = Math.max(...f.weights.map(Math.abs))
    filterBox.innerHTML = ''
    h('p', { class: 'conv-label' }, filterBox, '108 weights, one 3×3 per sheet')
    const grid = h('div', { class: 'filter-grid' }, filterBox)
    ORDER.split('').forEach((code, k) => {
      const cell = h('div', {}, grid)
      const g = s('svg', { viewBox: '0 0 30 30', 'aria-hidden': 'true' }, cell)
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        const v = w(k, dr, dc), size = 9 * Math.sqrt(Math.abs(v) / maxW)
        const lit = cr + dr >= 0 && cr + dr < 8 && cc + dc >= 0 && cc + dc < 8 && x[k][cr + dr][cc + dc] === 1
        s('rect', { x: (dc + 1) * 10, y: (dr + 1) * 10, width: 10, height: 10, fill: lit ? '#D6D3CB' : 'none', stroke: '#D6D3CB', 'stroke-width': 0.5 }, g)
        s('rect', { x: (dc + 1) * 10 + 5 - size / 2, y: (dr + 1) * 10 + 5 - size / 2, width: size, height: size, fill: v >= 0 ? BLUE : RED }, g)
      }
      h('span', { class: 'num' }, cell, code)
    })
    // Output map.
    outLayer.innerHTML = ''
    for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) {
      const v = Math.max(0, out[r][c]) / peak
      s('rect', { x: c * S, y: r * S, width: S, height: S, fill: PAPER, stroke: '#E4E1D8', 'stroke-width': 0.5 }, outLayer)
      const size = S * 0.9 * Math.sqrt(v)
      if (size > 0.5) s('rect', { x: c * S + (S - size) / 2, y: r * S + (S - size) / 2, width: size, height: size, fill: INK }, outLayer)
    }
    s('rect', { x: cc * S, y: cr * S, width: S, height: S, fill: 'none', stroke: RED, 'stroke-width': 2.5 }, outLayer)
    s('rect', { x: 0, y: 0, width: 8 * S, height: 8 * S, fill: 'none', stroke: INK, 'stroke-width': 1.25 }, outLayer)
    // What went into this one square.
    const parts: { text: string; v: number }[] = []
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      const rr = cr + dr, c2 = cc + dc
      if (rr < 0 || rr > 7 || c2 < 0 || c2 > 7) continue
      for (let k = 0; k < 12; k++) if (x[k][rr][c2]) parts.push({ text: `${ORDER[k]} on ${FILES[c2]}${8 - rr}`, v: w(k, dr, dc) })
    }
    const sum = f.bias + parts.reduce((a, p) => a + p.v, 0)
    readout.innerHTML = ''
    readout.append(`Window on ${sqName(centre)}: `)
    if (parts.length) parts.sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 4).forEach((p, i) => {
      if (i) readout.append(', ')
      readout.append(`${p.text} `)
      h('span', { class: 'num', style: `color:${p.v >= 0 ? BLUE : RED}` }, readout, `${p.v >= 0 ? '+' : ''}${p.v.toFixed(2)}`)
    })
    else readout.append('no pieces under it')
    readout.append(parts.length > 4 ? ', …' : '')
    readout.append('. Plus bias ')
    h('span', { class: 'num' }, readout, f.bias.toFixed(2))
    readout.append(' = ')
    h('span', { class: 'num' }, readout, sum.toFixed(2))
    readout.append(sum > 0 ? ', which passes through.' : ', which ReLU turns into 0.')
  }
  const pick = (svg: SVGSVGElement) => (e: PointerEvent) => {
    const [px, py] = svgPoint(svg, e)
    const c = Math.floor(px / S), r = Math.floor(py / S)
    if (c < 0 || c > 7 || r < 0 || r > 7) return
    const next = (7 - r) * 8 + c
    if (next !== centre) { centre = next; draw() }
  }
  draggable(boardSvg, pick(boardSvg))
  draggable(outSvg, pick(outSvg))
  onSeek(draw)
  draw()
  caption(fig, 4, `One real filter from the network's first layer, computed in your browser from its actual weights for the same position as Figure 3 (scrub there to change it). Drag the window over the board, left, or point at a square of the output, right. Middle: the filter's 108 weights, a 3×3 patch for each of the twelve sheets; blue adds, red subtracts, grey cells are lit by a piece under the window. The output square is the sum of every weight times the 1s beneath it, plus a bias, with negatives cut to zero (the ReLU). The network has 256 channels like this in its first layer alone.`)
}

// ---------- How far a square can see ----------

const STAGES = ['Opening convolution', 'Widening convolution', 'Residual block 1', 'Residual block 2', 'Residual block 3', 'Residual block 4', 'Residual block 5', 'Policy head']
// Each 3×3 convolution lets a square see one square further; a residual block has two of them.
const REACH = [1, 2, 4, 6, 8, 10, 12, 12]

export function reachFig(fig: HTMLElement): void {
  let sq = sqIndex('a1')
  const row = h('div', { class: 'reach-row' }, fig)
  const S = 10
  const minis = STAGES.map((name, i) => {
    const cell = h('div', { class: 'reach-mini' }, row)
    const svg = s('svg', { viewBox: `-1 -1 ${8 * S + 2} ${8 * S + 2}`, role: 'img', 'aria-label': `After ${name.toLowerCase()}` }, cell)
    const rects: SVGRectElement[] = []
    for (let k = 0; k < 64; k++) rects.push(s('rect', { x: sqX(k, S) + 0.5, y: sqY(k, S) + 0.5, width: S - 1, height: S - 1 }, svg))
    s('rect', { x: 0, y: 0, width: 8 * S, height: 8 * S, fill: 'none', stroke: INK, 'stroke-width': 0.75 }, svg)
    h('p', { class: 'num reach-stage' }, cell, `${i + 1}`)
    svg.addEventListener('pointerdown', (e) => {
      const [px, py] = svgPoint(svg, e)
      const c = Math.floor(px / S), r = Math.floor(py / S)
      if (c >= 0 && c < 8 && r >= 0 && r < 8) { sq = (7 - r) * 8 + c; draw() }
    })
    svg.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse') return
      const [px, py] = svgPoint(svg, e)
      const c = Math.floor(px / S), r = Math.floor(py / S)
      if (c >= 0 && c < 8 && r >= 0 && r < 8 && (7 - r) * 8 + c !== sq) { sq = (7 - r) * 8 + c; draw() }
    })
    return rects
  })
  const readout = h('p', { class: 'readout' }, fig)
  const draw = () => {
    const f = sq % 8, r = Math.floor(sq / 8)
    let whole = -1
    minis.forEach((rects, i) => {
      let all = true
      rects.forEach((rect, k) => {
        const d = Math.max(Math.abs((k % 8) - f), Math.abs(Math.floor(k / 8) - r))
        const seen = d <= REACH[i]
        if (!seen) all = false
        rect.setAttribute('fill', k === sq ? RED : seen ? INK : '#ECE9E1')
      })
      if (all && whole < 0) whole = i + 1
    })
    readout.innerHTML = ''
    readout.append(`From ${sqName(sq)}, the whole board is in view by stage `)
    h('span', { class: 'num' }, readout, String(whole))
    readout.append(whole <= 3 ? '. Central squares get there fast.' : '. Corners take longest.')
  }
  draw()
  caption(fig, 5, `How far one square can "see" after each of the eight stages. Every 3×3 convolution widens the view by one square in each direction; the five residual blocks hold two convolutions each. Point at or tap any square in any of the small boards to move the red square. A centre square sees everything by stage 3; the corners have to wait until stage 5.`)
}

// ---------- The Elo estimate ----------

export function eloFig(fig: HTMLElement): void {
  const elo = data.elo
  if (!elo) { fig.remove(); return }
  const games = elo.games
  const levels = [...new Set(games.map((g) => g.elo))].sort()
  const actual = levels.map((l) => { const gs = games.filter((g) => g.elo === l); return gs.reduce((a, g) => a + g.score, 0) / gs.length })
  const expected = (rating: number, opp: number) => 1 / (1 + 10 ** ((opp - rating) / 400))
  const logLik = (rating: number) => games.reduce((a, g) => {
    const e = expected(rating, g.elo)
    return a + g.score * Math.log(e) + (1 - g.score) * Math.log(1 - e)
  }, 0)
  const W = 640, H = 260, L = 20, B = 36, T = 16, R = 20, lo = 500, hi = 1400
  const x = (rt: number) => L + ((rt - lo) / (hi - lo)) * (W - L - R)
  const best = logLik(elo.rating)
  const rel = (rt: number) => Math.exp(logLik(rt) - best)
  const y = (v: number) => T + (1 - v) * (H - T - B)
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'How well each rating explains the 160 results' }, h('div', { class: 'plot' }, fig))
  s('rect', { x: x(elo.rating - elo.margin), y: T, width: x(elo.rating + elo.margin) - x(elo.rating - elo.margin), height: H - T - B, fill: '#ECE9E1' }, svg)
  for (let rt = lo; rt <= hi; rt += 100) {
    s('line', { x1: x(rt), x2: x(rt), y1: H - B, y2: H - B + 5, stroke: INK }, svg)
    s('text', { x: x(rt), y: H - B + 18, 'text-anchor': 'middle', class: 'axis' }, svg).textContent = String(rt)
  }
  s('line', { x1: L, x2: W - R, y1: H - B, y2: H - B, stroke: INK }, svg)
  let d = ''
  for (let rt = lo; rt <= hi; rt += 5) d += `${rt === lo ? 'M' : 'L'}${x(rt).toFixed(1)} ${y(rel(rt)).toFixed(1)}`
  s('path', { d, fill: 'none', stroke: INK, 'stroke-width': 2 }, svg)
  s('text', { x: x(elo.rating), y: T - 3 + 12, 'text-anchor': 'middle', class: 'plot-label' }, svg).textContent = `${elo.rating}`
  // Labels on alternate sides of their lines, since the levels sit close together.
  levels.forEach((l, i) => {
    s('line', { x1: x(l), x2: x(l), y1: T, y2: H - B, stroke: RED, 'stroke-dasharray': '4 3' }, svg)
    s('text', { x: x(l) + (i % 2 ? 4 : -4), y: T + 12, 'text-anchor': i % 2 ? 'start' : 'end', class: 'plot-label', fill: RED }, svg).textContent = `Stockfish ${l}`
  })
  const handle = s('g', {}, svg)
  const guide = s('line', { stroke: BLUE, 'stroke-width': 1.5, y1: T, y2: H - B }, handle)
  const dot = s('circle', { r: 6, fill: BLUE }, handle)
  const readout = h('p', { class: 'readout' }, fig)
  const set = (rt: number) => {
    rt = Math.round(Math.max(lo, Math.min(hi, rt)) / 5) * 5
    guide.setAttribute('x1', String(x(rt))); guide.setAttribute('x2', String(x(rt)))
    dot.setAttribute('cx', String(x(rt))); dot.setAttribute('cy', String(y(rel(rt))))
    readout.innerHTML = ''
    readout.append('A ')
    h('span', { class: 'num', style: `color:${BLUE}` }, readout, String(rt))
    readout.append(' player would expect to score ')
    levels.forEach((l, i) => {
      if (i) readout.append(' and ')
      h('span', { class: 'num' }, readout, percent(expected(rt, l)))
      readout.append(` against ${l}`)
    })
    readout.append('. KnightVision actually scored ')
    levels.forEach((l, i) => {
      if (i) readout.append(' and ')
      h('span', { class: 'num' }, readout, percent(actual[i]))
    })
    readout.append('.')
  }
  draggable(svg, (e) => { const [px] = svgPoint(svg, e); set(lo + ((px - L) / (W - L - R)) * (hi - lo)) })
  set(1200)
  caption(fig, 7, `How well each possible rating explains the ${games.length} results against Stockfish held to ${levels.join(' and ')} Elo. The curve is the likelihood, scaled so its peak is 1: the rating that makes the actual results most probable is ${elo.rating}. The shaded band, ±${elo.margin}, is the 95% interval. Drag to try a rating: a ${levels[0]} player would expect 50% against Stockfish ${levels[0]}, and the further the rating falls, the lower that expected score.`)
}
