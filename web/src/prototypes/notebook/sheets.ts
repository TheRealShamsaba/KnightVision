// Figure 2: the board beside the twelve 0/1 sheets the network is actually given.
import { data } from '../../data'
import { parseFen } from '../../world/board3d'
import { INK, PAPER, GRAPHITE, s, h, piece, sqX, sqY, sqIndex, sqName, uciSquares, glyph, caption, fileRankLabels } from './ink'
import { fenAt, LAST, onSeek, current, scrubber } from './game'

const ORDER = 'PNBRQKpnbrqk' // the trained order
const NAMES: Record<string, string> = { p: 'pawns', n: 'knights', b: 'bishops', r: 'rooks', q: 'queens', k: 'kings' }

/** An SVG board with hatched dark squares; returns the layer to draw pieces into. */
export function plainBoard(parent: Element, S: number, id: string, hatch = true): { svg: SVGSVGElement; layer: SVGGElement } {
  const svg = s('svg', { viewBox: `-18 -4 ${8 * S + 22} ${8 * S + 22}` }, parent)
  const defs = s('defs', {}, svg)
  const pat = s('pattern', { id, width: 5, height: 5, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs)
  s('line', { x1: 0, y1: 0, x2: 0, y2: 5, stroke: INK, 'stroke-width': 1 }, pat)
  for (let sq = 0; sq < 64; sq++) {
    const dark = (sq % 8 + Math.floor(sq / 8)) % 2 === 0
    s('rect', { x: sqX(sq, S), y: sqY(sq, S), width: S, height: S, fill: dark && hatch ? `url(#${id})` : PAPER, stroke: hatch ? 'none' : GRAPHITE, 'stroke-width': 0.5 }, svg)
  }
  s('rect', { x: 0, y: 0, width: 8 * S, height: 8 * S, fill: 'none', stroke: INK, 'stroke-width': 1.25 }, svg)
  fileRankLabels(svg, S, 9)
  return { svg, layer: s('g', {}, svg) }
}

export function sheets(fig: HTMLElement): void {
  const S = 40
  const body = h('div', { class: 'sheets-fig' }, fig)
  const boardBox = h('div', { class: 'sheets-board' }, body)
  const { layer } = plainBoard(boardBox, S, 'hatch-2')

  const grid = h('div', { class: 'sheets-grid' }, body)
  const cells: SVGRectElement[][] = []
  const counts: HTMLElement[] = []
  ORDER.split('').forEach((code, plane) => {
    const white = code === code.toUpperCase()
    const cell = h('div', { class: 'sheet', title: `Plane ${plane}: ${white ? 'white' : 'black'} ${NAMES[code.toLowerCase()]}` }, grid)
    const svg = s('svg', { viewBox: '-1 -1 82 82', 'aria-hidden': 'true' }, cell)
    const rects: SVGRectElement[] = []
    for (let sq = 0; sq < 64; sq++) {
      rects.push(s('rect', { x: sqX(sq, 10) + 0.75, y: sqY(sq, 10) + 0.75, width: 8.5, height: 8.5, fill: PAPER, stroke: GRAPHITE, 'stroke-width': 0.35 }, svg))
    }
    s('rect', { x: 0, y: 0, width: 80, height: 80, fill: 'none', stroke: INK, 'stroke-width': 1 }, svg)
    cells.push(rects)
    const tag = h('div', { class: 'sheet-tag' }, cell)
    h('span', { class: white ? 'g g-w' : 'g' }, tag, glyph(code.toLowerCase()))
    h('span', { class: 'num' }, tag, String(plane))
    counts.push(h('span', { class: 'num dim' }, tag, ''))
  })
  const turn = h('p', { class: 'turn-note' }, fig)

  scrubber(fig, 'scrub-2')

  const draw = (i: number): void => {
    const fen = fenAt(i)
    const board = parseFen(fen)
    layer.innerHTML = ''
    // Outline the move that led here, so a step is easy to follow.
    if (i > 0) for (const sq of uciSquares(data.opera.plies[i - 1].uci)) {
      s('rect', { x: sqX(sq, S) + 1.5, y: sqY(sq, S) + 1.5, width: S - 3, height: S - 3, fill: 'none', stroke: INK, 'stroke-width': 2, 'stroke-dasharray': '3 2' }, layer)
    }
    for (const [name, pc] of board) {
      const sq = sqIndex(name)
      piece(layer, pc.type, pc.color === 'w', sqX(sq, S) + S / 2, sqY(sq, S) + S / 2, S * 0.8)
    }
    ORDER.split('').forEach((code, plane) => {
      let n = 0
      cells[plane].forEach((rect, sq) => {
        const pc = board.get(sqName(sq))
        const on = !!pc && pc.type === code.toLowerCase() && (pc.color === 'w') === (code === code.toUpperCase())
        if (on) n++
        rect.setAttribute('fill', on ? INK : PAPER)
        rect.setAttribute('stroke', on ? INK : GRAPHITE)
      })
      counts[plane].textContent = `×${n}`
    })
    const side = fen.split(' ')[1] === 'w' ? 'White' : 'Black'
    turn.textContent = i === LAST
      ? `${side} to move here, and checkmated. Neither fact appears on any sheet.`
      : `${side} to move here. No sheet says so: there is no thirteenth sheet for whose turn it is.`
  }
  onSeek(draw)
  draw(current())

  caption(fig, 2, `The Opera Game as the network receives it. Left, the board; right, the twelve input sheets in the order it was trained on: white P N B R Q K, then black p n b r q k. A filled cell is a 1. Every other cell is 0. The dashed squares mark the move just played. Scrub, step, or press play to advance one move per second.`)
}
