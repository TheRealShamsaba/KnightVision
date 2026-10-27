// Figure 4: the mirror bug on an all-white board (after Yoko Ono), then a table of six openings.
import { data, numbered, percent, type Candidate } from '../../data'
import { parseFen } from '../../world/board3d'
import { BLUE, RED, h, piece, arrow, sqX, sqY, sqIndex, uciSquares, onTick, caption } from './ink'
import { plainBoard } from './sheets'

// Planes went in as K Q R B N P where P N B R Q K was expected, so each piece read as its mirror.
const SEEN_AS: Record<string, string> = { k: 'p', q: 'n', r: 'b', b: 'r', n: 'q', p: 'k' }

type Mode = 'v1' | 'v2'

export function mirror(fig: HTMLElement, tableBox: HTMLElement): void {
  const row = data.mirror[2] // after 1.e4 e5
  const S = 50
  const toggle = h('div', { class: 'segmented', role: 'group', 'aria-label': 'Which board to show' }, fig)
  const buttons: Record<Mode, HTMLButtonElement> = {
    v1: h('button', { type: 'button', 'aria-pressed': 'false' }, toggle, 'What v1 showed it'),
    v2: h('button', { type: 'button', 'aria-pressed': 'false' }, toggle, 'What it learned'),
  }
  const boardBox = h('div', { class: 'mirror-board' }, fig)
  const { layer } = plainBoard(boardBox, S, 'none-4', false)
  const note = h('p', { class: 'readout' }, fig)

  let untick: (() => void) | null = null
  const show = (mode: Mode | null, colour: boolean): void => {
    layer.innerHTML = ''
    for (const [name, pc] of parseFen(row.fen)) {
      const sq = sqIndex(name)
      const type = mode === 'v1' ? SEEN_AS[pc.type] : pc.type
      piece(layer, type, !colour || pc.color === 'w', sqX(sq, S) + S / 2, sqY(sq, S) + S / 2, S * 0.8)
    }
    note.innerHTML = ''
    if (!mode) {
      note.textContent = 'After 1.e4 e5, every piece painted white. Whose is whose? Choose a board.'
      return
    }
    const move: Candidate = mode === 'v1' ? row.v1[0] : row.v2[0]
    const ink = mode === 'v1' ? RED : BLUE
    arrow(layer, ...uciSquares(move.uci), 2 + move.p * 14, S, ink)
    const tag = h('span', { class: 'move-tag' }, note, mode === 'v1' ? 'v1 played ' : 'Fixed, it wants ')
    tag.style.borderColor = ink
    h('b', {}, tag, `2.${move.san}`)
    tag.append(' ')
    h('span', { class: 'num' }, tag, percent(move.p))
    note.append(colour ? ' Colour restored.' : '')
  }

  const choose = (mode: Mode): void => {
    untick?.()
    for (const m of ['v1', 'v2'] as Mode[]) buttons[m].setAttribute('aria-pressed', String(m === mode))
    show(mode, false)
    // Two ticks of white, then the colour snaps back.
    let ticks = 0
    untick = onTick(() => { if (++ticks === 2) { untick?.(); untick = null; show(mode, true) } })
  }
  buttons.v1.addEventListener('click', () => choose('v1'))
  buttons.v2.addEventListener('click', () => choose('v2'))
  show(null, false)

  caption(fig, 4, `The position after 1.e4 e5 on an all-white board. "What v1 showed it" swaps every piece by the scrambled order: kings appear as pawns, pawns as kings, and so on. The red arrow is what v1 played, the blue arrow what the fixed network wants, each as thick as its probability. After two ticks the colour comes back.`)

  // The table: same story, six openings.
  const table = h('table', {}, tableBox)
  const head = h('tr', {}, h('thead', {}, table))
  h('th', { scope: 'col' }, head, 'After')
  const v1h = h('th', { scope: 'col' }, head); h('span', { class: 'key', style: `background:${RED}` }, v1h); v1h.append('v1, scrambled')
  const v2h = h('th', { scope: 'col' }, head); h('span', { class: 'key', style: `background:${BLUE}` }, v2h); v2h.append('Fixed')
  const body = h('tbody', {}, table)
  data.mirror.forEach((m, i) => {
    const moves = m.line ? m.line.split(' ') : []
    const tr = h('tr', i === 2 ? { class: 'this-one' } : {}, body)
    h('td', {}, tr, moves.length ? numbered(moves) : 'The starting position')
    for (const c of [m.v1[0], m.v2[0]]) {
      const td = h('td', {}, tr, numbered([c.san], moves.length) + ' ')
      h('span', { class: 'num' }, td, percent(c.p))
    }
  })
  const cap = h('p', { class: 'table-caption' }, tableBox)
  h('b', {}, cap, 'Table 1. ')
  cap.append('Top move in six positions, scrambled and fixed. The highlighted row is Figure 4.')
}

