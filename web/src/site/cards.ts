// Drawings for the cards the prototype didn't have: what flows through the layers, and the ledger
// of games against Stockfish. Same inks, same misregistered overprint.
import { data } from '../data'
import { inkLayer, overprint, sizeCanvas } from './ink'

const decode = (b64: string): Uint8Array => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))

/** One 8×8 activation map as a halftone: every square a dot, its area the signal's strength. */
export function activationTile(c: HTMLCanvasElement, b64: string, size: number, seed: number): void {
  const g = sizeCanvas(c, size, size)
  const bytes = decode(b64)
  const cell = size / 8
  const frame = inkLayer(size, size, 'orange', seed, (t) => {
    t.lineWidth = 1; t.strokeRect(0.5, 0.5, size - 1, size - 1)
  })
  const dots = inkLayer(size, size, 'blue', seed + 1, (t) => {
    t.beginPath()
    bytes.forEach((v, i) => {
      const r = cell * 0.5 * Math.sqrt(v / 255)
      if (r < 0.4) return
      const x = (i % 8 + 0.5) * cell, y = (Math.floor(i / 8) + 0.5) * cell
      t.moveTo(x + r, y); t.arc(x, y, r, 0, 7)
    })
    t.fill()
  })
  overprint(g, [[frame, 'orange'], [dots, 'blue']])
}

/**
 * The ledger: every game against Stockfish as a small stamped mark, longest games first.
 * Won = a solid orange square; drawn = a purple half; lost = a blue outline, cancelled.
 */
export function ledger(c: HTMLCanvasElement, width: number): void {
  const games = data.elo?.games ?? []
  const levels = [...new Set(games.map((g) => g.elo))].sort()
  const cols = width < 420 ? 16 : 20
  const cell = width / cols
  const rowsPer = Math.ceil(games.length / levels.length / cols)
  const labelH = 26
  const height = levels.length * (rowsPer * cell + labelH + 14)
  const g = sizeCanvas(c, width, height)
  const at = (level: number, i: number): [number, number] => {
    const top = level * (rowsPer * cell + labelH + 14) + labelH
    return [(i % cols) * cell, top + Math.floor(i / cols) * cell]
  }
  const sorted = levels.map((l) => games.filter((x) => x.elo === l).sort((a, b) => b.score - a.score || b.plies - a.plies))
  const pad = cell * 0.16, s = cell - 2 * pad
  const orange = inkLayer(width, height, 'orange', 71, (t) => {
    sorted.forEach((list, l) => list.forEach((x, i) => {
      if (x.score !== 1) return
      const [px, py] = at(l, i); t.fillRect(px + pad, py + pad, s, s)
    }))
  })
  const purple = inkLayer(width, height, 'purple', 72, (t) => {
    sorted.forEach((list, l) => list.forEach((x, i) => {
      if (x.score !== 0.5) return
      const [px, py] = at(l, i)
      t.beginPath(); t.moveTo(px + pad, py + pad + s); t.lineTo(px + pad + s, py + pad); t.lineTo(px + pad + s, py + pad + s); t.fill()
      t.lineWidth = 1; t.strokeRect(px + pad, py + pad, s, s)
    }))
  })
  const blue = inkLayer(width, height, 'blue', 73, (t) => {
    t.lineWidth = Math.max(1, cell * 0.06)
    sorted.forEach((list, l) => list.forEach((x, i) => {
      if (x.score !== 0) return
      const [px, py] = at(l, i)
      t.strokeRect(px + pad, py + pad, s, s)
      t.beginPath(); t.moveTo(px + pad, py + pad); t.lineTo(px + pad + s, py + pad + s); t.stroke()
    }))
  })
  const black = inkLayer(width, height, 'black', 74, (t) => {
    t.font = '13px "Courier Prime"'
    sorted.forEach((list, l) => {
      const [, y] = at(l, 0)
      const won = list.filter((x) => x.score === 1).length, drawn = list.filter((x) => x.score === 0.5).length
      t.fillText(`vs stockfish ${levels[l]}: ${won} won, ${drawn} drawn, ${list.length - won - drawn} lost`, 0, y - 9)
    })
  })
  overprint(g, [[orange, 'orange'], [purple, 'purple'], [blue, 'blue'], [black, 'black']])
}
