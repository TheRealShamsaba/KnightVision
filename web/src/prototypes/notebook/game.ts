// One cursor into the Opera Game, shared by Figures 2 and 3, so scrubbing either moves both.
import { data, moveLabel } from '../../data'
import { h, onTick } from './ink'

const plies = data.opera.plies
export const LAST = plies.length // index plies.length = the final position, after mate
export const fenAt = (i: number): string => (i < LAST ? plies[i].fen : data.opera.finalFen)
export const labelAt = (i: number): string =>
  i < LAST ? `Before ${moveLabel(i, plies[i].san)}` : `After ${moveLabel(LAST - 1, plies[LAST - 1].san)}, the end`

let index = 0
const listeners = new Set<(i: number) => void>()
export const current = (): number => index
export function seek(i: number): void {
  index = Math.max(0, Math.min(LAST, i))
  listeners.forEach(fn => fn(index))
}
export const onSeek = (fn: (i: number) => void): void => { listeners.add(fn) }

// Playback advances one ply per clock tick; any manual touch stops it.
let playing: (() => void) | null = null
const playButtons = new Set<HTMLButtonElement>()
function setPlaying(on: boolean): void {
  playing?.()
  playing = null
  if (on) {
    if (index >= LAST) seek(0)
    playing = onTick(() => (index >= LAST ? setPlaying(false) : seek(index + 1)))
  }
  playButtons.forEach(b => { b.textContent = on ? 'Pause' : 'Play'; b.setAttribute('aria-pressed', String(on)) })
}

export function scrubber(parent: Element, id: string): void {
  const row = h('div', { class: 'scrubber' }, parent)
  const play = h('button', { type: 'button', 'aria-pressed': 'false', class: 'play' }, row, 'Play')
  const prev = h('button', { type: 'button', 'aria-label': 'Previous move' }, row, '‹')
  const range = h('input', { type: 'range', min: 0, max: LAST, step: 1, value: index, id, 'aria-label': 'Position in the game' }, row)
  const next = h('button', { type: 'button', 'aria-label': 'Next move' }, row, '›')
  const label = h('output', { for: id, class: 'position' }, parent)
  playButtons.add(play)

  play.addEventListener('click', () => setPlaying(!playing))
  prev.addEventListener('click', () => { setPlaying(false); seek(index - 1) })
  next.addEventListener('click', () => { setPlaying(false); seek(index + 1) })
  range.addEventListener('input', () => { setPlaying(false); seek(Number(range.value)) })

  const sync = (i: number): void => {
    range.value = String(i)
    label.innerHTML = ''
    label.append(labelAt(i), ' ')
    h('span', { class: 'num dim' }, label, `${i}/${LAST}`)
  }
  onSeek(sync)
  sync(index)
}
