// One position for the whole stack of mail: every scrubber moves it, every listener redraws.
import { data, moveLabel, type Ply } from '../data'

const plies = data.opera.plies
type Listener = (ply: Ply, index: number) => void

let index = 10 // before 6.Bc4, the position the opening card shows
const listeners: Listener[] = []
const controls: { range: HTMLInputElement; label: HTMLElement; play: HTMLButtonElement }[] = []
let timer = 0

export const onPosition = (fn: Listener) => { listeners.push(fn) }
export const current = (): Ply => plies[index]

export function setPosition(i: number) {
  index = Math.max(0, Math.min(plies.length - 1, i))
  for (const c of controls) {
    c.range.value = String(index)
    c.label.textContent = `before ${moveLabel(index, plies[index].san)}`
  }
  listeners.forEach((fn) => fn(plies[index], index))
}

function toggle() {
  if (timer) { clearInterval(timer); timer = 0 } else {
    if (index >= plies.length - 1) setPosition(0)
    // One move a second, like a clock.
    timer = window.setInterval(() => {
      if (index >= plies.length - 1) { clearInterval(timer); timer = 0; sync(); return }
      setPosition(index + 1)
    }, 1000)
  }
  sync()
}
const sync = () => controls.forEach((c) => { c.play.textContent = timer ? 'pause' : 'play the game' })

export function mountScrubs() {
  document.querySelectorAll<HTMLElement>('[data-scrub]').forEach((host) => {
    host.innerHTML = `
      <button type="button" class="step" data-d="-1" aria-label="Previous move">‹</button>
      <input type="range" min="0" max="${plies.length - 1}" step="1" aria-label="Position in the Opera Game">
      <button type="button" class="step" data-d="1" aria-label="Next move">›</button>
      <span class="scrub-label" aria-live="polite"></span>
      <button type="button" class="scrub-play">play the game</button>`
    const range = host.querySelector('input')!
    range.addEventListener('input', () => setPosition(Number(range.value)))
    host.querySelectorAll<HTMLButtonElement>('.step').forEach((b) => b.addEventListener('click', () => setPosition(index + Number(b.dataset.d))))
    const play = host.querySelector<HTMLButtonElement>('.scrub-play')!
    play.addEventListener('click', toggle)
    controls.push({ range, label: host.querySelector('.scrub-label')!, play })
  })
}
