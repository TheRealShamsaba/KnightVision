import './style.css'
import { data, numbered, percent } from '../../data'
import { dotBar, pieceSheet, postage, rng, stampBoard, SWAP } from './ink'
import { mountQuilt } from './quilt'
import { setSound, thunk } from './sound'

const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
const opera = data.opera.plies[10]
const mirror = data.mirror[2]
const $ = <T extends HTMLElement = HTMLElement>(sel: string) => document.querySelector<T>(sel)!

// Newsprint grain, generated once and laid over every card (and the desk) with multiply.
function grain(): string {
  const c = document.createElement('canvas'); c.width = c.height = 220
  const g = c.getContext('2d')!, img = g.createImageData(220, 220), r = rng(9)
  for (let i = 0; i < img.data.length; i += 4) {
    const v = 255 - Math.floor(r() * r() * 40)
    img.data[i] = v; img.data[i + 1] = v - 1; img.data[i + 2] = v - 4; img.data[i + 3] = 255
  }
  g.putImageData(img, 0, 0)
  g.strokeStyle = 'rgba(110,90,70,.12)'; g.lineWidth = 0.6
  for (let i = 0; i < 46; i++) { // stray fibres
    const x = r() * 220, y = r() * 220, a = r() * 6.3, l = 4 + r() * 10
    g.beginPath(); g.moveTo(x, y); g.quadraticCurveTo(x + Math.cos(a) * l, y + Math.sin(a + 1) * l, x + Math.cos(a) * l * 2, y + Math.sin(a) * l * 2); g.stroke()
  }
  return c.toDataURL()
}
document.documentElement.style.setProperty('--grain', `url(${grain()})`)

// Postmarks: double ring, curved town lines, a date in the middle, wavy cancellation bars.
document.querySelectorAll<HTMLElement>('.postmark').forEach((el, i) => {
  const { top = '', bottom = '', mid = '' } = el.dataset
  const waves = [0, 1, 2, 3].map((k) => `<path d="M106 ${30 + k * 15} q6 -6 12 0 t12 0 t12 0 t12 0 t12 0 t12 0 t12 0"/>`).join('')
  el.innerHTML = `<svg viewBox="0 0 200 110" aria-hidden="true"><g filter="url(#worn)">
    <g fill="none" stroke="#111" stroke-width="2.2"><circle cx="55" cy="55" r="46"/><circle cx="55" cy="55" r="30"/>${waves}</g>
    <path id="pt${i}" d="M20 55 A35 35 0 0 1 90 55" fill="none"/><path id="pb${i}" d="M13 55 A42 42 0 0 0 97 55" fill="none"/>
    <g fill="#111" font-family="Courier Prime, monospace" font-size="11.5" text-anchor="middle">
      <text><textPath href="#pt${i}" startOffset="50%">${top}</textPath></text>
      <text><textPath href="#pb${i}" startOffset="50%">${bottom}</textPath></text>
      <text x="55" y="61" font-family="Bricolage Grotesque, sans-serif" font-weight="700" font-size="16">${mid}</text>
    </g></g></svg>`
})

// Card 1: the candidate list (typed, with dot bars).
const moves = $('#opera-moves')
moves.innerHTML = opera.top.map((m) => `<li><span>${numbered([m.san], 10)}</span><canvas></canvas><span class="pc">${percent(m.p)}</span></li>`).join('')

// Card 2: the twelve sheets, in the order the network was trained on.
const NAMES: Record<string, string> = { p: 'pawns', n: 'knights', b: 'bishops', r: 'rooks', q: 'queens', k: 'kings' }
const ORDER = 'PNBRQKpnbrqk'.split('')
const tilt = rng(4)
$('#sheets').innerHTML = ORDER.map((t) =>
  `<figure class="sheet" style="--r:${(tilt() * 5 - 2.5).toFixed(2)}deg"><canvas aria-hidden="true"></canvas>` +
  `<figcaption><b>${t}</b>${t === t.toUpperCase() ? 'white' : 'black'} ${NAMES[t.toLowerCase()]}</figcaption></figure>`).join('')

// Card 3: facts straight from the data.
$('#share').textContent = percent(opera.share)
$('#legal').textContent = percent(opera.legalMass)
const drawQuilt = mountQuilt($('#quilt'), $('#readout'), opera)

// Card 4: the two prints.
$('#v2-move').textContent = `${numbered([mirror.v2[0].san], 2)}, ${percent(mirror.v2[0].p)}`
$('#v1-move').textContent = `${numbered([mirror.v1[0].san], 2)}, ${percent(mirror.v1[0].p)}`

// Canvases size themselves to their column: stretch, measure, then draw at that size.
function fit(c: HTMLCanvasElement): number { c.style.width = '100%'; return Math.floor(c.clientWidth) || 120 }

function drawAll(): void {
  const board = $<HTMLCanvasElement>('#opera-board')
  stampBoard(board, opera.fen, fit(board), 1, opera.top.slice(0, 3))
  moves.querySelectorAll('canvas').forEach((c, i) => dotBar(c, opera.top[i].p, fit(c), 12))
  document.querySelectorAll<HTMLCanvasElement>('.sheet canvas').forEach((c, i) => pieceSheet(c, opera.fen, ORDER[i], fit(c), 100 + i * 7))
  drawQuilt()
  const front = $<HTMLCanvasElement>('#print-v2'), back = $<HTMLCanvasElement>('#print-v1')
  // Measure both before drawing either, so one print's old size can't squeeze the other's column.
  const [frontSize, backSize] = [fit(front), fit(back)]
  stampBoard(front, mirror.fen, frontSize, 60, [mirror.v2[0]])
  stampBoard(back, mirror.fen, backSize, 70, [mirror.v1[0]], SWAP)
  document.querySelectorAll<HTMLCanvasElement>('canvas.postage').forEach((c, i) => {
    const { value = '', unit = '', piece = 'n' } = c.dataset
    postage(c, value, unit, piece, 200 + i * 11)
  })
}

// Cards land on the desk as they snap into view (and get picked up again once they leave).
const slots = [...document.querySelectorAll<HTMLElement>('.slot')]
const land = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const slot = e.target as HTMLElement
    const seen = e.intersectionRatio >= 0.55 || e.intersectionRect.height >= (e.rootBounds?.height ?? 1e9) * 0.5
    if (seen && !slot.classList.contains('landed')) { slot.classList.add('landed'); thunk() }
    else if (!e.isIntersecting && !reduce) slot.classList.remove('landed')
  }
}, { threshold: [0, 0.25, 0.4, 0.55, 0.7] })

const toggle = $<HTMLButtonElement>('.sound')
toggle.addEventListener('click', () => {
  const on = toggle.getAttribute('aria-pressed') !== 'true'
  toggle.setAttribute('aria-pressed', String(on)); toggle.textContent = on ? 'sound: on' : 'sound: off'
  setSound(on); thunk(0.8)
})

const fonts = ['800 20px "Bricolage Grotesque"', '700 12px "Courier Prime"', '12px "Courier Prime"'].map((f) => document.fonts.load(f))
void Promise.all(fonts).catch(() => []).then(() => {
  drawAll()
  if (reduce) slots.forEach((s) => s.classList.add('landed'))
  requestAnimationFrame(() => slots.forEach((s) => land.observe(s)))
  let width = innerWidth, timer = 0
  addEventListener('resize', () => {
    if (innerWidth === width) return // phones resize height when the URL bar moves; ignore
    width = innerWidth; clearTimeout(timer); timer = window.setTimeout(drawAll, 160)
  })
  // three.js only loads once the stamp card is close.
  const stage = $('#stage')
  const near = new IntersectionObserver(([e]) => {
    if (!e.isIntersecting) return
    near.disconnect()
    void import('./stamp3d').then((m) => m.mountStamp(stage, mirror.fen, reduce, () => thunk(0.55)))
  }, { rootMargin: '100% 0px' })
  near.observe(stage)
})
