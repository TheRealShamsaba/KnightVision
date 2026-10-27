// The mail itself: each chapter is a different object, but they all share the same inks and paper.
import { data, numbered, percent, type Ply } from '../data'
import { dotBar, drawPiece, inkLayer, overprint, pieceSheet, rng, sizeCanvas, stampBoard } from './ink'

const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
const decode = (b64: string): Uint8Array => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))

/** Runs `fn` the first time `el` is mostly on screen. */
export function whenSeen(el: Element, fn: () => void, threshold = 0.35): void {
  const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { io.disconnect(); fn() } }, { threshold })
  io.observe(el)
}

// ---- 2. Telegram: ticker tape, typed out a character at a time, pasted onto the form ----------
const TAPES = [
  'GRADUATION PROJECT KNIGHTVISION STOP',
  'A CHESS ENGINE THAT LEARNS FROM HUMAN GAMES STOP',
  'BY ARYAN ALISHAMSABADIFARD AND BAKER WALEED SALEM ALDHMOUR STOP',
  'SUPERVISED BY DR ALPARSLAN HORASAN STOP',
  'PRESENTED TO THE JURY JUNE 2025 STOP',
  'ONE BUG STILL HIDING STOP DETAILS TO FOLLOW STOP',
]

export function telegram(root: HTMLElement): void {
  const host = root.querySelector<HTMLElement>('[data-tapes]')!
  const words = root.querySelector<HTMLElement>('[data-tg-words]')!
  const tilt = rng(12)
  const strips = TAPES.map((text) => {
    const strip = document.createElement('p')
    strip.className = 'tape'
    strip.style.setProperty('--r', `${(tilt() * 1.4 - 0.7).toFixed(2)}deg`)
    strip.setAttribute('aria-label', text)
    host.append(strip)
    return { strip, text }
  })
  const count = (n: number) => { words.textContent = String(n) }
  if (reduce) {
    strips.forEach(({ strip, text }) => { strip.textContent = text; strip.classList.add('pasted') })
    count(TAPES.join(' ').split(' ').length)
    return
  }
  whenSeen(root, async () => {
    let total = 0
    for (const { strip, text } of strips) {
      strip.classList.add('pasted')
      for (let i = 1; i <= text.length; i++) {
        strip.textContent = text.slice(0, i)
        if (text[i - 1] === ' ' || i === text.length) count(++total)
        // The teleprinter pauses on each STOP.
        await new Promise((r) => setTimeout(r, text.slice(i - 4, i) === 'STOP' ? 260 : 26))
      }
    }
  })
}

// ---- 3. Parcel: index cards, one training record each, flipped through like a card file ----------
export function riffle(root: HTMLElement): void {
  const plies = data.opera.plies
  const result = data.opera.result
  const cards = plies.map((ply, i) => {
    const card = document.createElement('div')
    card.className = 'index-card'
    card.innerHTML = `<span class="ic-no">record ${String(i + 1).padStart(7, '0')}</span><canvas aria-hidden="true"></canvas>
      <code>{"fen": "${ply.fen.split(' ').slice(0, 2).join(' ')} …",<br>&nbsp;"move": "${ply.san}", "result": "${result}"}</code>`
    return card
  })
  let order = cards.map((_, i) => i)
  const place = () => order.forEach((c, depth) => {
    const card = cards[c]
    card.style.setProperty('--d', String(depth))
    card.style.zIndex = String(cards.length - depth)
    card.hidden = depth > 5
  })
  cards.forEach((c) => root.append(c))
  place()
  // Boards are drawn only once their card nears the top of the pile.
  const drawn = new Set<number>()
  const draw = () => order.slice(0, 3).forEach((i) => {
    if (drawn.has(i)) return
    drawn.add(i)
    stampBoard(cards[i].querySelector('canvas')!, plies[i].fen, 128, 300 + i)
  })
  draw()
  const next = () => {
    const top = cards[order[0]]
    top.classList.add('flip')
    setTimeout(() => {
      top.classList.remove('flip')
      order = [...order.slice(1), order[0]]
      place(); draw()
    }, reduce ? 0 : 380)
  }
  root.addEventListener('click', next)
  root.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); next() } })
}

// ---- 4. Envelope: the twelve sheets, drawn for whichever position the scrubber is on ----------
const ORDER = 'PNBRQKpnbrqk'.split('')
const NAMES: Record<string, string> = { p: 'pawns', n: 'knights', b: 'bishops', r: 'rooks', q: 'queens', k: 'kings' }

export function envelope(root: HTMLElement): (ply: Ply) => void {
  const env = root.querySelector<HTMLElement>('[data-env]')!
  const sheets = root.querySelector<HTMLElement>('#sheets')!
  const board = root.querySelector<HTMLCanvasElement>('#sees-board')!
  sheets.innerHTML = ORDER.map((t, i) =>
    `<figure class="sheet" style="--i:${i}"><canvas aria-hidden="true"></canvas>` +
    `<figcaption><b>${t}</b>${t === t.toUpperCase() ? 'white' : 'black'} ${NAMES[t.toLowerCase()]}</figcaption></figure>`).join('')
  whenSeen(env, () => env.classList.add('open'), 0.45)
  return (ply) => {
    const w = Math.floor(board.parentElement!.clientWidth)
    if (w) stampBoard(board, ply.fen, Math.min(w, 260), 40)
    sheets.querySelectorAll('canvas').forEach((c, i) => pieceSheet(c, ply.fen, ORDER[i], Math.max(60, Math.floor(c.parentElement!.clientWidth) - 8), 100 + i * 7))
  }
}

// ---- 5. Blueprint: eight rooms, each lined with its real activation maps -------------------------
// How many squares out a corner square can see after each room: a 3×3 look adds one, a block adds two.
const REACH = [1, 2, 4, 6, 8, 10, 12, 12]

function chalkTile(c: HTMLCanvasElement, b64: string, size: number): void {
  const g = sizeCanvas(c, size, size)
  const cell = size / 8
  g.strokeStyle = 'rgba(242, 237, 227, .55)'; g.lineWidth = 0.6
  g.strokeRect(0.5, 0.5, size - 1, size - 1)
  g.fillStyle = '#F2EDE3'
  decode(b64).forEach((v, i) => {
    const r = cell * 0.48 * Math.sqrt(v / 255)
    if (r < 0.35) return
    g.beginPath(); g.arc((i % 8 + 0.5) * cell, (Math.floor(i / 8) + 0.5) * cell, r, 0, 7); g.fill()
  })
}

function reachGrid(radius: number): string {
  let cells = ''
  for (let row = 0; row < 8; row++) for (let col = 0; col < 8; col++) {
    const d = Math.max(col, 7 - row)
    cells += `<i class="${d === 0 ? 'o' : d <= radius ? 's' : ''}"></i>`
  }
  return `<div class="reach" aria-hidden="true">${cells}</div>`
}

export function blueprint(root: HTMLElement): () => void {
  const rooms = root.querySelector<HTMLElement>('#rooms')!
  const stages = data.layers.stages
  const millions = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}m` : `${Math.round(n / 1e3)}k`)
  rooms.innerHTML = stages.map((s, i) => `
    <li class="room${s.name.startsWith('Residual') ? ' res' : ''}">
      <p class="room-no">room ${i + 1}</p>
      <p class="room-name">${s.name.toLowerCase()}</p>
      <div class="tiles">${s.maps.map(() => '<canvas></canvas>').join('')}</div>
      <p class="room-spec">${s.channels} ch · ${millions(s.parameters)} w</p>
      ${reachGrid(REACH[i])}
    </li>`).join('') + '<li class="belt" aria-hidden="true"><span class="letter"></span></li>'
  root.querySelector('[data-fill="weights-note"]')!.textContent =
    `${(data.parameters / 1e6).toFixed(1)} million weights in all. the small grids show how far the bottom-left corner can see by then.`
  return () => rooms.querySelectorAll<HTMLElement>('.room').forEach((room, i) => {
    room.querySelectorAll('canvas').forEach((c, j) => chalkTile(c, stages[i].maps[j], 44))
  })
}

// ---- 7. Marked paper: the same question, answered untrained and trained ---------------------------
export function graded(root: HTMLElement): () => void {
  const l = data.learning
  const list = (id: string, top: typeof l.trained.top) => {
    root.querySelector(id)!.innerHTML = top.slice(0, 4).map((m) =>
      `<li class="${m.san === l.answer ? 'right' : ''}"><span>${numbered([m.san], 2)}</span><canvas></canvas><span>${percent(m.p)}</span></li>`).join('')
  }
  list('#guess-before', l.untrained.top)
  list('#guess-after', l.trained.top)
  root.querySelector('[data-grade]')!.innerHTML =
    `<span class="grade-label">loss</span><s>${l.untrained.loss.toFixed(2)}</s><b>${l.trained.loss.toFixed(2)}</b><span class="grade-note">lower is better</span>`
  root.querySelector('#learn-remark')!.innerHTML =
    `the person played <b>${numbered([l.answer], 2)}</b>. untrained, i spread my guess over all ${l.untrained.legalMoves} legal moves like a dartboard and gave it ${percent(l.untrained.answerShare)}. ` +
    `after training: ${percent(l.trained.answerShare)}. nobody taught me to develop knights. i picked it up from the mail.`
  const t = l.training
  root.querySelector('#learn-facts')!.innerHTML = ([
    ['positions shown', t.positions.toLocaleString('en')],
    ['kept aside to check', `${t.validation * 100}%`],
    ['positions per step', t.batch.toLocaleString('en')],
    ['optimiser', `${t.optimizer}, rate ${t.lr}`],
    ['best version', `after ${data.checkpoint.epoch} passes`],
    ['held-back loss', String(data.checkpoint.validationLoss)],
  ] as const).map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('')
  return () => {
    for (const [id, top, seed] of [['#learn-before', l.untrained.top, 81], ['#learn-after', l.trained.top, 82]] as const) {
      const c = root.querySelector<HTMLCanvasElement>(id)!
      stampBoard(c, l.fen, Math.min(240, Math.floor(c.parentElement!.clientWidth)), seed, top.slice(0, 4))
    }
    root.querySelectorAll<HTMLCanvasElement>('.guesses canvas').forEach((c, i) => {
      const top = i < 4 ? l.untrained.top : l.trained.top
      dotBar(c, top[i % 4].p, 80, 10)
    })
  }
}

// ---- 9. Stamp album: every game against Stockfish is a stamp ----------------------------------
export function albumPage(c: HTMLCanvasElement, width: number): void {
  // A page laid out at zero width (a hidden tab) draws on the next resize instead.
  if (!width) return
  const games = data.elo?.games ?? []
  const levels = [...new Set(games.map((g) => g.elo))].sort()
  const cols = width < 460 ? 10 : 20
  const sw = width / cols, sh = sw * 1.2
  const per = levels.map((l) => games.filter((g) => g.elo === l).sort((a, b) => b.score - a.score || b.plies - a.plies))
  const rows = per.map((p) => Math.ceil(p.length / cols))
  const band = 30
  const height = rows.reduce((a, r) => a + band + r * sh + 18, 0)
  const g = sizeCanvas(c, width, height)
  const spots: { x: number; y: number; score: number }[] = []
  const labels: { y: number; text: string }[] = []
  let y = 0
  per.forEach((list, k) => {
    const won = list.filter((x) => x.score === 1).length, drawn = list.filter((x) => x.score === 0.5).length
    // Narrow pages get the short form, so the label never runs off the album page.
    const lost = list.length - won - drawn
    labels.push({ y: y + 20, text: width < 460 ? `vs ${levels[k]}: ${won} won · ${drawn} drawn · ${lost} lost` : `against stockfish ${levels[k]}: ${won} won, ${drawn} drawn, ${lost} lost` })
    y += band
    list.forEach((x, i) => spots.push({ x: (i % cols) * sw, y: y + Math.floor(i / cols) * sh, score: x.score }))
    y += rows[k] * sh + 18
  })
  const m = sw * 0.12, w = sw - 2 * m, h = sh - 2 * m
  const orange = inkLayer(width, height, 'orange', 61, (t) => {
    for (const s of spots) if (s.score === 1) t.fillRect(s.x + m, s.y + m, w, h)
  })
  const purple = inkLayer(width, height, 'purple', 62, (t) => {
    for (const s of spots) {
      if (s.score === 1) drawPiece(t, 'n', s.x + m + w * 0.15, s.y + m + h * 0.1, w * 0.7)
      if (s.score === 0.5) { t.fillRect(s.x + m, s.y + m + h / 2, w, h / 2); drawPiece(t, 'n', s.x + m + w * 0.2, s.y + m + h * 0.14, w * 0.6, true) }
    }
  })
  const blue = inkLayer(width, height, 'blue', 63, (t) => {
    t.lineWidth = Math.max(0.8, sw * 0.045)
    for (const s of spots) {
      if (s.score !== 0) continue
      t.globalAlpha = 0.5; drawPiece(t, 'n', s.x + m + w * 0.2, s.y + m + h * 0.14, w * 0.6, true); t.globalAlpha = 1
      // The cancellation: three wavy bars across the stamp.
      for (let k = 0; k < 3; k++) {
        t.beginPath()
        for (let i = 0; i <= 12; i++) {
          const px = s.x + m * 0.4 + (i / 12) * (w + m * 1.2), py = s.y + m + h * (0.3 + k * 0.2) + Math.sin(i * 1.3) * h * 0.04
          if (i) t.lineTo(px, py); else t.moveTo(px, py)
        }
        t.stroke()
      }
    }
  })
  const black = inkLayer(width, height, 'black', 64, (t) => {
    t.font = '13px "Courier Prime"'
    for (const l of labels) t.fillText(l.text, 0, l.y)
  })
  overprint(g, [[orange, 'orange'], [purple, 'purple'], [blue, 'blue'], [black, 'black']])
  // Perforations: punch the paper colour back in around every stamp.
  g.fillStyle = '#F2EDE3'
  for (const s of spots) for (let k = 0; k <= 6; k++) {
    for (const [px, py] of [[s.x + m + (k / 6) * w, s.y + m], [s.x + m + (k / 6) * w, s.y + m + h], [s.x + m, s.y + m + (k / 6) * h], [s.x + m + w, s.y + m + (k / 6) * h]]) {
      g.beginPath(); g.arc(px, py, Math.max(0.8, sw * 0.035), 0, 7); g.fill()
    }
  }
}
