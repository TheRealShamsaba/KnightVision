// Play by mail: you post a move on the stamped board, and the network writes back a postcard with
// its reply, the moves it also considered, and a postmark. The replies pile up into a stack.
import { Chess, type Square } from 'chess.js'
import { numbered, percent } from '../data'
import type { Thought } from '../engine'
import { FILES } from '../world/board3d'
import { dotBar, sizeCanvas, stampBoard } from './ink'
import { thunk } from './sound'

type Engine = typeof import('../engine')

export function mountMail(root: HTMLElement): () => void {
  const boardCanvas = root.querySelector<HTMLCanvasElement>('.mail-board canvas.ink')!
  const marks = root.querySelector<HTMLCanvasElement>('.mail-board canvas.marks')!
  const status = root.querySelector<HTMLElement>('[data-status]')!
  const stack = root.querySelector<HTMLElement>('[data-stack]')!
  const wake = root.querySelector<HTMLButtonElement>('[data-wake]')!
  const meter = root.querySelector<HTMLElement>('[data-meter]')!
  const undo = root.querySelector<HTMLButtonElement>('[data-undo]')!

  let game = new Chess()
  let human: 'w' | 'b' = 'w'
  let engine: Engine | null = null
  let thinking = false
  let selected: Square | null = null
  let generation = 0
  let size = 0

  const flipped = () => human === 'b'
  // Board geometry matches stampBoard: a 6.5% margin, then eight squares.
  const geometry = () => { const pad = size * 0.065; return { pad, sq: (size - 2 * pad) / 8 } }
  const squareAt = (x: number, y: number): Square | null => {
    const { pad, sq } = geometry()
    let f = Math.floor((x - pad) / sq), r = 7 - Math.floor((y - pad) / sq)
    if (f < 0 || f > 7 || r < 0 || r > 7) return null
    if (flipped()) { f = 7 - f; r = 7 - r }
    return `${FILES[f]}${r + 1}` as Square
  }
  const topLeft = (s: string): [number, number] => {
    const { pad, sq } = geometry()
    let f = FILES.indexOf(s[0]), r = Number(s[1]) - 1
    if (flipped()) { f = 7 - f; r = 7 - r }
    return [pad + f * sq, pad + (7 - r) * sq]
  }

  function drawBoard() {
    size = Math.floor(boardCanvas.parentElement!.clientWidth)
    if (!size) return
    stampBoard(boardCanvas, game.fen(), size, 500 + game.history().length, [], undefined, flipped())
    drawMarks()
  }

  function drawMarks() {
    const g = sizeCanvas(marks, size, size)
    const { sq } = geometry()
    g.strokeStyle = '#111'; g.fillStyle = '#111'; g.lineWidth = 1.5
    const last = game.history({ verbose: true }).at(-1)
    if (last) for (const s of [last.from, last.to]) {
      const [x, y] = topLeft(s)
      g.setLineDash([3, 3]); g.strokeRect(x + 2, y + 2, sq - 4, sq - 4); g.setLineDash([])
    }
    if (!selected) return
    const [x, y] = topLeft(selected)
    g.lineWidth = 2.5; g.strokeRect(x + 1, y + 1, sq - 2, sq - 2)
    for (const m of game.moves({ square: selected, verbose: true })) {
      const [tx, ty] = topLeft(m.to)
      g.beginPath(); g.arc(tx + sq / 2, ty + sq / 2, sq * 0.11, 0, 7); g.fill()
    }
  }

  const over = (): string | null => {
    if (game.isCheckmate()) return game.turn() === human ? 'checkmate. it got you. good game.' : 'checkmate. you beat me. fair and square.'
    if (game.isDraw()) return 'a draw. we can both live with that.'
    return null
  }
  const setStatus = (text?: string) => {
    status.textContent = text ?? over() ?? (thinking ? 'writing back…' : engine ? 'your move. click a piece, then a square.' : 'the letterbox is closed.')
  }

  // A reply postcard: the move, a stamped board after it, the alternatives as dot bars, a postmark.
  function postReply(thoughts: Thought[], moveNumber: string) {
    const best = thoughts[0]
    const card = document.createElement('article')
    card.className = 'reply'
    card.style.setProperty('--r', `${(Math.random() * 4 - 2).toFixed(2)}deg`)
    const others = thoughts.slice(1, 4)
    card.innerHTML = `
      <header><span class="typed">from: knightvision</span><span class="mini-mark">${moveNumber}</span></header>
      <div class="reply-body">
        <canvas class="reply-board" aria-hidden="true"></canvas>
        <div>
          <p class="reply-move">${numbered([best.san], game.history().length - 1)}</p>
          <p class="typed small">one look, ${percent(best.p)} sure.${others.length ? ' also thought about:' : ''}</p>
          <ol class="also">${others.map((t) => `<li><span>${t.san}</span><canvas></canvas><span>${percent(t.p)}</span></li>`).join('')}</ol>
        </div>
      </div>`
    stack.prepend(card)
    requestAnimationFrame(() => {
      const board = card.querySelector<HTMLCanvasElement>('.reply-board')!
      stampBoard(board, game.fen(), 112, 900 + game.history().length, [{ san: best.san, uci: best.from + best.to, p: best.p }], undefined, flipped())
      card.querySelectorAll<HTMLCanvasElement>('.also canvas').forEach((c, i) => dotBar(c, others[i].p, 70, 10))
      card.classList.add('in')
    })
    // Keep the stack readable: the latest few replies, older ones tucked away.
    ;[...stack.children].slice(6).forEach((c) => c.remove())
  }

  async function reply() {
    if (!engine || over() || game.turn() === human) return
    const mine = generation
    thinking = true
    setStatus()
    const thoughts = await engine.think(game)
    // A beat of "postage", so the reply feels sent rather than instant.
    await new Promise((r) => setTimeout(r, 450))
    if (mine !== generation) return
    const best = thoughts[0]
    const moveNumber = game.turn() === 'w' ? `${game.moveNumber()}.` : `${game.moveNumber()}…`
    game.move({ from: best.from, to: best.to, promotion: 'q' })
    thinking = false
    postReply(thoughts, moveNumber)
    thunk(0.9)
    drawBoard()
    setStatus()
    undo.disabled = false
  }

  function tryMove(from: Square, to: Square): boolean {
    try { game.move({ from, to, promotion: 'q' }) } catch { return false }
    selected = null
    thunk(0.5)
    drawBoard()
    setStatus()
    void reply()
    return true
  }

  marks.addEventListener('pointerdown', (e) => {
    // offsetX/Y are in the canvas's own space, so the card's tilt doesn't throw the aim off.
    const square = squareAt(e.offsetX, e.offsetY)
    if (!square) return
    if (selected && selected !== square && tryMove(selected, square)) return
    const piece = game.get(square)
    const mine = engine && !thinking && !over() && game.turn() === human && piece && piece.color === human
    selected = mine && selected !== square ? square : null
    drawMarks()
  })

  function newGame(colour: 'w' | 'b') {
    generation++
    game = new Chess()
    human = colour
    thinking = false
    selected = null
    stack.replaceChildren()
    undo.disabled = true
    drawBoard()
    setStatus()
    void reply()
  }
  root.querySelectorAll<HTMLButtonElement>('[data-new]').forEach((b) => b.addEventListener('click', () => newGame(b.dataset.new as 'w' | 'b')))
  undo.addEventListener('click', () => {
    if (thinking) return
    game.undo()
    if (game.turn() !== human) game.undo()
    stack.firstElementChild?.remove()
    selected = null
    undo.disabled = game.history().length === 0
    drawBoard()
    setStatus()
  })

  wake.addEventListener('click', async () => {
    wake.disabled = true
    meter.hidden = false
    setStatus('opening the letterbox…')
    try {
      const mod = await import('../engine')
      await mod.loadEngine((f) => { (meter.firstElementChild as HTMLElement).style.width = `${Math.round(f * 100)}%` })
      engine = mod
      root.classList.add('open')
      setStatus()
      void reply()
    } catch {
      setStatus('the letterbox jammed. check your connection and try again.')
      wake.disabled = false
      meter.hidden = true
    }
  })

  setStatus()
  return drawBoard
}
