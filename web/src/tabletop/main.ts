// The Desk: KnightVision as a working model on one desk, under one lamp. Eight stops stand around
// the desk; scrolling glides the camera from one to the next like a model photographer, and the lamp
// swings to follow. Each stop has a card that explains one idea.
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import './desk.css'
import { data, numbered, percent, quiltBytes } from '../data'
import { Bench, type Wood } from './bench'
import { CardBox } from './cardbox'
import { Desk } from './desk'
import { Pegboard } from './pegboard'
import { tracingSides } from './pencil'
import { PlayBoard } from './playboard'
import { Slides } from './slides'
import { Clack } from './sound'
import { Tally } from './tally'
import { grainCanvas, toTexture } from './textures'
import { tiltShift } from './tiltshift'
import { Tracing } from './tracing'

const opera = data.opera.plies[10]
const mirror = data.mirror[2]
const learn = data.learning
const elo = data.elo
const V1: Record<string, string> = { k: 'p', q: 'n', r: 'b', b: 'r', n: 'q', p: 'k' }
const reducedQuery = matchMedia('(prefers-reduced-motion: reduce)')
const deg = THREE.MathUtils.degToRad

// Where each stop stands on the desk, and the angle the camera looks at it from.
const STOP = {
  board: { at: new THREE.Vector3(0, 0, 0), turn: 20 },
  cards: { at: new THREE.Vector3(-12, 0, -12), turn: -135 },
  trays: { at: new THREE.Vector3(-19, 0, 7), turn: -25 },
  slides: { at: new THREE.Vector3(-3, 0, 19), turn: -10 },
  pegs: { at: new THREE.Vector3(15, 0, 11), turn: 45 },
  paper: { at: new THREE.Vector3(20, 0, -8), turn: 100 },
  tally: { at: new THREE.Vector3(4, 0, -19), turn: 170 },
  play: { at: new THREE.Vector3(-22, 0, 23), turn: -30 },
}

// ---------- The cards ----------

const wins = elo?.games.filter((g) => g.score === 1).length ?? 0
const draws = elo?.games.filter((g) => g.score === 0.5).length ?? 0
const card = (n: number, title: string, body: string) =>
  `<section class="beat"><div class="card"><p class="stop">stop ${n} of 8</p><h2>${title}</h2>${body}</div></section>`
const idea = (name: string, text: string) => `<p class="idea"><b>${name}</b>${text}</p>`

const app = document.querySelector<HTMLElement>('#app')!
app.innerHTML = `
<canvas class="table" aria-label="A model of the KnightVision network built on a desk"></canvas>
<button class="sound" type="button" aria-pressed="false">Sound off</button>
<main>
  <section class="beat"><div class="card">
    <h1>KnightVision</h1>
    <p class="lead">A chess network that never calculates. It looks once, and moves.</p>
    <p>This desk is a working model of how it works, in eight stops. Scroll, and the lamp follows you round.</p>
    <p class="note">A graduation project from Istanbul Aydın University, June 2025. This is version 2.</p>
  </div></section>

  ${card(1, 'One look', `
    <p>Engines like Stockfish think by searching: they play out millions of possible futures before choosing. KnightVision doesn't search at all. It looks at the position once and names a move, the way a strong player sometimes just <em>sees</em> it.</p>
    <p>The Opera Game, Paris 1858, White to play. The red pencil is its opinion; heavier means surer.</p>
    <ol class="candidates">${opera.top.map((m) => `<li><b>${m.san}</b><i style="--p:${m.p.toFixed(3)}"></i><span>${percent(m.p)}</span></li>`).join('')}</ol>
    ${idea('The idea: a policy network', 'A function from a position to a probability for every move. With no lookahead it answers instantly, but it can never double-check its own tactics.')}
    <p class="fact">${(data.parameters / 1e6).toFixed(1)}m weights · 0 positions searched${elo ? ` · ≈${elo.rating} elo` : ''}</p>`)}

  ${card(2, 'It learned by imitation', `
    <p>We took 84,000 human games and cut every one into positions: the board before a move, the move the person played, and how the game ended. About five million cards like the ones in this box.</p>
    <p>For each card it guessed the move. Whenever it was wrong, its weights were nudged so the right move became a little more likely. Repeat five million times.</p>
    ${idea('The idea: supervised learning', `The score it's trained on is the loss, −ln(the probability it gave the right answer). After 1.e4 e5, where the person played ${learn.answer}: ${learn.untrained.loss.toFixed(2)} untrained, ${learn.trained.loss.toFixed(2)} trained. Lower is better.`)}
    <p class="fact">${learn.training.optimizer} · batches of ${learn.training.batch.toLocaleString('en')} · best after ${data.checkpoint.epoch} passes</p>`)}

  ${card(3, 'What it sees', `
    <p>A network only understands numbers, so the board is sorted into twelve trays: white pawns, knights, bishops, rooks, queens and king, then black's. Each tray is an 8×8 grid with a 1 wherever that piece stands.</p>
    <p>That's 12 × 64 = 768 numbers, and it's everything it gets. Notice what's missing: whose turn it is, castling rights, en passant.</p>
    ${idea('The idea: one-hot planes', 'The usual way to feed a board game to a network. AlphaZero does the same, plus extra planes for the rules this one never sees.')}
    <p class="fact">input shape: 12 × 8 × 8</p>`)}

  ${card(4, 'Inside: eight stages', `
    <p>Each stage is a convolution: a 3×3 window slides over the board, so every square only talks to its neighbours. Stack them and news travels. By stage five, the corner square a1 can feel the whole board.</p>
    <p>The slides show real readings for the Opera position: each panel is one channel, a pattern detector the network invented for itself. The ball is the single forward pass, through once, never back.</p>
    ${idea('The idea: residual blocks', 'Stages 3 to 7 add their output to their input instead of replacing it, so the signal can skip through. That shortcut is what lets deep networks train at all.')}
    <p class="fact">5 residual blocks × 512 channels</p>`)}

  ${card(5, 'What it says back', `
    <p>Out come 4,096 numbers: every square a piece could leave × every square it could land on. Each big block on the pegboard is a from-square, each peg in it a to-square, and peg height is belief.</p>
    <p>Most pegs are moves that can't happen, and it learned to keep them short. To play, we throw out the illegal ones and take the tallest peg left: the red flag.</p>
    ${idea('The idea: softmax', `Turns raw scores into probabilities that add up to 1. Here ${percent(opera.legalMass)} of all its belief landed on legal moves, though nobody told it the rules.`)}
    <p class="fact">${opera.san}: ${percent(opera.share)} of its legal belief, the move Morphy played</p>`)}

  ${card(6, 'The bug', `
    <p>For two years, every program that used the network fed it the board in a different order from the one it trained on. Kings arrived as pawns, queens as knights, and every answer was read upside down. So after 1.e4 e5 it played <span class="pick">${numbered([mirror.v1[0].san], 2)}</span>.</p>
    <p>Nothing crashed. It kept giving confident answers, just about a board that doesn't exist. That's why nobody noticed.</p>
    ${idea('The lesson', 'Training and playing must share one encoder, and a test should prove they agree. Now one function does both, and a test replays random games checking every move.')}
    <div class="tools"><button class="tool flip" type="button">Turn the sheet over</button></div>
    <p class="fact">fixed: ${numbered([mirror.v2[0].san], 2)} ${percent(mirror.v2[0].p)}, instead of ${numbered([mirror.v1[0].san], 2)} ${percent(mirror.v1[0].p)}</p>`)}

  ${card(7, 'How strong is one look?', `
    <p>It played ${elo?.games.length ?? 160} games against Stockfish turned down to 1320 and 1400 Elo, with Stockfish on a real clock. It won ${wins} and drew ${draws}: red pegs and maple pegs on the board.</p>
    <p>From those results, the most likely rating is ${elo?.rating ?? 935}, give or take ${elo?.margin ?? 99}. Not strong, but far from random, and all of it imitation with zero lookahead.</p>
    ${idea('The idea: Elo', 'Turns results into a rating. A 400-point gap means the stronger side is expected to score about 91%.')}
    <p class="fact">one win ended 17.Nc7#, mate with a knight</p>`)}

  ${card(8, 'Your move', `
    <p>This is the real network, squeezed to 25 MB and running on your device. Before it moves, it pencils in what it's considering.</p>
    <div class="tools"><button class="tool wake" type="button">Wake it up</button></div>
    <div class="meter" hidden><span></span></div>
    <p class="status">Asleep.</p>
    <ol class="candidates play-candidates"></ol>
    <div class="tools">
      <button class="tool" type="button" data-new="w">New game as White</button>
      <button class="tool" type="button" data-new="b">as Black</button>
      <button class="tool" type="button" data-undo disabled>Take back</button>
    </div>
    <p class="note">Next for KnightVision: a search on top of this instinct, the way AlphaZero does, so it can check before it moves.</p>`)}
</main>`

// ---------- Renderer, room, lamp ----------

const canvas = app.querySelector<HTMLCanvasElement>('.table')!
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
renderer.shadowMap.enabled = true
renderer.shadowMap.type = THREE.PCFSoftShadowMap
renderer.toneMapping = THREE.ACESFilmicToneMapping
renderer.toneMappingExposure = 1.0

const scene = new THREE.Scene()
scene.background = new THREE.Color('#17110D')
scene.fog = new THREE.Fog('#17110D', 70, 170)
const camera = new THREE.PerspectiveCamera(30, 1, 0.5, 400)
scene.add(new THREE.HemisphereLight('#FFF1DC', '#3A2516', 1.15))

const desk = new Desk()
scene.add(desk.group)

const composer = new EffectComposer(renderer)
composer.addPass(new RenderPass(scene, camera))
composer.addPass(new OutputPass())
const lens = tiltShift()
composer.addPass(lens)

const sound = new Clack()
const soundButton = app.querySelector<HTMLButtonElement>('.sound')!
soundButton.addEventListener('click', async () => {
  const on = await sound.toggle()
  soundButton.textContent = on ? 'Sound on' : 'Sound off'
  soundButton.setAttribute('aria-pressed', String(on))
  sound.play()
})

// Labels on the wood and paper use the web fonts, so build once they're in (or after 2 s regardless).
const fonts = Promise.all(['500 40px Jost', 'italic 28px Literata', '30px "Courier Prime"'].map((f) => document.fonts.load(f)))
Promise.race([fonts, new Promise((r) => setTimeout(r, 2000))]).then(build, build)

function build() {
  const grain = grainCanvas(512, 7)
  const wood: Wood = {
    grain,
    maple: new THREE.MeshStandardMaterial({ color: '#E3CFA4', map: toTexture(grain), roughness: 0.58 }),
    beech: new THREE.MeshStandardMaterial({ color: '#B9824E', map: toTexture(grainCanvas(512, 21)), roughness: 0.42 }),
  }
  const place = (obj: THREE.Object3D, stop: { at: THREE.Vector3; turn: number }) => {
    obj.position.copy(stop.at)
    obj.rotation.y = deg(stop.turn)
    scene.add(obj)
  }

  const bench = new Bench(wood, opera.fen, opera.top.slice(0, 5), STOP.trays.at)
  scene.add(bench.group)
  const cards = new CardBox(wood)
  place(cards.group, STOP.cards)
  const slides = new Slides(wood)
  place(slides.group, STOP.slides)
  const pegs = new Pegboard(quiltBytes(opera), opera.uci, wood)
  place(pegs.group, STOP.pegs)
  const [front, back] = tracingSides(
    { fen: mirror.fen, title: 'What it learned', move: mirror.v2[0] },
    { fen: mirror.fen, title: 'What v1 showed it', move: mirror.v1[0], swap: V1 },
  )
  const tracing = new Tracing(front, back, wood)
  place(tracing.group, STOP.paper)
  const tally = new Tally(wood)
  place(tally.group, STOP.tally)
  const play = new PlayBoard(wood)
  const playHolder = new THREE.Group()
  playHolder.add(play.group)
  place(playHolder, STOP.play)

  // ---------- The game panel ----------

  const panel = app.querySelectorAll<HTMLElement>('.card')[8]
  const status = panel.querySelector<HTMLElement>('.status')!
  const list = panel.querySelector<HTMLElement>('.play-candidates')!
  const wake = panel.querySelector<HTMLButtonElement>('.wake')!
  const meter = panel.querySelector<HTMLElement>('.meter')!
  const undo = panel.querySelector<HTMLButtonElement>('[data-undo]')!
  play.onChange = ({ status: text, thoughts, canUndo }) => {
    status.textContent = text
    undo.disabled = !canUndo
    if (thoughts) list.innerHTML = thoughts.slice(0, 5).map((t) => `<li><b>${t.san}</b><i style="--p:${t.p.toFixed(3)}"></i><span>${percent(t.p)}</span></li>`).join('')
  }
  play.onLand = () => sound.play()
  wake.addEventListener('click', async () => {
    wake.disabled = true
    meter.hidden = false
    status.textContent = 'Waking up…'
    try {
      await play.wake((f) => { (meter.firstElementChild as HTMLElement).style.width = `${Math.round(f * 100)}%` })
      wake.parentElement!.hidden = true
      meter.hidden = true
    } catch {
      status.textContent = 'It couldn’t load. Check your connection and try again.'
      wake.disabled = false
    }
  })
  panel.querySelectorAll<HTMLButtonElement>('[data-new]').forEach((b) => b.addEventListener('click', () => { list.innerHTML = ''; play.newGame(b.dataset.new as 'w' | 'b') }))
  undo.addEventListener('click', () => play.undo())

  const ray = new THREE.Raycaster()
  const pointer = new THREE.Vector2()
  let down = { x: 0, y: 0 }
  canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY } })
  canvas.addEventListener('pointerup', (e) => {
    if (shown < 7.6 || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 6) return
    pointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1)
    ray.setFromCamera(pointer, camera)
    play.click(ray)
  })

  let flipTouched = false
  const flipButton = app.querySelector<HTMLButtonElement>('.flip')!
  flipButton.addEventListener('click', () => { flipTouched = true; tracing.set(!tracing.flipped) })

  // ---------- Camera: one pose per stop ----------

  type Pose = { at: THREE.Vector3; w: number; h: number; elev: number; turn: number }
  let wide = true, rowOfSix = true
  const poses = (): Pose[] => [
    { at: new THREE.Vector3(1, 0, 3), w: 31, h: 25, elev: 54, turn: 28 },
    { at: new THREE.Vector3(0, 0.3, 0.2), w: 4.9, h: 4.1, elev: 48, turn: STOP.board.turn },
    { at: STOP.cards.at.clone().setY(2.2), w: 4.4, h: 4.4, elev: 30, turn: STOP.cards.turn },
    { at: STOP.trays.at.clone(), w: rowOfSix ? 10.6 : 5.4, h: rowOfSix ? 4.6 : 8.4, elev: 62, turn: STOP.trays.turn },
    // The rack is shot from one end at three-quarters, so the slides recede like pages in a file.
    { at: STOP.slides.at.clone().setY(2.2), w: rowOfSix ? 8.5 : 6, h: 3.6, elev: 20, turn: STOP.slides.turn - 32 },
    { at: STOP.pegs.at.clone().setY(0.6), w: 6.1, h: 5.4, elev: 47, turn: STOP.pegs.turn },
    { at: STOP.paper.at.clone().setY(0.3), w: 5.5, h: 5.1, elev: 66, turn: STOP.paper.turn },
    { at: STOP.tally.at.clone().setY(0.5), w: 6.4, h: 4.6, elev: 52, turn: STOP.tally.turn },
    { at: STOP.play.at.clone().setY(0.3), w: 4.9, h: 4.6, elev: 56, turn: STOP.play.turn },
  ]
  let POSES = poses()

  function resize() {
    const w = innerWidth, h = innerHeight
    wide = w >= 820
    rowOfSix = w / h > 0.9
    // Pixel ratio can change after load (a window dragged to another screen), so set it every time.
    const ratio = Math.min(devicePixelRatio, 2)
    renderer.setPixelRatio(ratio)
    composer.setPixelRatio(ratio)
    renderer.setSize(w, h, false)
    composer.setSize(w, h)
    lens.uniforms.resolution.value.set(w * renderer.getPixelRatio(), h * renderer.getPixelRatio())
    // The sharp band follows the subject: centred beside the cards on desktop, high on phones.
    lens.uniforms.focus.value = wide ? 0.5 : 0.7
    lens.uniforms.amount.value = wide ? 7 : 4
    camera.aspect = w / h
    if (wide) camera.setViewOffset(w, h, -w * 0.18, 0, w, h)
    else camera.setViewOffset(w, h, 0, h * 0.2, w, h)
    camera.updateProjectionMatrix()
    bench.layout(rowOfSix ? 6 : 3)
    POSES = poses()
  }
  addEventListener('resize', resize)
  resize()

  const sections = [...app.querySelectorAll<HTMLElement>('.beat')]
  const last = sections.length - 1
  /** Scroll as a stop number: 0 with the first card centred, 8 with the last. */
  function beatFromScroll(): number {
    const mid = innerHeight / 2
    for (let i = 0; i < sections.length; i++) {
      const r = sections[i].getBoundingClientRect()
      if (mid < r.bottom || i === last) return Math.min(last, Math.max(0, i + (mid - r.top) / r.height - 0.5))
    }
    return 0
  }

  const clamp01 = (x: number) => Math.min(1, Math.max(0, x))
  const smooth = (a: number, b: number, x: number) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t) }
  const pose = { at: new THREE.Vector3(), w: 0, h: 0, elev: 0, turn: 0 }

  function placeCamera(beat: number, reduced: boolean, time: number) {
    // Hold while a card is read; glide through the middle of the gap, rising like a crane on the way.
    const g = Math.min(last - 1, Math.floor(beat)), raw = beat - g
    const f = reduced ? Math.round(clamp01(raw)) : smooth(0.22, 0.78, raw)
    const a = POSES[g], b = POSES[g + 1]
    pose.at.lerpVectors(a.at, b.at, f)
    for (const k of ['w', 'h', 'elev'] as const) pose[k] = a[k] + (b[k] - a[k]) * f
    // Turn the short way round the desk.
    let dt = b.turn - a.turn
    dt = ((dt + 540) % 360) - 180
    pose.turn = a.turn + dt * f
    const crane = Math.sin(Math.PI * f)
    const t = Math.tan(deg(camera.fov / 2))
    const usable = wide ? [1, 0.6] : [0.58, 1]
    const distance = Math.max(pose.h / (t * usable[0]), pose.w / (t * camera.aspect * usable[1])) * (1 + crane * 0.35)
    const e = deg(pose.elev + crane * 14), r = deg(pose.turn + (reduced ? 0 : Math.sin(time * 0.00011) * 1.5))
    camera.position.set(Math.sin(r) * Math.cos(e), Math.sin(e), Math.cos(r) * Math.cos(e)).multiplyScalar(distance).add(pose.at)
    camera.lookAt(pose.at)
    desk.point(pose.at)
  }

  // ---------- Loop ----------

  let shown = beatFromScroll()
  let lastBeat = shown
  let then = performance.now()
  if (!reducedQuery.matches && shown < 1.5) bench.setUp()
  if (shown >= 6.1) tracing.set(true)
  const near = (stop: number) => Math.abs(shown - stop) < 0.55
  // Dev only: pin the camera to a stop without scrolling (for checking scenes).
  let pinned: number | null = null
  if (import.meta.env.DEV) Object.assign(window, { __desk: { go: (b: number) => { pinned = b; shown = b } } })

  renderer.setAnimationLoop((now) => {
    const dt = Math.min(0.05, (now - then) / 1000)
    then = now
    const reduced = reducedQuery.matches
    const target = pinned ?? beatFromScroll()
    shown = reduced ? target : shown + (target - shown) * (1 - Math.exp(-dt * 5))

    if (!flipTouched && (lastBeat < 6.1) !== (target < 6.1)) tracing.set(target >= 6.1)
    lastBeat = target
    const label = tracing.flipped ? 'Turn it back' : 'Turn the sheet over'
    if (flipButton.textContent !== label) flipButton.textContent = label

    bench.update(dt, clamp01((shown - 2.55) / 0.4), reduced, (v) => sound.play(v))
    cards.update(dt, near(2), reduced, () => sound.play(0.35))
    slides.update(dt, near(4), reduced, () => sound.play(0.25))
    pegs.update(dt, clamp01((shown - 4.45) / 0.5), reduced, () => sound.play(0.5))
    tracing.update(dt, reduced, () => sound.play())
    tally.update(dt, near(7), reduced, () => sound.play(0.4))
    play.update(dt)
    placeCamera(shown, reduced, now)
    composer.render()
  })
}
