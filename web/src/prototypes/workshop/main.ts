// A · The Workshop. One tabletop, four stations along it; the camera keeps its three-quarter angle
// and slides from one to the next as the reader scrolls. Scroll position is mapped to a beat number
// (0..3), and each station turns that into its own small, hand-placed motions.
import * as THREE from 'three'
import './workshop.css'
import { data, quiltBytes, percent } from '../../data'
import { Bench, type Wood } from './bench'
import { Pegboard } from './pegboard'
import { Tracing } from './tracing'
import { tracingSides } from './pencil'
import { feltTexture, grainCanvas, toTexture } from './textures'
import { Clack } from './sound'

const opera = data.opera.plies[10]
const mirror = data.mirror[2]
// What v1 fed the network: trained order P N B R Q K, delivered in K Q R B N P.
const V1: Record<string, string> = { k: 'p', q: 'n', r: 'b', b: 'r', n: 'q', p: 'k' }

const STATION = { board: new THREE.Vector3(0, 0, 0), trays: new THREE.Vector3(17, 0, 0), pegs: new THREE.Vector3(34, 0, 0), paper: new THREE.Vector3(50, 0, 0) }
const reducedQuery = matchMedia('(prefers-reduced-motion: reduce)')

// ---------- Copy ----------

const elo = data.elo ? `about ${data.elo.rating} Elo (±${data.elo.margin}, from ${data.elo.games.length} games against Stockfish)` : 'a modest club level'
const app = document.querySelector<HTMLElement>('#app')!
app.innerHTML = `
<canvas class="table" aria-hidden="true"></canvas>
<button class="sound" type="button" aria-pressed="false">Sound off</button>
<main>
  <section class="beat">
    <div class="card">
      <h1>KnightVision</h1>
      <p class="lead">A chess network that never calculates. It looks once, and moves.</p>
      <p>The Opera Game, White to play move six. The red pencil is the network's opinion: the heavier the mark, the surer it is.</p>
      <ol class="candidates">
        ${opera.top.map(m => `<li><b>${m.san}</b><i style="--p:${m.p.toFixed(3)}"></i><span>${percent(m.p)}</span></li>`).join('')}
      </ol>
      <p class="note">${(data.parameters / 1e6).toFixed(1)} million weights, no search. It plays at ${elo}.</p>
    </div>
  </section>
  <section class="beat">
    <div class="card">
      <h2>What it sees</h2>
      <p>Before it looks, the board is taken apart into twelve trays, one for each kind of piece: white pawns, knights, bishops, rooks, queens and king, then the same again for black.</p>
      <p>Each tray says only where its pieces stand. That is all it gets. Nobody even tells it whose turn it is.</p>
    </div>
  </section>
  <section class="beat">
    <div class="card">
      <h2>What it says</h2>
      <p>Out come 4,096 numbers, one for every “from here, to there”. On this pegboard each big block is the square a piece leaves; the pegs inside it are where it could land. Rank 8 is at the back.</p>
      <p>Peg height is belief on a log scale: a flat stub is one in ten million or less, a full peg is certain.</p>
      <p>The red flag is <span class="pick">${opera.san}</span>, the move it played: ${percent(opera.share)} of what it gave to legal moves. The other ${percent(1 - opera.legalMass)} or so went on moves that aren't legal at all.</p>
    </div>
  </section>
  <section class="beat">
    <div class="card">
      <h2>The bug</h2>
      <p>For two years, every program that used it handed over the trays in the wrong order: K Q R B N P instead of P N B R Q K. Kings arrived as pawns, queens as knights, rooks as bishops, bishops as rooks, knights as queens, pawns as kings. Then every move it named was read upside down.</p>
      <p>After 1.e4 e5 it wants <span class="pick">2.${mirror.v2[0].san}</span> (${percent(mirror.v2[0].p)}). Shown the scrambled board, it played <span class="pick">2.${mirror.v1[0].san}</span> (${percent(mirror.v1[0].p)}).</p>
      <button class="flip" type="button">Turn the sheet over</button>
    </div>
  </section>
</main>
<footer><a href="./index.html">← all three directions</a><a href="/">Full site</a></footer>`

// ---------- Table, light, wood ----------

const canvas = app.querySelector<HTMLCanvasElement>('.table')!
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
renderer.shadowMap.enabled = true
renderer.shadowMap.type = THREE.PCFSoftShadowMap
renderer.toneMapping = THREE.ACESFilmicToneMapping
renderer.toneMappingExposure = 1.05

const scene = new THREE.Scene()
scene.background = new THREE.Color('#23463A')
scene.fog = new THREE.Fog('#23463A', 40, 95)
const camera = new THREE.PerspectiveCamera(30, 1, 0.5, 200)

// One warm lamp over whichever station is in front of the reader, plus a soft fill from the room.
const lamp = new THREE.SpotLight('#FFE2BC', 3.2, 0, 0.62, 0.7, 0)
lamp.castShadow = true
lamp.shadow.mapSize.set(2048, 2048)
lamp.shadow.bias = -0.0004
lamp.shadow.normalBias = 0.02
scene.add(lamp, lamp.target, new THREE.HemisphereLight('#FFF3E0', '#2E5A45', 0.85))

const felt = new THREE.Mesh(new THREE.PlaneGeometry(300, 300), new THREE.MeshStandardMaterial({ color: '#2E5A45', map: feltTexture(), roughness: 1 }))
felt.rotation.x = -Math.PI / 2
felt.receiveShadow = true
scene.add(felt)

const sound = new Clack()
const soundButton = app.querySelector<HTMLButtonElement>('.sound')!
soundButton.addEventListener('click', async () => {
  const on = await sound.toggle()
  soundButton.textContent = on ? 'Sound on' : 'Sound off'
  soundButton.setAttribute('aria-pressed', String(on))
  sound.play()
})

// Tray labels and the paper sketches use the web fonts, so draw them once the fonts are in (or give up after 2s).
const fonts = Promise.all(['500 40px Jost', 'italic 28px Literata'].map(f => document.fonts.load(f)))
Promise.race([fonts, new Promise(r => setTimeout(r, 2000))]).then(build, build)

function build() {
  const grain = grainCanvas(512, 7)
  const wood: Wood = {
    grain,
    maple: new THREE.MeshStandardMaterial({ color: '#E3CFA4', map: toTexture(grain), roughness: 0.58 }),
    beech: new THREE.MeshStandardMaterial({ color: '#B9824E', map: toTexture(grainCanvas(512, 21)), roughness: 0.42 }),
  }

  const bench = new Bench(wood, opera.fen, opera.top.slice(0, 5), STATION.trays)
  const pegs = new Pegboard(quiltBytes(opera), opera.uci, wood)
  pegs.group.position.copy(STATION.pegs)
  const [front, back] = tracingSides(
    { fen: mirror.fen, title: 'What it learned', move: mirror.v2[0] },
    { fen: mirror.fen, title: 'What v1 showed it', move: mirror.v1[0], swap: V1 },
  )
  const tracing = new Tracing(front, back, wood)
  tracing.group.position.copy(STATION.paper)
  scene.add(bench.group, pegs.group, tracing.group)

  let flipTouched = false
  const flipButton = app.querySelector<HTMLButtonElement>('.flip')!
  flipButton.addEventListener('click', () => {
    flipTouched = true
    tracing.set(!tracing.flipped)
  })

  // ---------- Camera poses, one per beat ----------

  type Pose = { at: THREE.Vector3; w: number; h: number; elev: number; turn: number } // w, h: half-extents to keep in frame
  let wide = true // matches the CSS breakpoint: cards beside the table rather than over it
  let rowOfSix = true
  const poses = (): Pose[] => [
    { at: new THREE.Vector3(0, 0.3, 0.2), w: 4.9, h: 4.1, elev: 48, turn: 20 },
    { at: STATION.trays.clone(), w: rowOfSix ? 10.6 : 5.4, h: rowOfSix ? 4.6 : 8.4, elev: 60, turn: 10 },
    { at: STATION.pegs.clone().setY(0.5), w: 6.1, h: 5.4, elev: 47, turn: 16 },
    { at: STATION.paper.clone().setY(0.3), w: 5.5, h: 5.1, elev: 66, turn: 6 },
  ]
  let POSES = poses()

  function resize() {
    const w = innerWidth, h = innerHeight
    wide = w >= 820
    rowOfSix = w / h > 0.9
    // Pixel ratio can change after load (a window dragged to another screen), so set it every time.
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    renderer.setSize(w, h, false)
    camera.aspect = w / h
    // Keep the subject clear of the cards: to the right of them on desktop, above them on phones.
    if (wide) camera.setViewOffset(w, h, -w * 0.17, 0, w, h)
    else camera.setViewOffset(w, h, 0, h * 0.2, w, h)
    camera.updateProjectionMatrix()
    bench.layout(rowOfSix ? 6 : 3)
    POSES = poses()
  }
  addEventListener('resize', resize)
  resize()

  const sections = [...app.querySelectorAll<HTMLElement>('.beat')]
  /** Scroll as a beat number: 0 with the first card centred, 3 with the last. */
  function beatFromScroll(): number {
    const mid = innerHeight / 2
    for (let i = 0; i < sections.length; i++) {
      const r = sections[i].getBoundingClientRect()
      if (mid < r.bottom || i === sections.length - 1) return Math.max(0, i + (mid - r.top) / r.height - 0.5)
    }
    return 0
  }

  const clamp01 = (x: number) => Math.min(1, Math.max(0, x))
  const smooth = (a: number, b: number, x: number) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t) }
  const pose = { at: new THREE.Vector3(), w: 0, h: 0, elev: 0, turn: 0 }

  function placeCamera(beat: number, reduced: boolean) {
    // Hold still while a card is being read; glide only through the middle of the gap between beats.
    // With reduced motion the camera cuts between stations instead of travelling.
    const g = Math.min(2, Math.floor(beat)), f = reduced ? Math.round(clamp01(beat - g)) : smooth(0.3, 0.7, beat - g)
    const a = POSES[g], b = POSES[g + 1]
    pose.at.lerpVectors(a.at, b.at, f)
    for (const k of ['w', 'h', 'elev', 'turn'] as const) pose[k] = a[k] + (b[k] - a[k]) * f
    const t = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))
    const usable = wide ? [1, 0.62] : [0.6, 1] // share of screen height, width the subject may use
    const distance = Math.max(pose.h / (t * usable[0]), pose.w / (t * camera.aspect * usable[1]))
    const e = THREE.MathUtils.degToRad(pose.elev), r = THREE.MathUtils.degToRad(pose.turn)
    camera.position.set(Math.sin(r) * Math.cos(e), Math.sin(e), Math.cos(r) * Math.cos(e)).multiplyScalar(distance).add(pose.at)
    camera.lookAt(pose.at)
    lamp.position.copy(pose.at).add(new THREE.Vector3(6, 19, 9))
    lamp.target.position.copy(pose.at)
  }

  // ---------- Loop ----------

  let shown = beatFromScroll()
  let lastBeat = shown
  let then = performance.now()
  if (!reducedQuery.matches && shown < 0.5) bench.setUp()
  if (shown >= 3.1) tracing.set(true)

  renderer.setAnimationLoop(now => {
    const dt = Math.min(0.05, (now - then) / 1000)
    then = now
    const reduced = reducedQuery.matches
    const target = beatFromScroll()
    shown = reduced ? target : shown + (target - shown) * (1 - Math.exp(-dt * 7))

    // Scroll turns the sheet as you pass the middle of the last card, unless the reader has taken over.
    if (!flipTouched && (lastBeat < 3.1) !== (target < 3.1)) tracing.set(target >= 3.1)
    lastBeat = target
    const label = tracing.flipped ? 'Turn it back' : 'Turn the sheet over'
    if (flipButton.textContent !== label) flipButton.textContent = label

    bench.update(dt, clamp01((shown - 0.1) / 0.75), reduced, v => sound.play(v))
    pegs.update(dt, clamp01((shown - 1.45) / 0.5), reduced, () => sound.play(0.5))
    tracing.update(dt, reduced, () => sound.play())
    placeCamera(shown, reduced)
    renderer.render(scene, camera)
  })
}
