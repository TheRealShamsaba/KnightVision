// Stop 8: a second board at the front of the desk where you play the real network. Its candidate
// moves are drawn in red pencil on the board before it moves, so you can see what it considered.
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { Chess, type Square } from 'chess.js'
import type { Thought } from '../engine'
import { BOARD_TOP, inlay, type Wood } from './bench'
import { arrowSheet } from './pencil'
import { makePiece } from './pieces'
import { INK, RED, toTexture } from './textures'

type Engine = typeof import('../engine')
const FILES = 'abcdefgh'
/** Square centre on the board, white at the front (+z). */
function centre(s: string): THREE.Vector3 {
  return new THREE.Vector3(FILES.indexOf(s[0]) - 3.5, BOARD_TOP, 3.5 - (Number(s[1]) - 1))
}

type Hop = { obj: THREE.Object3D; from: THREE.Vector3; to: THREE.Vector3; t: number }

export class PlayBoard {
  readonly group = new THREE.Group()
  private pieces = new Map<string, THREE.Group>()
  private hops: Hop[] = []
  private sheet: THREE.MeshStandardMaterial
  private marks = new THREE.Group()
  private surface: THREE.Mesh
  private game = new Chess()
  private engine: Engine | null = null
  private thinking = false
  private selected: Square | null = null
  private generation = 0
  human: 'w' | 'b' = 'w'
  onChange: (state: { status: string; thoughts?: Thought[]; canUndo: boolean }) => void = () => {}
  onLand: () => void = () => {}

  constructor(private wood: Wood) {
    const frame = new THREE.Mesh(new RoundedBoxGeometry(8.8, 0.2, 8.8, 2, 0.04), wood.beech)
    frame.position.y = 0.1
    const cells = new THREE.Mesh(inlay(), new THREE.MeshStandardMaterial({ map: wood.maple.map, vertexColors: true, roughness: 0.55 }))
    for (const m of [frame, cells]) m.castShadow = m.receiveShadow = true
    this.sheet = new THREE.MeshStandardMaterial({ transparent: true, depthWrite: false, roughness: 0.9, opacity: 0 })
    const sheet = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), this.sheet)
    sheet.rotation.x = -Math.PI / 2
    sheet.position.y = BOARD_TOP + 0.003
    // An invisible plane on the squares, for finding which one was clicked.
    this.surface = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), new THREE.MeshBasicMaterial({ visible: false }))
    this.surface.rotation.x = -Math.PI / 2
    this.surface.position.y = BOARD_TOP
    this.group.add(frame, cells, sheet, this.surface, this.marks)
    this.place()
  }

  private status(): string {
    if (this.game.isCheckmate()) return this.game.turn() === this.human ? 'Checkmate. It got you.' : 'Checkmate. You beat it.'
    if (this.game.isDraw()) return 'Draw.'
    if (!this.engine) return 'Asleep. Wake it to play.'
    return this.thinking ? 'Looking at the board…' : 'Your move: click a piece, then a square.'
  }
  private emit(thoughts?: Thought[]) { this.onChange({ status: this.status(), thoughts, canUndo: !this.thinking && this.game.history().length > 0 }) }

  /** Put every piece where the game says, sliding the one that just moved. */
  private place(move?: { from: string; to: string }) {
    const board = this.game.board()
    const seen = new Set<string>()
    board.flat().forEach((p) => {
      if (!p) return
      seen.add(p.square)
      const key = p.color + p.type
      let obj = this.pieces.get(p.square)
      if (obj && obj.userData.key !== key) { this.group.remove(obj); obj = undefined }
      if (!obj) {
        obj = makePiece(p.type, p.color === 'w', this.wood)
        obj.userData.key = key
        this.group.add(obj)
        obj.position.copy(centre(p.square))
        this.pieces.set(p.square, obj)
      }
    })
    for (const [square, obj] of this.pieces) if (!seen.has(square)) { this.group.remove(obj); this.pieces.delete(square) }
    if (move) {
      const obj = this.pieces.get(move.to)
      if (obj) { this.hops.push({ obj, from: centre(move.from), to: centre(move.to), t: 0 }); obj.position.copy(centre(move.from)) }
    }
    this.drawMarks()
  }

  private drawMarks() {
    this.marks.clear()
    if (!this.selected) return
    const red = new THREE.MeshStandardMaterial({ color: RED, roughness: 0.7 })
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.03, 8, 40), red)
    ring.rotation.x = -Math.PI / 2
    ring.position.copy(centre(this.selected)).setY(BOARD_TOP + 0.02)
    this.marks.add(ring)
    const ink = new THREE.MeshStandardMaterial({ color: INK, roughness: 0.8 })
    for (const m of this.game.moves({ square: this.selected, verbose: true })) {
      const dot = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.02, 16), ink)
      dot.position.copy(centre(m.to)).setY(BOARD_TOP + 0.01)
      this.marks.add(dot)
    }
  }

  private pencil(thoughts: Thought[]) {
    const old = this.sheet.map
    this.sheet.map = toTexture(arrowSheet(thoughts.slice(0, 5).map((t) => ({ san: t.san, uci: t.from + t.to, p: t.p }))))
    this.sheet.map.wrapS = this.sheet.map.wrapT = THREE.ClampToEdgeWrapping
    this.sheet.needsUpdate = true
    this.sheet.opacity = 1
    old?.dispose()
  }

  async wake(progress: (f: number) => void) {
    const mod = await import('../engine')
    await mod.loadEngine(progress)
    this.engine = mod
    this.emit()
    void this.reply()
  }

  private async reply() {
    if (!this.engine || this.game.isGameOver() || this.game.turn() === this.human) return
    const mine = this.generation
    this.thinking = true
    this.emit()
    const thoughts = await this.engine.think(this.game)
    if (mine !== this.generation) return
    // Show its thinking in pencil first, then make the move.
    this.pencil(thoughts)
    this.emit(thoughts)
    await new Promise((r) => setTimeout(r, 900))
    if (mine !== this.generation) return
    const best = thoughts[0]
    this.game.move({ from: best.from, to: best.to, promotion: 'q' })
    this.thinking = false
    this.place(best)
    this.emit(thoughts)
  }

  /** A click on the board, as a ray. Returns true if it hit the board. */
  click(ray: THREE.Raycaster): boolean {
    const hit = ray.intersectObject(this.surface)[0]
    if (!hit) return false
    const local = this.group.worldToLocal(hit.point.clone())
    const f = Math.floor(local.x + 4), r = 7 - Math.floor(local.z + 4)
    if (f < 0 || f > 7 || r < 0 || r > 7) return true
    const square = `${FILES[f]}${r + 1}` as Square
    if (this.selected && this.selected !== square) {
      try {
        this.game.move({ from: this.selected, to: square, promotion: 'q' })
        const from = this.selected
        this.selected = null
        this.sheet.opacity = 0
        this.place({ from, to: square })
        this.emit()
        void this.reply()
        return true
      } catch { /* not a legal move: fall through and treat it as picking a piece */ }
    }
    const piece = this.game.get(square)
    const mine = this.engine && !this.thinking && !this.game.isGameOver() && this.game.turn() === this.human && piece?.color === this.human
    this.selected = mine && this.selected !== square ? square : null
    this.drawMarks()
    return true
  }

  newGame(colour: 'w' | 'b') {
    this.generation++
    this.game = new Chess()
    this.human = colour
    this.thinking = false
    this.selected = null
    this.sheet.opacity = 0
    // Playing Black, the board turns round so your pieces are nearest you.
    this.group.rotation.y = colour === 'b' ? Math.PI : 0
    this.place()
    this.emit()
    void this.reply()
  }

  undo() {
    if (this.thinking) return
    this.game.undo()
    if (this.game.turn() !== this.human) this.game.undo()
    this.selected = null
    this.sheet.opacity = 0
    this.place()
    this.emit()
  }

  update(dt: number) {
    this.hops = this.hops.filter((h) => {
      h.t = Math.min(1, h.t + dt / 0.55)
      const e = h.t < 0.5 ? 2 * h.t * h.t : 1 - Math.pow(-2 * h.t + 2, 2) / 2
      h.obj.position.lerpVectors(h.from, h.to, e).y += Math.sin(Math.PI * h.t) * 0.9
      if (h.t >= 1) this.onLand()
      return h.t < 1
    })
  }
}
