// Beat 3: the 4,096 outputs as pegs. A board of boards: each of the 64 blocks is the square a piece
// moves from (a1 front left), and the 8×8 pegs inside it are where it moves to (rank 8 at the back).
// Peg height follows the quilt byte (log belief) on a steep curve, so the few moves it really believes
// in stand well above the rest; those also get a red-pencil cap, darker the surer it is.
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { baseTexture, RED, INK } from './textures'
import type { Wood } from './bench'

const PEG = 0.14, BLOCK = 1.34, BASE = 11.6, TOP = 0.26, TALL = 2.8
const CAP_FROM = 0.002 // belief at which a peg gets a red cap
const easeOutBack = (x: number) => 1 + 2.4 * Math.pow(x - 1, 3) + 1.4 * Math.pow(x - 1, 2)

const sq = (i: number) => (i % 8) - 3.5
const rk = (i: number) => 3.5 - Math.floor(i / 8)
const index = (s: string) => 'abcdefgh'.indexOf(s[0]) + (Number(s[1]) - 1) * 8

export class Pegboard {
  readonly group = new THREE.Group()
  private pegs: THREE.InstancedMesh
  private caps: THREE.InstancedMesh
  private capped: number[] = []
  private heights = new Float32Array(4096)
  private level = new Float32Array(64) // how far each block's pegs have risen, 0..1 (overshoots briefly)
  private clock = new Float32Array(64).fill(-1) // time into the current rise, -1 when lowered
  private flag = new THREE.Group()
  private chosenFrom: number
  private chosen: number

  constructor(bytes: Uint8Array, uci: string, wood: Wood) {
    const top = new THREE.MeshStandardMaterial({ map: baseTexture(wood.grain, BLOCK, BASE), color: '#B9824E', roughness: 0.6 })
    const base = new THREE.Mesh(new RoundedBoxGeometry(BASE, 0.2, BASE, 2, 0.04), [wood.beech, wood.beech, top, wood.beech, wood.beech, wood.beech])
    base.position.y = 0.1

    const blocks = new THREE.InstancedMesh(new RoundedBoxGeometry(BLOCK - 0.08, 0.06, BLOCK - 0.08, 2, 0.015), wood.maple.clone(), 64)
    ;(blocks.material as THREE.MeshStandardMaterial).color.set('#ffffff')
    const m = new THREE.Matrix4(), light = new THREE.Color('#E3CFA4'), dark = new THREE.Color('#C99A64')
    for (let f = 0; f < 64; f++) {
      blocks.setMatrixAt(f, m.makeTranslation(sq(f) * BLOCK, TOP - 0.03, rk(f) * BLOCK))
      blocks.setColorAt(f, (f % 8 + Math.floor(f / 8)) % 2 ? light : dark)
    }

    const peg = new THREE.CylinderGeometry(0.05, 0.054, 1, 10).translate(0, 0.5, 0)
    this.pegs = new THREE.InstancedMesh(peg, new THREE.MeshStandardMaterial({ color: '#EBDBB8', map: wood.maple.map, roughness: 0.55 }), 4096)
    for (let i = 0; i < 4096; i++) this.heights[i] = 0.03 + Math.pow(bytes[i] / 255, 2.4) * TALL

    // Red caps on the moves it gives at least CAP_FROM, shading from pale to full red-pencil.
    const pale = new THREE.Color('#E58C79'), full = new THREE.Color(RED)
    for (let i = 0; i < 4096; i++) if (10 ** (-7 + (7 * bytes[i]) / 255) >= CAP_FROM) this.capped.push(i)
    this.caps = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.07, 0.07, 0.12, 12), new THREE.MeshStandardMaterial({ roughness: 0.6 }), Math.max(1, this.capped.length))
    this.capped.forEach((i, k) => {
      const p = 10 ** (-7 + (7 * bytes[i]) / 255)
      const t = Math.min(1, Math.log10(p / CAP_FROM) / Math.log10(0.5 / CAP_FROM))
      this.caps.setColorAt(k, pale.clone().lerp(full, t))
    })
    for (const mesh of [base, blocks, this.pegs, this.caps]) mesh.castShadow = mesh.receiveShadow = true

    // The move it played: a red-pencil ring round the peg and a paper flag on a pin above it.
    this.chosenFrom = index(uci.slice(0, 2))
    this.chosen = this.chosenFrom * 64 + index(uci.slice(2, 4))
    const red = new THREE.MeshStandardMaterial({ color: RED, roughness: 0.7, side: THREE.DoubleSide })
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.085, 0.014, 8, 32), red)
    ring.rotation.x = -Math.PI / 2
    ring.position.copy(this.pegAt(this.chosen)).setY(TOP + 0.01)
    const pin = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.6, 6).translate(0, 0.3, 0), new THREE.MeshStandardMaterial({ color: INK }))
    const cloth = new THREE.Shape([new THREE.Vector2(0, 0.6), new THREE.Vector2(0.3, 0.52), new THREE.Vector2(0, 0.42)])
    const banner = new THREE.Mesh(new THREE.ShapeGeometry(cloth), red)
    banner.position.z = 0.001
    pin.castShadow = banner.castShadow = true
    this.flag.add(pin, banner)
    this.flag.position.copy(this.pegAt(this.chosen))
    this.group.add(base, blocks, this.pegs, this.caps, ring, this.flag)
    this.draw()
  }

  private pegAt(i: number): THREE.Vector3 {
    const from = Math.floor(i / 64), to = i % 64
    return new THREE.Vector3(sq(from) * BLOCK + sq(to) * PEG, TOP, rk(from) * BLOCK + rk(to) * PEG)
  }

  private draw() {
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3()
    for (let i = 0; i < 4096; i++) {
      s.set(1, Math.max(0.004, this.heights[i] * this.level[i >> 6]), 1)
      this.pegs.setMatrixAt(i, m.compose(this.pegAt(i), q, s))
    }
    this.pegs.instanceMatrix.needsUpdate = true
    this.capped.forEach((i, k) => {
      const lvl = this.level[i >> 6]
      const at = this.pegAt(i)
      at.y = TOP + this.heights[i] * lvl + 0.05
      s.setScalar(lvl > 0.02 ? 1 : 0.001)
      this.caps.setMatrixAt(k, m.compose(at, q, s))
    })
    this.caps.instanceMatrix.needsUpdate = true
    this.flag.position.y = TOP + this.heights[this.chosen] * this.level[this.chosenFrom] + 0.02
  }

  /** progress 0..1 raises the blocks in turn, from a1 to h8; each pops up with a small overshoot. */
  update(dt: number, progress: number, reduced: boolean, rise: () => void) {
    let changed = false
    for (let f = 0; f < 64; f++) {
      const up = progress > 0.04 + 0.9 * (f / 64)
      if (up && this.clock[f] < 0) { this.clock[f] = 0; rise() }
      if (!up) this.clock[f] = -1
      const target = up ? (reduced ? 1 : easeOutBack(Math.min(1, (this.clock[f] += dt) / 0.55))) : Math.max(0, this.level[f] - dt * 4)
      if (target !== this.level[f]) { this.level[f] = target; changed = true }
    }
    if (changed) this.draw()
  }
}
