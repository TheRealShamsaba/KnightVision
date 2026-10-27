// Stop 2: a card-file box of training records. Each card is one line of the training file: a
// position, the move a person played there, and how the game ended. One card at a time rises.
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { data } from '../data'
import type { Wood } from './bench'
import { INK, PAPER, RED, toTexture } from './textures'

const W = 5.2, H = 2.6, D = 7.4, CARD_W = 4.6, CARD_H = 3.3

function label(): HTMLCanvasElement {
  const c = document.createElement('canvas'); c.width = 1040; c.height = 520
  const g = c.getContext('2d')!
  g.fillStyle = PAPER; g.fillRect(250, 120, 540, 280)
  g.strokeStyle = INK; g.lineWidth = 4; g.strokeRect(262, 132, 516, 256)
  g.fillStyle = INK; g.textAlign = 'center'
  g.font = '600 92px Jost, sans-serif'; g.fillText('84,000', 520, 255)
  g.font = '500 40px Jost, sans-serif'; g.fillText('human games', 520, 320)
  g.font = 'italic 30px Literata, serif'; g.fillText('≈ 5,000,000 positions', 520, 368)
  return c
}

function record(i: number): HTMLCanvasElement {
  const ply = data.opera.plies[i % data.opera.plies.length]
  const c = document.createElement('canvas'); c.width = 920; c.height = 660
  const g = c.getContext('2d')!
  g.fillStyle = '#FCFAF4'; g.fillRect(0, 0, 920, 660)
  // Index-card ruling: one red line at the top, blue lines below.
  g.fillStyle = RED; g.fillRect(0, 110, 920, 4)
  g.fillStyle = 'rgba(61, 85, 136, .35)'
  for (let y = 176; y < 660; y += 62) g.fillRect(0, y, 920, 2)
  g.fillStyle = INK
  g.font = '500 40px Jost, sans-serif'; g.fillText(`record ${String(i * 7919 % 4999999 + 1).padStart(7, '0')}`, 40, 78)
  g.font = '30px "Courier Prime", Courier, monospace'
  const fen = ply.fen.split(' ').slice(0, 2).join(' ')
  const lines = ['{"fen": "' + fen.slice(0, 34), fen.slice(34) + '",', `  "move": "${ply.san}",`, `  "result": "${data.opera.result}"}`]
  lines.forEach((l, k) => g.fillText(l, 40, 166 + k * 62))
  g.fillStyle = RED; g.font = 'italic 34px Literata, serif'
  g.fillText(`guess the move → ${ply.san}`, 40, 600)
  return c
}

export class CardBox {
  readonly group = new THREE.Group()
  private lifted: THREE.Mesh
  private textures: THREE.CanvasTexture[] = []
  private clock = 0
  private shown = -1

  constructor(wood: Wood) {
    const wall = (w: number, h: number, d: number, x: number, z: number, mat: THREE.Material | THREE.Material[] = wood.beech) => {
      const m = new THREE.Mesh(new RoundedBoxGeometry(w, h, d, 2, 0.03), mat)
      m.position.set(x, h / 2, z)
      m.castShadow = m.receiveShadow = true
      this.group.add(m)
    }
    const front = new THREE.MeshStandardMaterial({ map: toTexture(label()), color: '#B9824E', roughness: 0.5 })
    front.map!.wrapS = front.map!.wrapT = THREE.ClampToEdgeWrapping
    wall(W, 0.2, D, 0, 0)
    wall(W, H, 0.2, 0, D / 2, [wood.beech, wood.beech, wood.beech, wood.beech, front, wood.beech])
    wall(W, H * 1.15, 0.2, 0, -D / 2)
    wall(0.2, H, D, W / 2, 0)
    wall(0.2, H, D, -W / 2, 0)

    // The pack of cards, leaning a little this way and that.
    const card = new THREE.BoxGeometry(CARD_W, CARD_H, 0.025)
    const paper = new THREE.MeshStandardMaterial({ color: '#F4EFE4', roughness: 0.9 })
    for (let i = 0; i < 46; i++) {
      const m = new THREE.Mesh(card, paper)
      m.position.set(0, CARD_H / 2 + 0.1, -D / 2 + 0.5 + i * 0.13)
      m.rotation.x = -0.12 + Math.sin(i * 2.3) * 0.05
      m.castShadow = true
      this.group.add(m)
    }
    for (let i = 0; i < 6; i++) this.textures.push(toTexture(record(i * 5 + 3)))
    this.lifted = new THREE.Mesh(new THREE.PlaneGeometry(CARD_W, CARD_H), new THREE.MeshStandardMaterial({ map: this.textures[0], roughness: 0.9, side: THREE.DoubleSide }))
    this.lifted.castShadow = true
    this.group.add(this.lifted)
  }

  /** While the reader is here, a card rises out of the box every few seconds, shows its record, and drops back. */
  update(dt: number, active: boolean, reduced: boolean, pick: () => void) {
    this.clock = active ? this.clock + dt : 0
    const cycle = 3.2, t = (this.clock % cycle) / cycle
    const n = Math.floor(this.clock / cycle) % this.textures.length
    if (active && n !== this.shown) { this.shown = n; (this.lifted.material as THREE.MeshStandardMaterial).map = this.textures[n]; pick() }
    // Up over the first fifth, held, then down over the last fifth.
    const up = reduced ? (active ? 1 : 0) : !active ? 0 : t < 0.2 ? t / 0.2 : t > 0.82 ? (1 - t) / 0.18 : 1
    const e = up * up * (3 - 2 * up)
    this.lifted.position.set(0, CARD_H / 2 + 0.1 + e * (CARD_H * 0.95), D / 2 - 1.2)
    this.lifted.rotation.x = -0.12 - e * 0.35
    this.lifted.visible = e > 0.01
  }
}
