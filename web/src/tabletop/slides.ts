// Stop 4: the network's eight stages as eight paper slides standing in a wooden rack. Each slide shows
// that stage's six busiest channels for the Opera position, plus how far a corner square can see by
// then. A steel ball rolls along the rack's groove, through all eight once: one forward pass.
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { data } from '../data'
import type { Wood } from './bench'
import { INK, RED, toTexture } from './textures'

const SLIDE_W = 2.5, SLIDE_H = 3.4, GAP = 0.45
// How many squares out a corner square can see after each stage: a 3×3 look adds one, a block adds two.
const REACH = [1, 2, 4, 6, 8, 10, 12, 12]
const decode = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))

function slide(i: number): HTMLCanvasElement {
  const s = data.layers.stages[i]
  const c = document.createElement('canvas'); c.width = 500; c.height = 680
  const g = c.getContext('2d')!
  g.fillStyle = '#F6F1E6'; g.fillRect(0, 0, 500, 680)
  g.fillStyle = INK
  g.font = '500 30px Jost, sans-serif'; g.fillText(`stage ${i + 1}`, 30, 52)
  g.font = 'italic 26px Literata, serif'; g.fillText(s.name.toLowerCase(), 30, 90)
  // Six channels, each an 8×8 grid of graphite dots sized by how strongly it fires.
  const tile = 120, ox = 40, oy = 120
  s.maps.forEach((b64, k) => {
    const x = ox + (k % 3) * (tile + 22), y = oy + Math.floor(k / 3) * (tile + 22)
    g.strokeStyle = 'rgba(30,27,23,.45)'; g.lineWidth = 2; g.strokeRect(x, y, tile, tile)
    const cell = tile / 8
    decode(b64).forEach((v, j) => {
      const r = cell * 0.47 * Math.sqrt(v / 255)
      if (r < 0.6) return
      g.beginPath(); g.arc(x + (j % 8 + 0.5) * cell, y + (Math.floor(j / 8) + 0.5) * cell, r, 0, 7); g.fill()
    })
  })
  // Reach: the corner square in red, everything it can feel by now in graphite.
  const rx = 40, ry = 430, cell = 16
  for (let row = 0; row < 8; row++) for (let col = 0; col < 8; col++) {
    const d = Math.max(col, 7 - row)
    g.fillStyle = d === 0 ? RED : d <= REACH[i] ? 'rgba(30,27,23,.7)' : 'rgba(30,27,23,.1)'
    g.fillRect(rx + col * (cell + 2), ry + row * (cell + 2), cell, cell)
  }
  g.fillStyle = INK; g.font = '22px "Courier Prime", monospace'
  g.fillText('reach of a1', 200, 458)
  g.fillText(`${s.channels} channels`, 200, 498)
  g.fillText(`${(s.parameters / 1e6).toFixed(2)}m weights`, 200, 538)
  return c
}

export class Slides {
  readonly group = new THREE.Group()
  private ball: THREE.Mesh
  private clock = 0
  private lastSlide = -1
  readonly length = 8 * SLIDE_W + 7 * GAP

  constructor(wood: Wood) {
    const rack = new THREE.Mesh(new RoundedBoxGeometry(this.length + 1.4, 0.5, 1.6, 2, 0.06), wood.beech)
    rack.position.y = 0.25
    // The groove the ball runs in: a dark strip along the front of the rack.
    const groove = new THREE.Mesh(new THREE.BoxGeometry(this.length + 1, 0.04, 0.32), new THREE.MeshStandardMaterial({ color: '#3B2414', roughness: 0.8 }))
    groove.position.set(0, 0.5, 0.48)
    for (const m of [rack, groove]) m.castShadow = m.receiveShadow = true
    this.group.add(rack, groove)
    for (let i = 0; i < 8; i++) {
      const map = toTexture(slide(i))
      map.wrapS = map.wrapT = THREE.ClampToEdgeWrapping
      const panel = new THREE.Mesh(new RoundedBoxGeometry(SLIDE_W, SLIDE_H, 0.06, 2, 0.02),
        [wood.maple, wood.maple, wood.maple, wood.maple, new THREE.MeshStandardMaterial({ map, roughness: 0.9 }), wood.maple])
      panel.position.set(-this.length / 2 + SLIDE_W / 2 + i * (SLIDE_W + GAP), 0.5 + SLIDE_H / 2, -0.15)
      panel.rotation.x = -0.08
      panel.castShadow = panel.receiveShadow = true
      this.group.add(panel)
    }
    this.ball = new THREE.Mesh(new THREE.SphereGeometry(0.22, 24, 16), new THREE.MeshStandardMaterial({ color: '#C9CCD1', metalness: 0.9, roughness: 0.18 }))
    this.ball.castShadow = true
    this.group.add(this.ball)
  }

  /** The ball runs the length of the rack once every few seconds while the reader is here. */
  update(dt: number, active: boolean, reduced: boolean, tick: () => void) {
    this.clock = active ? this.clock + dt : 0
    const period = 5.5
    const t = reduced ? 1 : Math.min(1, (this.clock % period) / (period - 1.2))
    const x = -this.length / 2 + 0.2 + t * (this.length - 0.4)
    this.ball.position.set(x, 0.74, 0.48)
    this.ball.rotation.z = -x / 0.22
    const passing = Math.floor((x + this.length / 2) / (SLIDE_W + GAP))
    if (active && passing !== this.lastSlide && t < 1) { this.lastSlide = passing; tick() }
    if (t >= 1) this.lastSlide = -1
  }
}
