// Beat 4: a sheet of tracing paper on a drafting board. Turned over top to bottom, so rank 1 swaps
// with rank 8 the way v1 read every move. Front: what the network learned. Back: what v1 showed it.
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import type { Wood } from './bench'
import { toTexture } from './textures'

const SHEET = 9.6, DESK = 0.25

export class Tracing {
  readonly group = new THREE.Group()
  private sheet = new THREE.Group()
  private angle = 0
  private speed = 0
  private target = 0
  private landed = true

  constructor(front: HTMLCanvasElement, back: HTMLCanvasElement, wood: Wood) {
    const desk = new THREE.Mesh(new RoundedBoxGeometry(10.4, DESK, 10.4, 2, 0.04), wood.maple)
    desk.position.y = DESK / 2
    desk.receiveShadow = true
    const paper = (canvas: HTMLCanvasElement, facing: number) => {
      const map = toTexture(canvas)
      map.wrapS = map.wrapT = THREE.ClampToEdgeWrapping
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(SHEET, SHEET),
        new THREE.MeshStandardMaterial({ map, transparent: true, opacity: 0.9, roughness: 0.95 }))
      // The back is laid face down with its top toward the viewer, so after a turn about x it reads upright.
      mesh.rotation.x = facing * Math.PI / 2
      mesh.castShadow = mesh.receiveShadow = true
      return mesh
    }
    this.sheet.add(paper(front, -1), paper(back, 1))
    this.sheet.position.y = DESK + 0.012
    this.group.add(desk, this.sheet)
  }

  get flipped() { return this.target === 1 }

  set(flipped: boolean) {
    if (this.flipped === flipped) return
    this.target = flipped ? 1 : 0
    this.landed = false
  }

  /** A slightly underdamped spring, so the sheet lands, overshoots a touch, and lies flat. */
  update(dt: number, reduced: boolean, land: () => void) {
    const goal = this.target * Math.PI
    if (reduced) { this.angle = goal; this.speed = 0 }
    for (let i = 0; i < 4; i++) {
      const h = Math.min(dt, 0.05) / 4
      this.speed += ((goal - this.angle) * 38 - this.speed * 8.5) * h
      this.angle += this.speed * h
    }
    if (!this.landed && Math.abs(goal - this.angle) < 0.06) { this.landed = true; land() }
    this.sheet.rotation.x = this.angle
    // Lift by the half-sheet's height so the turning edge never cuts into the desk.
    this.sheet.position.y = DESK + 0.012 + Math.abs(Math.sin(this.angle)) * SHEET / 2
  }
}
