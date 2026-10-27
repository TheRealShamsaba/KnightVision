// Beats 1 and 2: the board with red-pencil candidates, and the twelve trays its pieces are sorted into.
// Pieces move by "hops": time-based arcs that end with a little rock as the hand lets go, so the
// motion feels placed whatever speed the reader scrolls at.
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { Candidate } from '../../data'
import { makePiece } from './pieces'
import { arrowSheet, parseFen } from './pencil'
import { toTexture, trayTexture } from './textures'

export type Wood = { maple: THREE.MeshStandardMaterial; beech: THREE.MeshStandardMaterial; grain: HTMLCanvasElement }

export const BOARD_TOP = 0.28
// Trays are big enough that a sorted piece is still readable from the camera.
const TRAY_TOP = 0.1, CELL = 0.36, TRAY_W = 3.15, TRAY_D = 3.75, TRAY_Z = -0.32
// Sorted pieces stay larger than a tray cell: the trays are sparse, and a piece the size of its cell vanishes.
const TRAY_PIECE = CELL * 1.7
const ORDER = 'PNBRQKpnbrqk'
const NAMES = ['pawns', 'knights', 'bishops', 'rooks', 'queens', 'king']
const HOP = 0.85 // seconds per hop; the last 30% is the settle

type Hop = { from: THREE.Vector3; to: THREE.Vector3; fromScale: number; toScale: number; t: number }
type Piece = { obj: THREE.Group; home: THREE.Vector3; square: string; tray: number; inTray: boolean; hop: Hop | null }

const easeInOut = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2)
const squareOffset = (square: string) => ['abcdefgh'.indexOf(square[0]) - 3.5, 4.5 - Number(square[1])]

/** 64 squares merged into one mesh, each cut from a different part of the plank, grain alternating. */
function inlay(): THREE.BufferGeometry {
  const maple = new THREE.Color('#E3CFA4'), beech = new THREE.Color('#B9824E')
  const parts = Array.from({ length: 64 }, (_, i) => {
    const g = new RoundedBoxGeometry(0.98, 0.1, 0.98, 2, 0.015)
    g.translate((i % 8) - 3.5, BOARD_TOP - 0.05, 3.5 - Math.floor(i / 8))
    const light = (i % 8 + Math.floor(i / 8)) % 2 === 1
    const uv = g.getAttribute('uv') as THREE.BufferAttribute
    const du = Math.random(), dv = Math.random()
    for (let k = 0; k < uv.count; k++) {
      const u = uv.getX(k) * 0.5, v = uv.getY(k) * 0.5
      uv.setXY(k, (light ? v : u) + du, (light ? u : v) + dv)
    }
    const c = light ? maple : beech, colors = new Float32Array(uv.count * 3)
    for (let k = 0; k < uv.count; k++) colors.set([c.r, c.g, c.b], k * 3)
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    return g
  })
  return mergeGeometries(parts)
}

export class Bench {
  readonly group = new THREE.Group()
  readonly trayCentres: THREE.Vector3[] = []
  private trays: THREE.Mesh[] = []
  private pieces: Piece[] = []
  private arrows: THREE.MeshStandardMaterial

  constructor(wood: Wood, fen: string, candidates: Candidate[], private trayOrigin: THREE.Vector3) {
    // The board: a beech frame with 64 inlaid squares, maple and beech.
    const frame = new THREE.Mesh(new RoundedBoxGeometry(8.8, 0.2, 8.8, 2, 0.04), wood.beech)
    frame.position.y = 0.1
    const cells = new THREE.Mesh(inlay(), new THREE.MeshStandardMaterial({ map: wood.maple.map, vertexColors: true, roughness: 0.55 }))
    for (const mesh of [frame, cells]) mesh.castShadow = mesh.receiveShadow = true

    // Red pencil lies flat on the squares, under the pieces.
    this.arrows = new THREE.MeshStandardMaterial({ map: toTexture(arrowSheet(candidates)), transparent: true, depthWrite: false, roughness: 0.9 })
    const sheet = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), this.arrows)
    sheet.rotation.x = -Math.PI / 2
    sheet.position.y = BOARD_TOP + 0.003
    sheet.receiveShadow = true
    this.group.add(frame, cells, sheet)

    const trayGeometry = new RoundedBoxGeometry(TRAY_W, TRAY_TOP, TRAY_D, 2, 0.02)
    for (let i = 0; i < 12; i++) {
      const label = `${i < 6 ? 'White' : 'Black'} ${NAMES[i % 6]}`
      // White's trays are dark beech so the maple pieces show; black's are maple, for the beech pieces.
      const map = trayTexture(wood.grain, label, i < 6 ? '#9C6A3E' : '#E3CFA4', { w: TRAY_W, d: TRAY_D, cell: CELL, zOffset: TRAY_Z })
      const material = new THREE.MeshStandardMaterial({ map, roughness: 0.62 })
      const tray = new THREE.Mesh(trayGeometry, material)
      tray.castShadow = tray.receiveShadow = true
      this.trays.push(tray)
      this.trayCentres.push(new THREE.Vector3())
      this.group.add(tray)
    }
    this.layout(6)

    parseFen(fen).forEach(p => {
      const obj = makePiece(p.type, p.white, wood)
      const [x, z] = squareOffset(p.square)
      const home = new THREE.Vector3(x, BOARD_TOP, z)
      const piece: Piece = { obj, home, square: p.square, tray: ORDER.indexOf(p.white ? p.type.toUpperCase() : p.type), inTray: false, hop: null }
      this.pieces.push(piece)
      this.group.add(obj)
      obj.position.copy(home)
    })
  }

  /** Six trays to a row on wide screens, three on narrow ones; white's trays before black's. */
  layout(columns: number) {
    const rows = 12 / columns
    this.trays.forEach((tray, i) => {
      const col = i % columns, row = Math.floor(i / columns)
      this.trayCentres[i].set((col - (columns - 1) / 2) * (TRAY_W + 0.3), 0, (row - (rows - 1) / 2) * (TRAY_D + 0.3)).add(this.trayOrigin)
      tray.position.copy(this.trayCentres[i]).setY(TRAY_TOP / 2)
    })
    for (const p of this.pieces) if (p.inTray && !p.hop) p.obj.position.copy(this.traySpot(p))
  }

  /** Pieces drop onto the board one after another, as if set up by hand. */
  setUp() {
    this.pieces.forEach((p, i) => {
      p.hop = { from: p.home.clone().setY(BOARD_TOP + 1.4), to: p.home, fromScale: 1, toScale: 1, t: -0.4 - i * 0.07 }
      p.obj.position.copy(p.hop.from)
    })
  }

  private traySpot(p: Piece): THREE.Vector3 {
    const [x, z] = squareOffset(p.square)
    return this.trayCentres[p.tray].clone().add(new THREE.Vector3(x * CELL, TRAY_TOP, TRAY_Z + z * CELL))
  }

  /** progress 0..1: how far through the sorting the reader has scrolled; each piece has its turn. */
  update(dt: number, progress: number, reduced: boolean, land: (volume: number) => void) {
    this.arrows.opacity = Math.max(0, 1 - progress * 5)
    let launched = 0
    this.pieces.forEach((p, i) => {
      const want = progress > 0.08 + 0.8 * (i / this.pieces.length)
      if (want !== p.inTray) {
        p.inTray = want
        p.hop = {
          from: p.obj.position.clone(), to: want ? this.traySpot(p) : p.home,
          fromScale: p.obj.scale.x, toScale: want ? TRAY_PIECE : 1, t: -0.06 * launched++,
        }
      }
      const hop = p.hop
      if (!hop) return
      const before = hop.t
      hop.t = reduced ? 1 : hop.t + dt / HOP
      const travel = Math.min(1, Math.max(0, hop.t / 0.7)), e = easeInOut(travel)
      const lift = Math.min(2.6, hop.from.distanceTo(hop.to) * 0.22)
      p.obj.position.lerpVectors(hop.from, hop.to, e).y += lift * Math.sin(Math.PI * travel)
      p.obj.scale.setScalar(hop.fromScale + (hop.toScale - hop.fromScale) * e)
      // Carried at a slight tilt, then a damped rock once it touches down.
      const settle = Math.max(0, (hop.t - 0.7) / 0.3)
      p.obj.rotation.z = hop.t >= 1 ? 0 : travel < 1 ? 0.14 * travel : 0.14 * Math.exp(-4 * settle) * Math.cos(9 * settle)
      if (before < 0.7 && hop.t >= 0.7) land(p.inTray ? 0.6 : 1)
      if (hop.t >= 1) p.hop = null
    })
  }
}
