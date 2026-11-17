// Stop 7: a cribbage-style tally board, one hole per game against Stockfish. A red peg for a win, a
// maple peg for a draw, a dark stub for a loss. The pegs go in one at a time, like keeping score.
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { data } from '../data'
import type { Wood } from './bench'
import { INK, RED, toTexture } from './textures'

const COLS = 20, PITCH = 0.5, ROW = 0.5, BLOCK_GAP = 1.5

export class Tally {
  readonly group = new THREE.Group()
  private pegs: THREE.InstancedMesh
  private order: { x: number; z: number; h: number }[] = []
  private shown = 0

  constructor(wood: Wood) {
    const games = data.elo?.games ?? []
    const levels = [...new Set(games.map((g) => g.elo))].sort()
    const per = levels.map((l) => games.filter((g) => g.elo === l).sort((a, b) => b.score - a.score || b.plies - a.plies))
    const rows = Math.ceil((per[0]?.length ?? 0) / COLS)
    const blockDepth = rows * ROW
    const width = COLS * PITCH + 1.6, depth = levels.length * blockDepth + (levels.length - 1) * BLOCK_GAP + 2.8

    // Top face: grain, the holes burned in, and the labels.
    const c = document.createElement('canvas'); c.width = 1200; c.height = Math.round(1200 * depth / width)
    const g = c.getContext('2d')!
    g.drawImage(wood.grain, 0, 0, c.width, c.height)
    const px = c.width / width
    const zOf = (block: number, row: number) => 1.6 + block * (blockDepth + BLOCK_GAP) + (row + 0.5) * ROW
    g.fillStyle = 'rgba(40, 20, 8, .85)'
    per.forEach((list, b) => list.forEach((_, i) => {
      const x = 0.8 + (i % COLS + 0.5) * PITCH, z = zOf(b, Math.floor(i / COLS))
      g.beginPath(); g.arc(x * px, z * px, 0.09 * px, 0, 7); g.fill()
    }))
    g.fillStyle = INK; g.font = `500 ${0.42 * px}px Jost, sans-serif`
    per.forEach((list, b) => {
      const won = list.filter((x) => x.score === 1).length, drawn = list.filter((x) => x.score === 0.5).length
      g.fillText(`vs stockfish ${levels[b]}   ${won} won · ${drawn} drawn · ${list.length - won - drawn} lost`, 0.8 * px, (zOf(b, 0) - 0.5) * px)
    })
    if (data.elo) {
      g.fillStyle = RED; g.font = `600 ${0.62 * px}px Jost, sans-serif`
      g.fillText(`${data.elo.rating} ± ${data.elo.margin} elo`, 0.8 * px, (depth - 0.4) * px)
    }
    const top = new THREE.MeshStandardMaterial({ map: toTexture(c), color: '#C99A64', roughness: 0.55 })
    top.map!.wrapS = top.map!.wrapT = THREE.ClampToEdgeWrapping
    const board = new THREE.Mesh(new RoundedBoxGeometry(width, 0.5, depth, 2, 0.06), [wood.beech, wood.beech, top, wood.beech, wood.beech, wood.beech])
    board.position.y = 0.25
    board.castShadow = board.receiveShadow = true
    this.group.add(board)

    // Pegs: one instanced cylinder, coloured per game.
    this.pegs = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.075, 0.085, 1, 10).translate(0, 0.5, 0),
      new THREE.MeshStandardMaterial({ roughness: 0.5 }), Math.max(1, games.length))
    const red = new THREE.Color(RED), maple = new THREE.Color('#E8D3A6'), dark = new THREE.Color('#4A2C17')
    per.forEach((list, b) => list.forEach((x, i) => {
      this.pegs.setColorAt(this.order.length, x.score === 1 ? red : x.score === 0.5 ? maple : dark)
      this.order.push({ x: -width / 2 + 0.8 + (i % COLS + 0.5) * PITCH, z: -depth / 2 + zOf(b, Math.floor(i / COLS)), h: x.score === 1 ? 0.9 : x.score === 0.5 ? 0.6 : 0.18 })
    }))
    this.pegs.castShadow = true
    this.group.add(this.pegs)
    this.draw()
  }

  private draw() {
    const m = new THREE.Matrix4()
    this.order.forEach((p, i) => {
      const h = i < this.shown ? p.h : 0.001
      this.pegs.setMatrixAt(i, m.makeScale(1, h, 1).setPosition(p.x, 0.5, p.z))
    })
    this.pegs.instanceMatrix.needsUpdate = true
  }

  /** Pegs go in one after another while the reader is here (about four seconds for all of them). */
  update(dt: number, active: boolean, reduced: boolean, peg: () => void) {
    const want = active ? (reduced ? this.order.length : Math.min(this.order.length, this.shown + Math.max(1, Math.round(dt * 40)))) : this.shown
    if (want !== this.shown) {
      if (Math.floor(want / 8) !== Math.floor(this.shown / 8)) peg()
      this.shown = want
      this.draw()
    }
  }
}
