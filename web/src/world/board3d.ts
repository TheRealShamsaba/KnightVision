// A chessboard in the world: 64 squares (one instanced mesh, so clicks can find a square) and turned pieces.
import * as THREE from 'three'
import { makePiece } from './pieces'

export const FILES = 'abcdefgh'

export type Piece = { type: string; color: 'w' | 'b' }

export function parseFen(fen: string): Map<string, Piece> {
  const pieces = new Map<string, Piece>()
  fen.split(' ')[0].split('/').forEach((row, r) => {
    let file = 0
    for (const ch of row) {
      if (/\d/.test(ch)) { file += Number(ch); continue }
      pieces.set(`${FILES[file]}${8 - r}`, { type: ch.toLowerCase(), color: ch === ch.toLowerCase() ? 'b' : 'w' })
      file++
    }
  })
  return pieces
}

/** Square centre in the board's local space: a1 at (-3.5, -, 3.5), white nearest +z. */
export function squarePosition(square: string, y = 0, out = new THREE.Vector3()): THREE.Vector3 {
  return out.set(FILES.indexOf(square[0]) - 3.5, y, 7 - (Number(square[1]) - 1) - 3.5)
}

export const squareAt = (index: number): string => `${FILES[index % 8]}${Math.floor(index / 8) + 1}`

type Slide = { piece: THREE.Object3D; from: THREE.Vector3; to: THREE.Vector3; start: number }

export class Board3D {
  readonly group = new THREE.Group()
  readonly squares: THREE.InstancedMesh
  readonly overlay = new THREE.Group() // light trails, markers
  private pieceLayer = new THREE.Group()
  private pieces = new Map<string, THREE.Object3D>()
  private slides: Slide[] = []
  private materials: THREE.Material[] = []

  constructor({ light = '#262833', dark = '#0c0d12', glow = 0.08 } = {}) {
    const top = new THREE.BoxGeometry(0.98, 0.08, 0.98)
    const material = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.2, transparent: true })
    this.squares = new THREE.InstancedMesh(top, material, 64)
    const m = new THREE.Matrix4()
    for (let i = 0; i < 64; i++) {
      const p = squarePosition(squareAt(i), -0.04)
      m.makeTranslation(p.x, p.y, p.z)
      this.squares.setMatrixAt(i, m)
      const lightSquare = (i % 8 + Math.floor(i / 8)) % 2 === 1
      this.squares.setColorAt(i, new THREE.Color(lightSquare ? light : dark))
    }
    this.squares.receiveShadow = true
    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(8.2, 0.12, 8.2)),
      new THREE.LineBasicMaterial({ color: '#f2c14e', transparent: true, opacity: 0.35 + glow }),
    )
    frame.position.y = -0.06
    const slab = new THREE.Mesh(new THREE.BoxGeometry(8.4, 0.3, 8.4), new THREE.MeshStandardMaterial({ color: '#07080b', roughness: 0.4, transparent: true }))
    slab.position.y = -0.24
    this.materials.push(material, frame.material as THREE.Material, slab.material as THREE.Material)
    this.group.add(slab, this.squares, frame, this.pieceLayer, this.overlay)
  }

  setFen(fen: string, move?: { from: string; to: string }) {
    for (const piece of this.pieces.values()) this.pieceLayer.remove(piece)
    this.pieces.clear()
    this.slides = []
    parseFen(fen).forEach((p, square) => {
      const piece = makePiece(p.type, p.color)
      squarePosition(square, 0, piece.position)
      this.pieceLayer.add(piece)
      this.pieces.set(square, piece)
      if (move && square === move.to) {
        this.slides.push({ piece, from: squarePosition(move.from), to: piece.position.clone(), start: performance.now() })
        piece.position.copy(squarePosition(move.from))
      }
    })
  }

  /** Swaps every piece for another type (used to show what v1 fed the network). */
  setPieces(pieces: Map<string, Piece>) {
    for (const piece of this.pieces.values()) this.pieceLayer.remove(piece)
    this.pieces.clear()
    pieces.forEach((p, square) => {
      const piece = makePiece(p.type, p.color)
      squarePosition(square, 0, piece.position)
      this.pieceLayer.add(piece)
      this.pieces.set(square, piece)
    })
  }

  /** Fades the board itself; the pieces are left alone unless `pieces` is set. */
  setOpacity(opacity: number, { pieces = true } = {}) {
    for (const m of this.materials) m.opacity = opacity * (m instanceof THREE.LineBasicMaterial ? 0.5 : 1)
    this.squares.visible = opacity > 0.01
    if (pieces) {
      this.pieceLayer.visible = opacity > 0.05
      this.pieceLayer.scale.setScalar(Math.max(0.001, opacity))
    }
  }

  forEachPiece(fn: (square: string, piece: THREE.Object3D) => void) { this.pieces.forEach((p, sq) => fn(sq, p)) }

  pieceAt(square: string): THREE.Object3D | undefined { return this.pieces.get(square) }

  update(now: number) {
    this.slides = this.slides.filter((s) => {
      const t = Math.min(1, (now - s.start) / 420)
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2
      s.piece.position.lerpVectors(s.from, s.to, e)
      s.piece.position.y = Math.sin(Math.PI * e) * 0.6
      return t < 1
    })
  }
}
