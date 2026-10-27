// Hartwig's 1923 Bauhaus set: each piece's shape is how it moves. Pawn a small cube, rook a cube
// (straight lines), bishop a saltire (diagonals), knight an L, queen a ball on a cube, king a cube
// crowned by a smaller one turned 45°. Origins sit on the square's surface.
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'

const box = (w: number, h: number, d: number) => new RoundedBoxGeometry(w, h, d, 2, Math.min(w, h, d) * 0.08)

const G = {
  pawn: box(0.34, 0.34, 0.34),
  rook: box(0.56, 0.56, 0.56),
  bishopFoot: box(0.3, 0.12, 0.3),
  bishopBar: box(0.13, 0.7, 0.3),
  knightFoot: box(0.54, 0.2, 0.3),
  knightPost: box(0.2, 0.56, 0.3),
  queenFoot: box(0.44, 0.44, 0.44),
  ball: new THREE.SphereGeometry(0.23, 32, 16),
  kingFoot: box(0.5, 0.5, 0.5),
  crown: box(0.3, 0.3, 0.3),
}

type Part = [THREE.BufferGeometry, number, number, number, number?, number?] // geometry, x, y, z, rotY, rotZ

const PARTS: Record<string, Part[]> = {
  p: [[G.pawn, 0, 0.17, 0]],
  r: [[G.rook, 0, 0.28, 0]],
  b: [[G.bishopFoot, 0, 0.06, 0], [G.bishopBar, 0, 0.4, 0, 0, Math.PI / 4], [G.bishopBar, 0, 0.4, 0, 0, -Math.PI / 4]],
  n: [[G.knightFoot, 0, 0.1, 0], [G.knightPost, -0.17, 0.48, 0]],
  q: [[G.queenFoot, 0, 0.22, 0], [G.ball, 0, 0.66, 0]],
  k: [[G.kingFoot, 0, 0.25, 0], [G.crown, 0, 0.65, 0, Math.PI / 4]],
}

export function makePiece(type: string, white: boolean, materials: { maple: THREE.Material; beech: THREE.Material }): THREE.Group {
  const group = new THREE.Group()
  for (const [geometry, x, y, z, ry = 0, rz = 0] of PARTS[type]) {
    const mesh = new THREE.Mesh(geometry, white ? materials.maple : materials.beech)
    mesh.position.set(x, y, z)
    mesh.rotation.set(0, ry, rz)
    mesh.castShadow = mesh.receiveShadow = true
    group.add(mesh)
  }
  // Knights face the opponent, so the L reads the other way round for black.
  if (type === 'n' && !white) group.rotation.y = Math.PI
  return group
}
