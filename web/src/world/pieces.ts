// Turned chess pieces: lathe profiles for everything round, an extruded silhouette for the knight.
import * as THREE from 'three'

type Profile = [number, number][]
const BASE: Profile = [[0, 0], [0.36, 0], [0.36, 0.05], [0.31, 0.08], [0.31, 0.11], [0.24, 0.14]]
const PROFILES: Record<string, Profile> = {
  p: [...BASE, [0.13, 0.2], [0.1, 0.34], [0.17, 0.36], [0.1, 0.39], [0.13, 0.42], [0.155, 0.47], [0.14, 0.53], [0.09, 0.57], [0, 0.585]],
  r: [...BASE, [0.2, 0.18], [0.18, 0.48], [0.25, 0.5], [0.25, 0.63], [0.17, 0.63], [0.17, 0.59], [0, 0.59]],
  b: [...BASE, [0.14, 0.22], [0.1, 0.47], [0.18, 0.49], [0.1, 0.52], [0.15, 0.6], [0.13, 0.68], [0.07, 0.75], [0.04, 0.78], [0.06, 0.8], [0, 0.82]],
  q: [...BASE, [0.16, 0.22], [0.1, 0.57], [0.21, 0.61], [0.12, 0.65], [0.2, 0.8], [0.14, 0.82], [0.08, 0.86], [0.06, 0.9], [0, 0.91]],
  k: [...BASE, [0.17, 0.22], [0.11, 0.61], [0.22, 0.65], [0.13, 0.69], [0.19, 0.84], [0.1, 0.88], [0, 0.89]],
  n: [...BASE, [0.2, 0.18], [0, 0.18]],
}

const geometries = new Map<string, THREE.BufferGeometry[]>()

function knightHead(): THREE.BufferGeometry {
  const s = new THREE.Shape()
  const pts: [number, number][] = [[-0.2, 0.16], [0.22, 0.16], [0.19, 0.32], [0.1, 0.44], [0.16, 0.5], [0.27, 0.47], [0.33, 0.53],
    [0.3, 0.6], [0.16, 0.72], [0.06, 0.8], [0.0, 0.86], [-0.06, 0.8], [-0.14, 0.76], [-0.22, 0.62], [-0.25, 0.42]]
  s.moveTo(...pts[0])
  pts.slice(1).forEach((p) => s.lineTo(...p))
  s.closePath()
  const g = new THREE.ExtrudeGeometry(s, { depth: 0.16, bevelEnabled: true, bevelSize: 0.03, bevelThickness: 0.03, bevelSegments: 3, curveSegments: 4 })
  g.translate(0, 0, -0.08)
  return g
}

export function pieceGeometry(type: string): THREE.BufferGeometry[] {
  let g = geometries.get(type)
  if (!g) {
    g = [new THREE.LatheGeometry(PROFILES[type].map(([x, y]) => new THREE.Vector2(x, y)), 40)]
    if (type === 'n') g.push(knightHead())
    if (type === 'k') {
      g.push(new THREE.BoxGeometry(0.05, 0.2, 0.05).translate(0, 0.98, 0))
      g.push(new THREE.BoxGeometry(0.15, 0.05, 0.05).translate(0, 1.0, 0))
    }
    geometries.set(type, g)
  }
  return g
}

export const pieceMaterials = {
  w: new THREE.MeshPhysicalMaterial({ color: '#efe6d8', roughness: 0.32, clearcoat: 0.7, clearcoatRoughness: 0.2, emissive: '#3a2a1a', emissiveIntensity: 0.25 }),
  b: new THREE.MeshPhysicalMaterial({ color: '#15161c', roughness: 0.22, metalness: 0.35, clearcoat: 1, clearcoatRoughness: 0.1, sheen: 1, sheenColor: '#5b6cff' }),
}

export function makePiece(type: string, color: 'w' | 'b'): THREE.Group {
  const group = new THREE.Group()
  for (const geometry of pieceGeometry(type)) {
    const mesh = new THREE.Mesh(geometry, pieceMaterials[color])
    mesh.castShadow = true
    group.add(mesh)
  }
  // Knights face the enemy.
  if (type === 'n') group.rotation.y = color === 'w' ? Math.PI / 2 : -Math.PI / 2
  return group
}
