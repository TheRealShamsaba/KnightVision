// The desk everything stands on: a walnut top, a baize mat under the stations, and one jointed
// desk lamp whose head swings to light whichever station the reader is looking at.
import * as THREE from 'three'
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js'
import { feltTexture, grainCanvas, toTexture } from './textures'

export const DESK = { w: 66, d: 62, y: 0 }
// Back-right corner: no camera stop looks through the lamp from here.
export const LAMP_BASE = new THREE.Vector3(27, 0, -25)

export class Desk {
  readonly group = new THREE.Group()
  readonly light: THREE.SpotLight
  private head = new THREE.Group()
  private arm = new THREE.Group()

  constructor() {
    const walnut = new THREE.MeshStandardMaterial({ color: '#6B4428', map: toTexture(grainCanvas(512, 33), 3), roughness: 0.5 })
    const top = new THREE.Mesh(new RoundedBoxGeometry(DESK.w, 1.2, DESK.d, 3, 0.25), walnut)
    top.position.y = -0.6
    top.receiveShadow = true

    // A baize mat, slightly inset, the way a desk blotter sits.
    const mat = new THREE.Mesh(new RoundedBoxGeometry(DESK.w - 8, 0.06, DESK.d - 10, 2, 0.02),
      new THREE.MeshStandardMaterial({ color: '#2E5A45', map: feltTexture(), roughness: 1 }))
    mat.position.set(1, 0.03, 3)
    mat.receiveShadow = true
    this.group.add(top, mat)

    // The lamp: a weighted base, two arms, a cone shade with a warm bulb inside.
    const metal = new THREE.MeshStandardMaterial({ color: '#2B2724', metalness: 0.6, roughness: 0.35 })
    const base = new THREE.Mesh(new THREE.CylinderGeometry(2.2, 2.6, 0.7, 40), metal)
    base.position.y = 0.35
    const lower = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 12, 12).translate(0, 6, 0), metal)
    lower.rotation.z = -0.32
    const elbow = new THREE.Group()
    elbow.position.set(Math.sin(0.32) * 12, Math.cos(0.32) * 12 + 0.7, 0)
    const upper = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 11, 12).translate(0, 5.5, 0), metal)
    this.arm.add(upper)
    this.arm.rotation.z = -1.2
    elbow.add(this.arm)
    const shade = new THREE.Mesh(new THREE.ConeGeometry(2.3, 3.2, 40, 1, true).translate(0, -1.2, 0),
      new THREE.MeshStandardMaterial({ color: '#C0392B', roughness: 0.55, side: THREE.DoubleSide }))
    const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.75, 20, 12), new THREE.MeshStandardMaterial({ color: '#fff4dc', emissive: '#ffdfa8', emissiveIntensity: 1.4 }))
    bulb.position.y = -1.6
    this.head.add(shade, bulb)
    this.head.position.y = 11
    this.arm.add(this.head)
    for (const m of [base, lower, upper, shade]) m.castShadow = true
    const lamp = new THREE.Group()
    lamp.position.copy(LAMP_BASE)
    // The arm reaches out over the middle of the desk.
    lamp.rotation.y = Math.atan2(LAMP_BASE.z, -LAMP_BASE.x)
    lamp.add(base, lower, elbow)
    this.group.add(lamp)

    this.light = new THREE.SpotLight('#FFE2BC', 190, 0, 0.55, 0.8, 1)
    this.light.castShadow = true
    this.light.shadow.mapSize.set(2048, 2048)
    this.light.shadow.bias = -0.0004
    this.light.shadow.normalBias = 0.03
    this.light.shadow.camera.near = 4
    this.light.shadow.camera.far = 90
    this.group.add(this.light, this.light.target)
  }

  /** Turn the shade towards `at`; the light comes from the bulb inside it. */
  point(at: THREE.Vector3) {
    this.group.updateMatrixWorld(true)
    const bulb = new THREE.Vector3(0, -1.6, 0).applyMatrix4(this.head.matrixWorld)
    this.light.position.copy(bulb)
    this.light.target.position.copy(at)
    // The shade's open end faces local -y: rotate that onto the direction of the target.
    const dir = at.clone().sub(bulb).normalize()
    const want = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, -1, 0), dir)
    const parent = new THREE.Quaternion()
    this.head.parent!.getWorldQuaternion(parent)
    this.head.quaternion.copy(parent.invert().multiply(want))
  }
}
