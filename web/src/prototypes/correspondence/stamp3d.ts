// The one 3D moment: a rubber stamp, carved readable (so it prints mirrored), pressed onto an
// orange board. The scene renders "ink coverage" (R = orange, G = purple, B = blue; dark = ink)
// into an offscreen target; a full-screen pass turns each channel into its own rotated halftone
// screen, out of register, multiplied together. The canvas then multiplies onto the card's paper.
import * as THREE from 'three'
import { parseFen, FILES } from '../../world/board3d'
import { drawPiece } from './ink'

const TW = 1280, TH = 1024, PW = 9, PH = 7.2 // face texture px ↔ rubber size in world units
const BX = 316, BY = 48, BS = 928, SQ = BS / 8
const SEP = { orange: '#00ffff', purple: '#ff00ff', blue: '#ffff00' } // one ink per channel

function sheet(fill: string, draw: (g: CanvasRenderingContext2D) => void): THREE.CanvasTexture {
  const c = document.createElement('canvas'); c.width = TW; c.height = TH
  const g = c.getContext('2d')!
  g.fillStyle = fill; g.fillRect(0, 0, TW, TH); g.globalCompositeOperation = 'multiply'
  draw(g)
  const tex = new THREE.CanvasTexture(c); tex.anisotropy = 4
  return tex
}

// What the carver cut: the plane order P N B R Q K and the position, both readable.
function carve(g: CanvasRenderingContext2D, fen: string): void {
  g.fillStyle = g.strokeStyle = SEP.purple
  g.lineWidth = 10; g.strokeRect(14, 14, TW - 28, TH - 28)
  g.font = '800 84px "Bricolage Grotesque"'
  'pnbrqk'.split('').forEach((t, i) => {
    const y = BY + i * (BS / 6)
    drawPiece(g, t, 42, y + 22, 112)
    g.fillText(t.toUpperCase(), 176, y + 120)
  })
  for (const [s, p] of parseFen(fen))
    drawPiece(g, p.type, BX + FILES.indexOf(s[0]) * SQ + SQ * 0.08, BY + (8 - Number(s[1])) * SQ + SQ * 0.07, SQ * 0.84, p.color === 'w')
}

const VERT = /* glsl */ `
varying vec3 vN; varying vec2 vUv;
void main() {
  vUv = uv; vN = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
}`
// Flat ink, or ink from a separation map (+ a second map for the print), shaded by a
// three-step posterised Lambert whose shadow is printed in the blue drum.
const INK_FRAG = /* glsl */ `
uniform vec3 ink; uniform float shade; uniform float useMap; uniform float amount; uniform vec3 light;
uniform sampler2D tMap; uniform sampler2D tMap2;
varying vec3 vN; varying vec2 vUv;
void main() {
  vec3 cov = ink;
  if (useMap > 0.5) {
    cov = 1.0 - texture2D(tMap, vUv).rgb;
    vec3 c2 = (1.0 - texture2D(tMap2, vUv).rgb) * amount;
    cov = 1.0 - (1.0 - cov) * (1.0 - c2);
  }
  float l = floor(max(dot(normalize(vN), light), 0.0) * 3.0 + 0.5) / 3.0;
  cov.b = 1.0 - (1.0 - cov.b) * (1.0 - (1.0 - l) * shade);
  gl_FragColor = vec4(1.0 - cov, 1.0);
}`
const SHADOW_FRAG = /* glsl */ `
uniform float amount; uniform float soft; varying vec2 vUv;
void main() {
  float d = length(max(abs(vUv - 0.5) * 2.0 - vec2(0.72, 0.68), 0.0));
  float a = (1.0 - smoothstep(0.0, 0.12 + soft, d)) * amount;
  gl_FragColor = vec4(1.0, 1.0, 1.0 - a, 1.0);
}`
// Threshold screen: f is 0 at each dot centre, 1 at the cell corners; ink wherever coverage > f.
// Tints become dots, solids stay crisp-edged.
const HALFTONE_FRAG = /* glsl */ `
uniform sampler2D tSrc; uniform vec2 res; uniform float cell;
uniform vec2 o1; uniform vec2 o2; uniform vec2 o3;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
vec3 cover(vec2 px) { return 1.0 - texture2D(tSrc, px / res).rgb; }
float screen(vec2 px, float ang, float c) {
  vec2 p = mat2(cos(ang), sin(ang), -sin(ang), cos(ang)) * px / cell;
  float f = 0.5 - 0.25 * (cos(6.2831853 * p.x) + cos(6.2831853 * p.y));
  float w = fwidth(f) * 0.8;
  return smoothstep(f - w, f + w, c * 1.04) * step(0.02, c);
}
void main() {
  vec2 px = gl_FragCoord.xy;
  float d = 0.84 + 0.16 * hash(floor(px * 0.5)); // riso ink never lies quite flat
  vec3 col = vec3(1.0);
  col *= mix(vec3(1.0), vec3(1.0, 0.424, 0.184), screen(px + o1, 0.2618, cover(px + o1).r) * d); // orange 15°
  col *= mix(vec3(1.0), vec3(0.463, 0.357, 0.655), screen(px + o2, 1.309, cover(px + o2).g) * d); // purple 75°
  col *= mix(vec3(1.0), vec3(0.239, 0.333, 0.533), screen(px + o3, 0.7854, cover(px + o3).b) * d); // blue 45°
  gl_FragColor = vec4(col, 1.0);
}`

const LIGHT = new THREE.Vector3(-0.45, 1, 0.55).normalize()
function inkMat(ink: [number, number, number], shade: number, map?: THREE.Texture, map2?: THREE.Texture): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: VERT, fragmentShader: INK_FRAG,
    uniforms: {
      ink: { value: new THREE.Vector3(...ink) }, shade: { value: shade }, light: { value: LIGHT },
      useMap: { value: map ? 1 : 0 }, amount: { value: 0 }, tMap: { value: map ?? null }, tMap2: { value: map2 ?? map ?? null },
    },
  })
}

type Pose = { x: number; y: number; z: number; rx: number; rz: number }
const HELD: Pose = { x: 0, y: 5.4, z: 3.4, rx: -2.2, rz: 0 } // face turned up to the viewer
const ABOVE: Pose = { x: 0, y: 2.2, z: 0, rx: 0, rz: 0 }
const DOWN: Pose = { x: 0, y: 0.03, z: 0, rx: 0, rz: 0 }
const ASIDE: Pose = { x: 8.6, y: 4.4, z: -2.6, rx: 0.35, rz: -0.45 }
const LOOP = 7, CONTACT = 3.05
const smooth = (k: number) => k * k * (3 - 2 * k)
const KEYS: [number, Pose, (k: number) => number][] = [
  [0, HELD, smooth], [1.6, HELD, smooth], [2.7, ABOVE, smooth], [CONTACT, DOWN, (k) => k * k],
  [3.5, DOWN, smooth], [4.5, ASIDE, smooth], [6.2, ASIDE, smooth], [LOOP, HELD, smooth],
]
function pose(t: number): Pose {
  let i = 0
  while (i < KEYS.length - 2 && t >= KEYS[i + 1][0]) i++
  const [t0, a] = KEYS[i], [t1, b, ease] = KEYS[i + 1]
  const k = ease(Math.min(1, Math.max(0, (t - t0) / (t1 - t0))))
  const mix = (u: number, v: number) => u + (v - u) * k
  return { x: mix(a.x, b.x), y: mix(a.y, b.y), z: mix(a.z, b.z), rx: mix(a.rx, b.rx), rz: mix(a.rz, b.rz) }
}

export function mountStamp(host: HTMLElement, fen: string, reduce: boolean, onContact: () => void): void {
  const canvas = host.querySelector('canvas')!
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'low-power' })
  renderer.setClearColor(0xffffff, 1)
  const target = new THREE.WebGLRenderTarget(1, 1, { samples: 4 })

  const face = sheet('#80b3ff', (g) => carve(g, fen)) // red-brown rubber: half orange, a third purple
  const paper = sheet('#ffffff', (g) => {
    g.fillStyle = g.strokeStyle = SEP.orange
    for (let f = 0; f < 8; f++) for (let r = 0; r < 8; r++) if ((f + r) % 2 === 0) g.fillRect(BX + f * SQ, BY + (7 - r) * SQ, SQ, SQ)
    g.lineWidth = 5; g.strokeRect(BX - 8, BY - 8, BS + 16, BS + 16)
    g.lineWidth = 3; for (let i = 0; i < 6; i++) g.strokeRect(30, BY + i * (BS / 6) + 6, 250, BS / 6 - 12)
  })
  // The print is the carving flipped top-to-bottom: exactly what pressing face-down does.
  const print = sheet('#ffffff', (g) => { g.translate(0, TH); g.scale(1, -1); carve(g, fen) })

  const scene = new THREE.Scene()
  const board = new THREE.Mesh(new THREE.PlaneGeometry(PW, PH), inkMat([0, 0, 0], 0, paper, print))
  board.rotation.x = -Math.PI / 2
  const shadowMat = new THREE.ShaderMaterial({
    vertexShader: VERT, fragmentShader: SHADOW_FRAG, depthWrite: false,
    uniforms: { amount: { value: 0 }, soft: { value: 0 } },
    // Multiply blending: the shadow overprints whatever ink is already on the paper.
    blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.DstColorFactor, blendDst: THREE.ZeroFactor,
  })
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(PW * 1.2, PH * 1.2), shadowMat)
  shadow.rotation.x = -Math.PI / 2; shadow.position.y = 0.02; shadow.renderOrder = 2

  const stamp = new THREE.Group()
  const part = (geo: THREE.BufferGeometry, mat: THREE.Material, y: number) => { const m = new THREE.Mesh(geo, mat); m.position.y = y; stamp.add(m); return m }
  part(new THREE.PlaneGeometry(PW, PH), inkMat([0, 0, 0], 0.35, face), 0).rotation.x = Math.PI / 2
  part(new THREE.BoxGeometry(PW, 0.36, PH), inkMat([0.5, 0.32, 0], 0.6), 0.19)
  const wood = inkMat([0.4, 0.06, 0], 0.75)
  part(new THREE.BoxGeometry(PW + 0.5, 1.2, PH + 0.5), wood, 0.97)
  part(new THREE.CylinderGeometry(1.1, 1.5, 2.6, 28), wood, 2.87)
  part(new THREE.SphereGeometry(1.7, 28, 16), wood, 5.3).scale.y = 0.8
  scene.add(board, shadow, stamp)

  const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100)
  const post = new THREE.Scene(), postCam = new THREE.Camera()
  const halftone = new THREE.ShaderMaterial({
    vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: HALFTONE_FRAG,
    uniforms: {
      tSrc: { value: target.texture }, res: { value: new THREE.Vector2() }, cell: { value: 4 },
      o1: { value: new THREE.Vector2() }, o2: { value: new THREE.Vector2() }, o3: { value: new THREE.Vector2() },
    },
  })
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), halftone); quad.frustumCulled = false
  post.add(quad)

  let pr = 1, width = 0
  function resize(): void {
    const w = host.clientWidth
    if (!w || w === width) return
    width = w
    const h = Math.round(w * (w < 520 ? 1.05 : 0.8))
    pr = Math.min(window.devicePixelRatio || 1, 2)
    renderer.setPixelRatio(pr); renderer.setSize(w, h)
    target.setSize(w * pr, h * pr)
    halftone.uniforms.res.value.set(w * pr, h * pr)
    halftone.uniforms.cell.value = 4.2 * pr
    camera.aspect = w / h
    camera.fov = w / h < 1.1 ? 42 : 34 // narrow cards: pull back so the stamp still fits
    camera.position.set(0, 18, 17); camera.lookAt(0, 2.4, 0.4)
    camera.updateProjectionMatrix()
  }

  function render(t: number, frame: number): void {
    const p = pose(t)
    stamp.position.set(p.x, p.y, p.z); stamp.rotation.set(p.rx, 0, p.rz)
    const squash = t >= CONTACT && t < CONTACT + 0.25 ? 0.93 : 1
    stamp.scale.set(1, squash, 1)
    const printed = t < CONTACT ? 0 : t < 6.2 ? 1 : 1 - (t - 6.2) / (LOOP - 6.2)
    ;(board.material as THREE.ShaderMaterial).uniforms.amount.value = printed
    shadow.position.set(p.x * 0.85, 0.02, p.z * 0.6)
    shadow.scale.setScalar(1 + p.y * 0.05)
    shadowMat.uniforms.amount.value = 0.55 * Math.max(0, 1 - p.y / 9)
    shadowMat.uniforms.soft.value = p.y * 0.06
    // Registration drifts a touch every frame, like a press that's been running all day.
    const j = (s: number) => Math.sin(frame * s) * 0.5 * pr
    halftone.uniforms.o1.value.set(j(1.7), j(2.3))
    halftone.uniforms.o2.value.set(2 * pr + j(3.1), -1 * pr + j(1.3))
    halftone.uniforms.o3.value.set(-1.5 * pr + j(2.9), 2 * pr + j(3.7))
    renderer.setRenderTarget(target); renderer.render(scene, camera)
    renderer.setRenderTarget(null); renderer.render(post, postCam)
  }

  resize()
  if (reduce) {
    render(5, 0)
    window.addEventListener('resize', () => { const w = width; resize(); if (w !== width) render(5, 0) })
    return
  }
  window.addEventListener('resize', resize)

  // Animate "on twos": 12 drawings a second, only while the card is on screen.
  let visible = false, raf = 0, last = -1
  const start = performance.now()
  function tick(now: number): void {
    raf = requestAnimationFrame(tick)
    const frame = Math.floor(((now - start) / 1000) * 12)
    if (frame === last) return
    const t = (frame / 12) % LOOP, prev = (last / 12) % LOOP
    if (last >= 0 && prev < CONTACT && t >= CONTACT) onContact()
    last = frame
    render(t, frame)
  }
  new IntersectionObserver(([e]) => {
    if (e.isIntersecting === visible) return
    visible = e.isIntersecting
    if (visible) raf = requestAnimationFrame(tick)
    else cancelAnimationFrame(raf)
  }).observe(host)
}
