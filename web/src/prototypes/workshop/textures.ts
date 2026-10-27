// Wood and felt drawn on canvases at load, since the prototype ships no image files.
// Textures are near-white; each material's colour supplies the species (maple, beech, baize).
import * as THREE from 'three'

export const INK = '#1E1B17'
export const RED = '#C0392B'
export const PAPER = '#F3EDE2'

function valueNoise(seed: number) {
  const table = new Float32Array(256 * 256)
  let s = seed
  for (let i = 0; i < table.length; i++) { s = (s * 1664525 + 1013904223) >>> 0; table[i] = s / 4294967296 }
  const at = (x: number, y: number) => table[(y & 255) * 256 + (x & 255)]
  return (x: number, y: number) => {
    const xi = Math.floor(x), yi = Math.floor(y), u = x - xi, v = y - yi
    const su = u * u * (3 - 2 * u), sv = v * v * (3 - 2 * v)
    const top = at(xi, yi) * (1 - su) + at(xi + 1, yi) * su
    const bottom = at(xi, yi + 1) * (1 - su) + at(xi + 1, yi + 1) * su
    return top * (1 - sv) + bottom * sv
  }
}

function canvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const c = document.createElement('canvas')
  c.width = w; c.height = h
  return [c, c.getContext('2d')!]
}

export function toTexture(c: HTMLCanvasElement, repeat = 1): THREE.CanvasTexture {
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  t.wrapS = t.wrapT = THREE.RepeatWrapping
  t.repeat.set(repeat, repeat)
  t.anisotropy = 8
  return t
}

// Grain: growth rings bent by low noise, then fine fibres stretched along the board.
export function grainCanvas(size = 512, seed = 7): HTMLCanvasElement {
  const [c, ctx] = canvas(size, size)
  const img = ctx.createImageData(size, size)
  const bend = valueNoise(seed), fibre = valueNoise(seed + 1)
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const u = x / size, v = y / size
      const ring = (v * 9 + bend(u * 2.2, v * 3) * 2.6) % 1
      const late = Math.pow(ring, 5) * 0.28
      const fib = (fibre(u * 4, v * 220) - 0.5) * 0.09
      const l = Math.max(0, Math.min(1, 0.97 - late + fib)) * 255
      const i = (y * size + x) * 4
      img.data[i] = l; img.data[i + 1] = l * 0.985; img.data[i + 2] = l * 0.96; img.data[i + 3] = 255
    }
  }
  ctx.putImageData(img, 0, 0)
  return c
}

export function feltTexture(): THREE.CanvasTexture {
  const size = 256
  const [c, ctx] = canvas(size, size)
  const img = ctx.createImageData(size, size)
  const mottle = valueNoise(3)
  for (let i = 0; i < size * size; i++) {
    const x = i % size, y = (i / size) | 0
    const l = (0.9 + (mottle(x / 24, y / 24) - 0.5) * 0.1 + (Math.random() - 0.5) * 0.16) * 255
    img.data.set([l, l, l, 255], i * 4)
  }
  ctx.putImageData(img, 0, 0)
  return toTexture(c, 36)
}

// A tray's top: tinted grain, an 8×8 grid burned in, and a paper label glued along the front.
// The tint is baked into the texture (not the material colour) so the paper label stays paper.
export type TrayLayout = { w: number; d: number; cell: number; zOffset: number }
export function trayTexture(grain: HTMLCanvasElement, label: string, tint: string, t: TrayLayout): THREE.CanvasTexture {
  const px = 200 // per world unit
  const W = Math.round(t.w * px), D = Math.round(t.d * px)
  const [c, ctx] = canvas(W, D)
  ctx.drawImage(grain, 0, 0, W, D)
  ctx.globalCompositeOperation = 'multiply'
  ctx.fillStyle = tint
  ctx.fillRect(0, 0, W, D)
  ctx.globalCompositeOperation = 'source-over'
  // Grid centred where the pieces sit: 8 cells of t.cell, shifted back by zOffset.
  const x0 = W / 2 - 4 * t.cell * px, z0 = D / 2 + (t.zOffset - 4 * t.cell) * px, step = t.cell * px
  ctx.strokeStyle = 'rgba(40, 20, 6, 0.55)'
  ctx.lineWidth = 2
  for (let i = 0; i <= 8; i++) {
    ctx.beginPath(); ctx.moveTo(x0 + i * step, z0); ctx.lineTo(x0 + i * step, z0 + 8 * step); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(x0, z0 + i * step); ctx.lineTo(x0 + 8 * step, z0 + i * step); ctx.stroke()
  }
  // The label: a strip of paper with a ruled edge, ink lettering.
  const top = z0 + 8 * step + 12, h = D - top - 12
  ctx.fillStyle = PAPER
  ctx.fillRect(x0, top, 8 * step, h)
  ctx.strokeStyle = 'rgba(30, 27, 23, 0.35)'
  ctx.strokeRect(x0 + 0.5, top + 0.5, 8 * step - 1, h - 1)
  ctx.fillStyle = INK
  ctx.font = `500 ${Math.round(h * 0.52)}px Jost, sans-serif`
  ctx.textBaseline = 'middle'
  ctx.fillText(label, x0 + 18, top + h / 2 + 2)
  const tex = toTexture(c)
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping
  return tex
}

// The pegboard's base: file letters along the front, rank numbers down the side, burned in.
export function baseTexture(grain: HTMLCanvasElement, pitch: number, size: number): THREE.CanvasTexture {
  const px = 1024 / size
  const [c, ctx] = canvas(1024, 1024)
  ctx.drawImage(grain, 0, 0, 1024, 1024)
  ctx.fillStyle = 'rgba(60, 30, 10, 0.8)'
  ctx.font = '500 26px Jost, sans-serif'
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  for (let i = 0; i < 8; i++) {
    const at = 512 + (i - 3.5) * pitch * px
    ctx.fillText('abcdefgh'[i], at, 1024 - 0.28 * px)
    ctx.fillText(String(8 - i), 0.26 * px, at)
  }
  return toTexture(c)
}
