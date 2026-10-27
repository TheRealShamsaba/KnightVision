import * as ort from 'onnxruntime-web/wasm'
import type { Chess, Square } from 'chess.js'
// Bundled by Vite: ONNX Runtime's wasm and its loader.
import ortWasm from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url'
import ortMjs from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url'

// KnightVision in the browser: the trained network (ChessNet, quantised to int8) run with ONNX Runtime.
// The encoding matches ai/ai.py, the layout the checkpoint was trained on: planes P N B R Q K
// (white 0–5, black 6–11), row 0 = rank 8, and move index = from_square × 64 + to_square with a1 = 0.

const MODEL = `${import.meta.env.BASE_URL}knightvision.onnx`
const PLANES = 'PNBRQK'

export type Thought = { from: Square; to: Square; san: string; p: number }

let session: Promise<ort.InferenceSession> | null = null

export function loadEngine(onProgress?: (fraction: number) => void): Promise<ort.InferenceSession> {
  if (session) return session
  ort.env.wasm.wasmPaths = { wasm: ortWasm, mjs: ortMjs }
  ort.env.wasm.numThreads = 1
  // Inference runs in ORT's own worker, so the page never freezes while it thinks.
  ort.env.wasm.proxy = true
  session = (async () => {
    const response = await fetch(MODEL)
    if (!response.ok) throw new Error(`model: ${response.status}`)
    const total = Number(response.headers.get('content-length')) || 25_575_213
    const reader = response.body!.getReader()
    const chunks: Uint8Array[] = []
    let received = 0
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      received += value.length
      onProgress?.(Math.min(1, received / total))
    }
    const bytes = new Uint8Array(received)
    let offset = 0
    chunks.forEach((chunk) => { bytes.set(chunk, offset); offset += chunk.length })
    return ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] })
  })()
  session.catch(() => { session = null })
  return session
}

const squareIndex = (square: string): number => (Number(square[1]) - 1) * 8 + (square.charCodeAt(0) - 97)

function encode(game: Chess): Float32Array {
  const planes = new Float32Array(12 * 64)
  game.board().forEach((rank, row) => rank.forEach((piece, col) => {
    if (!piece) return
    const plane = PLANES.indexOf(piece.type.toUpperCase()) + (piece.color === 'w' ? 0 : 6)
    planes[plane * 64 + row * 8 + col] = 1
  }))
  return planes
}

/** Every legal move with its share of the network's belief, best first. Promotions are to a queen. */
export async function think(game: Chess): Promise<Thought[]> {
  const engine = await loadEngine()
  const output = await engine.run({ board: new ort.Tensor('float32', encode(game), [1, 12, 8, 8]) })
  const logits = output.policy.data as Float32Array
  const legal = game.moves({ verbose: true }).filter((m) => !m.promotion || m.promotion === 'q')
  const scores = legal.map((m) => logits[squareIndex(m.from) * 64 + squareIndex(m.to)])
  const max = Math.max(...scores)
  const weights = scores.map((s) => Math.exp(s - max))
  const sum = weights.reduce((a, b) => a + b, 0)
  return legal
    .map((m, i) => ({ from: m.from, to: m.to, san: m.san, p: weights[i] / sum }))
    .sort((a, b) => b.p - a.p)
}
