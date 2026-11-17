// The network's real outputs, exported by scripts/export_web_data.py.
import raw from './data/knightvision.json'

export type Candidate = { san: string; uci: string; p: number }
export type Ply = {
  fen: string; san: string; uci: string
  share: number; legalMass: number; top: Candidate[]
  quilt?: string
}
export type Game = { plies: Ply[]; finalFen: string; result: string }
export type MirrorRow = { line: string; fen: string; v1: Candidate[]; v2: Candidate[] }
export type Belief = { from: string; to: string; p: number; legal: boolean }
export type EloGame = { elo: number; white: boolean; score: number; plies: number }

export type LayerStage = { name: string; channels: number; maps: string[]; parameters: number }
export type Lesson = { top: Candidate[]; answerShare: number; loss: number; legalMoves: number }

export type Data = {
  parameters: number
  checkpoint: { epoch: number; validationLoss: number }
  layers: { fen: string; stages: LayerStage[] }
  learning: {
    fen: string; answer: string; answerUci: string
    training: { positions: number; validation: number; batch: number; optimizer: string; lr: number }
    untrained: Lesson; trained: Lesson
  }
  filters: { channel: number; weights: number[]; bias: number }[]
  opera: Game
  win: Game
  mirror: MirrorRow[]
  constellation: { fen: string; next: string; legalMass: number; beliefs: Belief[] }
  elo: { rating: number; margin: number; games: EloGame[] } | null
}

export const data = raw as unknown as Data

// Quilt bytes: 4,096 values, index from_square × 64 + to_square (a1 = 0),
// 0 = one in ten million or less, 255 = certain, on a log scale.
export function quiltBytes(ply: Ply): Uint8Array {
  const binary = atob(ply.quilt ?? '')
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

// "e4 e5 Nf3" → "1.e4 e5 2.Nf3"
export function numbered(sans: string[], startPly = 0): string {
  return sans.map((san, i) => {
    const ply = startPly + i
    if (ply % 2 === 0) return `${ply / 2 + 1}.${san}`
    return i === 0 ? `${(ply + 1) / 2}…${san}` : san
  }).join(' ')
}

export function moveLabel(ply: number, san: string): string {
  const n = Math.floor(ply / 2) + 1
  return ply % 2 === 0 ? `${n}. ${san}` : `${n}… ${san}`
}

export const percent = (p: number): string =>
  p >= 0.995 ? '99%+' : p < 0.0005 ? '<0.1%' : p < 0.1 ? `${(p * 100).toFixed(1)}%` : `${Math.round(p * 100)}%`
