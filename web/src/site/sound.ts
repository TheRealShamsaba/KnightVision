// A rubber-stamp "thunk", synthesised: a low falling thump (the block hitting the desk)
// plus a short band-passed noise burst (paper giving under it). Off until asked for.
let audio: AudioContext | null = null
let on = false

export function setSound(enabled: boolean): void {
  on = enabled
  if (on) { audio ??= new AudioContext(); void audio.resume() }
}

export function thunk(level = 1): void {
  if (!on || !audio) return
  const a = audio, t = a.currentTime
  const out = a.createGain(); out.gain.value = level; out.connect(a.destination)

  const body = a.createOscillator(), bodyGain = a.createGain()
  body.type = 'sine'
  body.frequency.setValueAtTime(120, t); body.frequency.exponentialRampToValueAtTime(45, t + 0.12)
  bodyGain.gain.setValueAtTime(0.0001, t)
  bodyGain.gain.exponentialRampToValueAtTime(0.7, t + 0.004)
  bodyGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.22)
  body.connect(bodyGain).connect(out); body.start(t); body.stop(t + 0.25)

  const n = Math.floor(a.sampleRate * 0.05), buf = a.createBuffer(1, n, a.sampleRate), d = buf.getChannelData(0)
  for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / n, 3)
  const click = a.createBufferSource(), band = a.createBiquadFilter(), clickGain = a.createGain()
  click.buffer = buf; band.type = 'bandpass'; band.frequency.value = 2400; band.Q.value = 0.9; clickGain.gain.value = 0.35
  click.connect(band).connect(clickGain).connect(out); click.start(t + 0.003)
}
