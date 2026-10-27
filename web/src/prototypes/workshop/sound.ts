// A wooden clack, synthesised: a filtered noise burst for the knock, a short resonant tone for the
// body of the wood. Pitch varies on every hit so a row of landings never sounds like a machine.
export class Clack {
  on = false
  private ctx: AudioContext | null = null
  private noise: AudioBuffer | null = null
  private last = 0

  async toggle(): Promise<boolean> {
    this.on = !this.on
    if (this.on && !this.ctx) {
      this.ctx = new AudioContext()
      const length = Math.floor(this.ctx.sampleRate * 0.06)
      this.noise = this.ctx.createBuffer(1, length, this.ctx.sampleRate)
      const ch = this.noise.getChannelData(0)
      for (let i = 0; i < length; i++) ch[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, 4)
    }
    if (this.on) await this.ctx!.resume()
    return this.on
  }

  play(volume = 1) {
    const ctx = this.ctx
    if (!this.on || !ctx || !this.noise) return
    const now = ctx.currentTime
    if (now - this.last < 0.045) return // a dozen pieces landing together should still sound like a hand, not a hailstorm
    this.last = now
    const rate = 0.8 + Math.random() * 0.45

    const out = ctx.createGain()
    out.gain.value = 0.5 * volume
    out.connect(ctx.destination)

    const knock = ctx.createBufferSource()
    knock.buffer = this.noise
    knock.playbackRate.value = rate
    const band = ctx.createBiquadFilter()
    band.type = 'bandpass'
    band.frequency.value = 2400 * rate
    band.Q.value = 1.4
    knock.connect(band).connect(out)
    knock.start(now)

    const body = ctx.createOscillator()
    body.type = 'triangle'
    body.frequency.value = 720 * rate
    const env = ctx.createGain()
    env.gain.setValueAtTime(0.35, now)
    env.gain.exponentialRampToValueAtTime(0.001, now + 0.09)
    body.connect(env).connect(out)
    body.start(now)
    body.stop(now + 0.1)
  }
}
