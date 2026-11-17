// Tilt-shift: a sharp band across the frame, blur growing above and below it. It's the lens trick
// that makes real streets look like models, used here the other way round: a model desk shot like one.
import * as THREE from 'three'
import { ShaderPass } from 'three/examples/jsm/postprocessing/ShaderPass.js'

export function tiltShift(): ShaderPass {
  return new ShaderPass({
    uniforms: {
      tDiffuse: { value: null },
      resolution: { value: new THREE.Vector2(1, 1) },
      focus: { value: 0.5 },    // where the sharp band sits, 0 = bottom, 1 = top
      band: { value: 0.16 },    // half-height of the sharp band
      amount: { value: 7.0 },   // blur radius in pixels at the frame's edge
    },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D tDiffuse;
      uniform vec2 resolution;
      uniform float focus, band, amount;
      varying vec2 vUv;
      void main() {
        float d = abs(vUv.y - focus);
        float r = smoothstep(band, band + 0.32, d) * amount;
        vec4 sum = vec4(0.0);
        float total = 0.0;
        // A small disc of samples; its radius grows with distance from the sharp band.
        for (int i = -3; i <= 3; i++) {
          for (int j = -3; j <= 3; j++) {
            vec2 o = vec2(float(i), float(j));
            float w = 1.0 - length(o) / 4.5;
            if (w <= 0.0) continue;
            sum += texture2D(tDiffuse, vUv + o * r / (3.0 * resolution)) * w;
            total += w;
          }
        }
        vec3 col = (sum / total).rgb;
        // Model photographs run a touch warmer and more saturated, and fall off at the corners.
        float grey = dot(col, vec3(0.299, 0.587, 0.114));
        col = mix(vec3(grey), col, 1.12);
        vec2 c = vUv - 0.5;
        col *= 1.0 - dot(c, c) * 0.55;
        gl_FragColor = vec4(col, 1.0);
      }`,
  })
}
