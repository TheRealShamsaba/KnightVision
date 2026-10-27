import { resolve } from 'node:path'
import { defineConfig } from 'vite'

export default defineConfig({
  // onnxruntime-web ships its own wasm loader; let Vite serve it as-is.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        flight: resolve(__dirname, 'flight.html'),
        prototypes: resolve(__dirname, 'prototypes/index.html'),
        workshop: resolve(__dirname, 'prototypes/workshop.html'),
        correspondence: resolve(__dirname, 'prototypes/correspondence.html'),
        notebook: resolve(__dirname, 'prototypes/notebook.html'),
      },
    },
  },
})
