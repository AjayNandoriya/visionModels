import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import { viteStaticCopy } from 'vite-plugin-static-copy';


// https://vite.dev/config/
export default defineConfig({
  base: '/visionModels/', // Set the base path for GitHub Pages
  plugins: [react(), viteStaticCopy({
    targets: [
      {
        src: "node_modules/onnxruntime-web/dist/*.wasm",
        dest: "."
      }
    ]
  })],
  
  assetsInclude: ["**/*.onnx", "**/*.yaml"],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  server: {
    port: 3000,
    host:true,
  },
})
