import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import { viteStaticCopy } from 'vite-plugin-static-copy';


// https://vite.dev/config/
export default defineConfig({
  base: '/visionmodels/', // Set the base path for GitHub Pages
  plugins: [react(),viteStaticCopy({
    targets: [
      {
        src: "node_modules/onnxruntime-web/dist/*.wasm",
        dest: "."
      }
    ]
  })],
  
  assetsInclude: ["**/*.onnx"],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },

})
