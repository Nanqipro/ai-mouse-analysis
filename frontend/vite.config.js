import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

export default defineConfig({
  plugins: [
    vue(),
    AutoImport({
      resolvers: [ElementPlusResolver()],
    }),
    Components({
      resolvers: [ElementPlusResolver()],
    }),
  ],
  define: {
    __VUE_PROD_HYDRATION_MISMATCH_DETAILS__: 'false'
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 5175,
    hmr: {
      port: 24678
    },
    headers: {
      'Cache-Control': 'no-cache'
    },
    allowedHosts: [
      'localhost',
      '127.0.0.1',
      '102qldp675617.vicp.fun'
    ]
  }
})