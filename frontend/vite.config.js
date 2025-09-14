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
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 5173,
    // 增加请求头大小限制，解决431错误
    maxHttpHeaderSize: 32768, // 32KB请求头大小限制（默认8KB）
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
        // 增加代理超时和缓冲区设置
        timeout: 60000, // 60秒超时
        configure: (proxy, options) => {
          proxy.on('proxyReq', (proxyReq, req, res) => {
            // 增加请求头大小限制
            proxyReq.setHeader('max-http-header-size', '32768');
            // 设置更大的最大头部大小
            proxyReq.maxHeaderSize = 32768;
          });
        },
      },
    },
  },
  // 增加开发服务器配置
  define: {
    // 确保在开发环境中有足够的内存
    'process.env.NODE_OPTIONS': JSON.stringify('--max-http-header-size=32768'),
  },
})