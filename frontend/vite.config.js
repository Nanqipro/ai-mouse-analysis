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
    host: '0.0.0.0',
    // 增加请求头大小限制，解决431错误
    maxHttpHeaderSize: 65536, // 64KB请求头大小限制（增加到64KB）
    // 增加其他HTTP服务器选项
    hmr: {
      port: 5174,
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
        // 增加代理超时和缓冲区设置
        timeout: 120000, // 120秒超时
        // 增加buffer大小
        buffer: true,
        configure: (proxy, options) => {
          // 设置代理服务器的HTTP头部限制
          proxy.on('proxyReq', (proxyReq, req, res) => {
            // 移除可能导致问题的头部
            delete proxyReq.headers['if-none-match'];
            delete proxyReq.headers['if-modified-since'];
            
            // 设置更大的最大头部大小
            proxyReq.maxHeaderSize = 65536;
            
            // 设置超时
            proxyReq.setTimeout(120000);
          });
          
          proxy.on('error', (err, req, res) => {
            console.log('Proxy error:', err);
          });
        },
      },
    },
  },
  // 增加开发服务器配置
  define: {
    // 确保在开发环境中有足够的内存
    'process.env.NODE_OPTIONS': JSON.stringify('--max-http-header-size=65536 --max-old-space-size=4096'),
  },
  // 增加构建选项
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue', 'vue-router', 'pinia'],
          elementui: ['element-plus'],
        },
      },
    },
  },
})