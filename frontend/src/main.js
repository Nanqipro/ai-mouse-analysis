import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'

import App from './App.vue'
import router from './router'
import './style.css'

// 抑制ResizeObserver警告 - 使用更彻底的方法
const debounce = (func, wait) => {
  let timeout
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout)
      func(...args)
    }
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}

// 重写ResizeObserver以避免循环警告
if (window.ResizeObserver) {
  const OriginalResizeObserver = window.ResizeObserver
  window.ResizeObserver = class extends OriginalResizeObserver {
    constructor(callback) {
      const debouncedCallback = debounce(callback, 16) // 约60fps
      super(debouncedCallback)
    }
  }
}

// 抑制控制台中的ResizeObserver错误
const originalConsoleError = console.error
console.error = (...args) => {
  if (args[0] && args[0].toString().includes('ResizeObserver loop completed')) {
    return
  }
  originalConsoleError.apply(console, args)
}

// 抑制window error事件中的ResizeObserver错误
window.addEventListener('error', (e) => {
  if (e.message && e.message.includes('ResizeObserver loop completed')) {
    e.stopImmediatePropagation()
    e.preventDefault()
    return false
  }
})

const app = createApp(App)

// 注册所有图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

app.use(createPinia())
app.use(router)
app.use(ElementPlus, {
  locale: zhCn,
})

app.mount('#app')