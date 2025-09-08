import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Extraction from '../views/Extraction.vue'
import Clustering from '../views/Clustering.vue'
import Heatmap from '../views/Heatmap.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home,
    meta: {
      title: '首页'
    }
  },
  {
    path: '/extraction',
    name: 'Extraction',
    component: Extraction,
    meta: {
      title: '事件提取'
    }
  },
  {
    path: '/clustering',
    name: 'Clustering',
    component: Clustering,
    meta: {
      title: '聚类分析'
    }
  },
  {
    path: '/heatmap',
    name: 'Heatmap',
    component: Heatmap,
    meta: {
      title: '热力图分析'
    }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

// 路由守卫，设置页面标题
router.beforeEach((to, from, next) => {
  if (to.meta.title) {
    document.title = `${to.meta.title} - 钙信号分析平台`
  }
  next()
})

export default router