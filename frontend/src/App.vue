<template>
  <div id="app">
    <el-container class="layout-container">
      <!-- 顶部导航栏 -->
      <el-header class="header">
        <div class="header-content">
          <div class="logo">
            <el-icon class="logo-icon"><Monitor /></el-icon>
            <span class="logo-text">钙信号分析平台</span>
          </div>
          <el-menu
            :default-active="$route.path"
            class="header-menu"
            mode="horizontal"
            router
          >
            <el-menu-item index="/">
              <el-icon><House /></el-icon>
              <span>首页</span>
            </el-menu-item>
            <el-menu-item index="/extraction">
              <el-icon><DataAnalysis /></el-icon>
              <span>事件提取</span>
            </el-menu-item>
            <el-menu-item index="/clustering">
              <el-icon><PieChart /></el-icon>
              <span>聚类分析</span>
            </el-menu-item>
            <el-menu-item index="/heatmap">
              <el-icon><TrendCharts /></el-icon>
              <span>热力图分析</span>
            </el-menu-item>
          </el-menu>
        </div>
      </el-header>

      <!-- 主内容区域 -->
      <el-main class="main-content">
        <router-view />
      </el-main>

      <!-- 底部 -->
      <el-footer class="footer">
        <div class="footer-content">
          <p>&copy; 2024 钙信号分析平台. 专为神经科学研究设计.</p>
        </div>
      </el-footer>
    </el-container>
  </div>
</template>

<script setup>
import { Monitor, House, DataAnalysis, PieChart, TrendCharts } from '@element-plus/icons-vue'
</script>

<style scoped>
.layout-container {
  min-height: 100vh;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 0;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}

.logo {
  display: flex;
  align-items: center;
  color: white;
  font-size: 20px;
  font-weight: bold;
}

.logo-icon {
  font-size: 28px;
  margin-right: 10px;
}

.logo-text {
  font-size: 24px;
}

.header-menu {
  background: transparent;
  border: none;
}

.header-menu .el-menu-item {
  color: rgba(255, 255, 255, 0.8);
  border-bottom: 2px solid transparent;
  transition: all 0.3s;
}

.header-menu .el-menu-item:hover,
.header-menu .el-menu-item.is-active {
  color: white;
  background: rgba(255, 255, 255, 0.1);
  border-bottom-color: white;
}

.main-content {
  background: #f5f7fa;
  min-height: calc(100vh - 120px);
  padding: 20px;
}

.footer {
  background: #2c3e50;
  color: white;
  text-align: center;
  padding: 20px 0;
}

.footer-content {
  max-width: 1200px;
  margin: 0 auto;
}

.footer p {
  margin: 0;
  opacity: 0.8;
}
</style>