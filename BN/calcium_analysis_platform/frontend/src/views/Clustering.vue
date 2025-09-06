<template>
  <div class="clustering">
    <h1 class="page-title">
      <el-icon><Histogram /></el-icon>
      聚类分析
    </h1>
    
    <el-alert
      title="功能说明"
      type="info"
      :closable="false"
      show-icon
      class="info-alert"
    >
      <template #default>
        在此页面，您可以：<br>
        1. <strong>选择数据文件</strong>: 从事件提取结果中选择要分析的文件。<br>
        2. <strong>配置聚类参数</strong>: 设置K值、降维方法等参数。<br>
        3. <strong>查看聚类结果</strong>: 可视化聚类结果和特征分布。
      </template>
    </el-alert>

    <el-row :gutter="20">
      <!-- 左侧参数面板 -->
      <el-col :xs="24" :sm="24" :md="8" :lg="6">
        <div class="params-panel card">
          <h3 class="section-title">
            <el-icon><Setting /></el-icon>
            聚类参数
          </h3>
          
          <el-form :model="params" label-width="120px" size="small">
            <el-form-item label="数据文件">
              <el-select
                v-model="params.selectedFile"
                placeholder="选择数据文件"
                style="width: 100%"
                @change="handleFileChange"
              >
                <el-option
                  v-for="file in resultFiles"
                  :key="file.filename"
                  :label="file.filename"
                  :value="file.filename"
                >
                  <div style="display: flex; justify-content: space-between;">
                    <span>{{ file.filename }}</span>
                    <span style="color: #8492a6; font-size: 12px;">
                      {{ formatDate(file.created_at) }}
                    </span>
                  </div>
                </el-option>
              </el-select>
              <div class="param-help">选择事件提取的结果文件</div>
            </el-form-item>
            
            <el-form-item label="聚类数量 (K)">
              <el-input-number
                v-model="params.k"
                :min="2"
                :max="20"
                style="width: 100%"
              />
              <div class="param-help">K-means聚类的簇数</div>
            </el-form-item>
            
            <el-form-item label="降维方法">
              <el-select v-model="params.reduction_method" style="width: 100%">
                <el-option label="PCA" value="pca" />
                <el-option label="t-SNE" value="tsne" />
              </el-select>
              <div class="param-help">用于2D可视化的降维方法</div>
            </el-form-item>
            
            <el-form-item label="特征权重">
              <el-switch
                v-model="params.use_weights"
                active-text="启用"
                inactive-text="禁用"
              />
              <div class="param-help">是否对特征进行加权</div>
            </el-form-item>
            
            <el-form-item v-if="params.use_weights" label="振幅权重">
              <el-input-number
                v-model="params.amplitude_weight"
                :min="0.1"
                :max="5"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
            </el-form-item>
            
            <el-form-item v-if="params.use_weights" label="持续时间权重">
              <el-input-number
                v-model="params.duration_weight"
                :min="0.1"
                :max="5"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
            </el-form-item>
            
            <el-form-item v-if="params.use_weights" label="上升时间权重">
              <el-input-number
                v-model="params.rise_time_weight"
                :min="0.1"
                :max="5"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
            </el-form-item>
            
            <el-form-item v-if="params.use_weights" label="衰减时间权重">
              <el-input-number
                v-model="params.decay_time_weight"
                :min="0.1"
                :max="5"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
            </el-form-item>
          </el-form>
          
          <el-button
            type="primary"
            :loading="analysisLoading"
            :disabled="!params.selectedFile"
            @click="startClustering"
            style="width: 100%; margin-top: 20px;"
          >
            <el-icon><Cpu /></el-icon>
            开始聚类分析
          </el-button>
        </div>
      </el-col>
      
      <!-- 右侧结果展示 -->
      <el-col :xs="24" :sm="24" :md="16" :lg="18">
        <!-- 聚类结果概览 -->
        <div v-if="clusteringResult" class="result-overview card">
          <h3 class="section-title">
            <el-icon><PieChart /></el-icon>
            聚类结果概览
          </h3>
          
          <el-row :gutter="20">
            <el-col :span="8" v-for="(summary, index) in clusteringResult.cluster_summary" :key="index">
              <div class="cluster-card">
                <div class="cluster-header">
                  <span class="cluster-title">簇 {{ index }}</span>
                  <el-tag :color="getClusterColor(index)" class="cluster-tag">
                    {{ summary.count }} 个事件
                  </el-tag>
                </div>
                <div class="cluster-stats">
                  <div class="stat-item">
                    <span class="stat-label">平均振幅:</span>
                    <span class="stat-value">{{ summary.mean_amplitude?.toFixed(2) || 'N/A' }}</span>
                  </div>
                  <div class="stat-item">
                    <span class="stat-label">平均持续时间:</span>
                    <span class="stat-value">{{ summary.mean_duration?.toFixed(2) || 'N/A' }}s</span>
                  </div>
                  <div class="stat-item">
                    <span class="stat-label">平均上升时间:</span>
                    <span class="stat-value">{{ summary.mean_rise_time?.toFixed(2) || 'N/A' }}s</span>
                  </div>
                  <div class="stat-item">
                    <span class="stat-label">平均衰减时间:</span>
                    <span class="stat-value">{{ summary.mean_decay_time?.toFixed(2) || 'N/A' }}s</span>
                  </div>
                </div>
              </div>
            </el-col>
          </el-row>
        </div>
        
        <!-- 2D聚类可视化 -->
        <div v-if="clusteringResult?.cluster_plot" class="visualization-section card">
          <h3 class="section-title">
            <el-icon><TrendCharts /></el-icon>
            2D聚类可视化 ({{ params.reduction_method.toUpperCase() }})
          </h3>
          
          <div class="plot-container">
            <img :src="clusteringResult.cluster_plot" alt="聚类可视化" class="result-image" />
          </div>
        </div>
        
        <!-- 特征分布图 -->
        <div v-if="clusteringResult?.feature_plots" class="feature-distribution card">
          <h3 class="section-title">
            <el-icon><DataLine /></el-icon>
            特征分布分析
          </h3>
          
          <el-row :gutter="20">
            <el-col :span="12" v-for="(plot, feature) in clusteringResult.feature_plots" :key="feature">
              <div class="feature-plot">
                <h4>{{ getFeatureLabel(feature) }}</h4>
                <img :src="plot" :alt="feature + '分布'" class="feature-image" />
              </div>
            </el-col>
          </el-row>
        </div>
        
        <!-- 下载区域 -->
        <div v-if="clusteringResult" class="download-section card">
          <h3 class="section-title">
            <el-icon><Download /></el-icon>
            结果下载
          </h3>
          
          <el-row :gutter="20">
            <el-col :span="12">
              <el-button
                type="success"
                @click="downloadClusteredData"
                style="width: 100%"
              >
                <el-icon><Download /></el-icon>
                下载聚类结果数据
              </el-button>
            </el-col>
            <el-col :span="12">
              <el-button
                type="primary"
                @click="refreshResultFiles"
                style="width: 100%"
              >
                <el-icon><Refresh /></el-icon>
                刷新文件列表
              </el-button>
            </el-col>
          </el-row>
        </div>
        
        <!-- 空状态 -->
        <div v-if="!clusteringResult && !analysisLoading" class="empty-state card">
          <el-empty description="请选择数据文件并开始聚类分析">
            <el-button type="primary" @click="refreshResultFiles">
              <el-icon><Refresh /></el-icon>
              刷新文件列表
            </el-button>
          </el-empty>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Histogram,
  Setting,
  PieChart,
  TrendCharts,
  DataLine,
  Download,
  Refresh,
  Cpu
} from '@element-plus/icons-vue'
import { clusteringAPI, downloadAPI } from '@/api'

// 响应式数据
const resultFiles = ref([])
const analysisLoading = ref(false)
const clusteringResult = ref(null)

// 参数配置
const params = reactive({
  selectedFile: '',
  k: 3,
  reduction_method: 'pca',
  use_weights: false,
  amplitude_weight: 1.0,
  duration_weight: 1.0,
  rise_time_weight: 1.0,
  decay_time_weight: 1.0
})

// 生命周期
onMounted(() => {
  loadResultFiles()
})

// 加载结果文件列表
const loadResultFiles = async () => {
  try {
    const response = await clusteringAPI.getResultFiles()
    if (response.success) {
      resultFiles.value = response.files
    }
  } catch (error) {
    console.error('加载文件列表失败:', error)
  }
}

// 刷新文件列表
const refreshResultFiles = () => {
  loadResultFiles()
  ElMessage.success('文件列表已刷新')
}

// 文件选择变化
const handleFileChange = () => {
  clusteringResult.value = null
}

// 开始聚类分析
const startClustering = async () => {
  if (!params.selectedFile) {
    ElMessage.warning('请选择数据文件')
    return
  }
  
  analysisLoading.value = true
  try {
    const requestData = {
      filename: params.selectedFile,
      k: params.k,
      reduction_method: params.reduction_method
    }
    
    // 如果启用权重，添加权重参数
    if (params.use_weights) {
      requestData.feature_weights = {
        amplitude: params.amplitude_weight,
        duration: params.duration_weight,
        rise_time: params.rise_time_weight,
        decay_time: params.decay_time_weight
      }
    }
    
    const response = await clusteringAPI.analyze(requestData)
    if (response.success) {
      clusteringResult.value = response
      ElMessage.success('聚类分析完成')
    }
  } catch (error) {
    console.error('聚类分析失败:', error)
  } finally {
    analysisLoading.value = false
  }
}

// 下载聚类结果
const downloadClusteredData = async () => {
  if (!clusteringResult.value?.clustered_file) {
    ElMessage.error('没有可下载的聚类结果文件')
    return
  }
  
  try {
    const response = await downloadAPI.downloadFile(clusteringResult.value.clustered_file)
    
    // 创建下载链接
    const blob = new Blob([response], {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = clusteringResult.value.clustered_file
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
    
    ElMessage.success('文件下载成功')
  } catch (error) {
    console.error('下载失败:', error)
  }
}

// 工具函数
const formatDate = (dateString) => {
  return new Date(dateString).toLocaleString('zh-CN')
}

const getClusterColor = (index) => {
  const colors = ['#409EFF', '#67C23A', '#E6A23C', '#F56C6C', '#909399', '#9C27B0', '#FF9800']
  return colors[index % colors.length]
}

const getFeatureLabel = (feature) => {
  const labels = {
    amplitude: '振幅',
    duration: '持续时间',
    rise_time: '上升时间',
    decay_time: '衰减时间',
    fwhm: '半高宽',
    auc: '曲线下面积',
    snr: '信噪比'
  }
  return labels[feature] || feature
}
</script>

<style scoped>
.clustering {
  max-width: 1400px;
  margin: 0 auto;
}

.info-alert {
  margin-bottom: 20px;
}

.param-help {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}

.result-overview {
  margin-bottom: 20px;
}

.cluster-card {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 15px;
  background: #fafafa;
  height: 100%;
}

.cluster-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.cluster-title {
  font-weight: bold;
  font-size: 16px;
  color: #2c3e50;
}

.cluster-tag {
  color: white;
  border: none;
}

.cluster-stats {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  font-size: 14px;
}

.stat-label {
  color: #606266;
}

.stat-value {
  font-weight: bold;
  color: #2c3e50;
}

.visualization-section {
  margin-bottom: 20px;
}

.plot-container {
  text-align: center;
}

.feature-distribution {
  margin-bottom: 20px;
}

.feature-plot {
  text-align: center;
  margin-bottom: 20px;
}

.feature-plot h4 {
  margin-bottom: 10px;
  color: #2c3e50;
  font-size: 14px;
}

.feature-image {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.download-section {
  margin-bottom: 20px;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .cluster-card {
    margin-bottom: 15px;
  }
  
  .feature-plot {
    margin-bottom: 30px;
  }
}
</style>