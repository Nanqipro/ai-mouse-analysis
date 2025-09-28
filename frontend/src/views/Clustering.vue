<template>
  <div class="clustering">
    <h1 class="page-title">
      <el-icon><PieChart /></el-icon>
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
                :loading="filesLoading"
                :disabled="filesLoading"
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
              <div class="param-help">
                {{ filesLoading ? '正在加载文件列表...' : '选择事件提取的结果文件' }}
              </div>
              
              <!-- 如果没有文件显示提示 -->
              <div v-if="!filesLoading && resultFiles.length === 0" class="no-files-tip">
                <el-alert
                  title="暂无数据文件"
                  type="warning"
                  :closable="false"
                  show-icon
                  style="margin-top: 10px;"
                >
                  <template #default>
                    请先在 <strong>事件提取</strong> 页面处理数据文件，然后返回此页面进行聚类分析。
                  </template>
                </el-alert>
              </div>
            </el-form-item>
            
            <el-form-item label="聚类算法">
              <el-select v-model="params.algorithm" style="width: 100%" @change="handleAlgorithmChange">
                <el-option label="K-means" value="kmeans" />
                <el-option label="DBSCAN" value="dbscan" />
              </el-select>
              <div class="param-help">选择聚类算法</div>
            </el-form-item>
            
            <!-- K-means 特定参数 -->
            <template v-if="params.algorithm === 'kmeans'">
              <el-form-item label="自动确定K值">
                <el-switch
                  v-model="params.auto_k"
                  active-text="启用"
                  inactive-text="禁用"
                  @change="handleAutoKChange"
                />
                <div class="param-help">自动确定最佳聚类数</div>
              </el-form-item>
              
              <el-form-item v-if="!params.auto_k" label="聚类数量 (K)">
                <el-input-number
                  v-model="params.k"
                  :min="2"
                  :max="20"
                  style="width: 100%"
                />
                <div class="param-help">K-means聚类的簇数</div>
              </el-form-item>
              
              <el-form-item v-if="params.auto_k" label="K值搜索范围">
                <el-row :gutter="10">
                  <el-col :span="11">
                    <el-input-number
                      v-model="params.auto_k_min"
                      :min="2"
                      :max="15"
                      placeholder="最小值"
                      style="width: 100%"
                    />
                  </el-col>
                  <el-col :span="2" style="text-align: center; line-height: 32px;">-</el-col>
                  <el-col :span="11">
                    <el-input-number
                      v-model="params.auto_k_max"
                      :min="3"
                      :max="20"
                      placeholder="最大值"
                      style="width: 100%"
                    />
                  </el-col>
                </el-row>
                <div class="param-help">搜索最佳K值的范围</div>
              </el-form-item>
            </template>
            
            <!-- DBSCAN 特定参数 -->
            <template v-if="params.algorithm === 'dbscan'">
              <el-form-item label="邻域半径 (eps)">
                <el-input-number
                  v-model="params.dbscan_eps"
                  :min="0.1"
                  :max="2.0"
                  :step="0.1"
                  :precision="1"
                  style="width: 100%"
                />
                <div class="param-help">DBSCAN算法的邻域半径参数</div>
              </el-form-item>
              
              <el-form-item label="最小样本数">
                <el-input-number
                  v-model="params.dbscan_min_samples"
                  :min="2"
                  :max="20"
                  style="width: 100%"
                />
                <div class="param-help">形成核心点的最小邻居数</div>
              </el-form-item>
            </template>
            
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
            
            <!-- 特征权重配置 -->
            <template v-if="params.use_weights">
              <el-form-item label="振幅权重">
                <el-input-number
                  v-model="params.amplitude_weight"
                  :min="0.1"
                  :max="5"
                  :step="0.1"
                  :precision="1"
                  style="width: 100%"
                />
              </el-form-item>
              
              <el-form-item label="持续时间权重">
                <el-input-number
                  v-model="params.duration_weight"
                  :min="0.1"
                  :max="5"
                  :step="0.1"
                  :precision="1"
                  style="width: 100%"
                />
              </el-form-item>
              
              <el-form-item label="半高宽权重">
                <el-input-number
                  v-model="params.fwhm_weight"
                  :min="0.1"
                  :max="5"
                  :step="0.1"
                  :precision="1"
                  style="width: 100%"
                />
              </el-form-item>
              
              <el-form-item label="上升时间权重">
                <el-input-number
                  v-model="params.rise_time_weight"
                  :min="0.1"
                  :max="5"
                  :step="0.1"
                  :precision="1"
                  style="width: 100%"
                />
              </el-form-item>
              
              <el-form-item label="衰减时间权重">
                <el-input-number
                  v-model="params.decay_time_weight"
                  :min="0.1"
                  :max="5"
                  :step="0.1"
                  :precision="1"
                  style="width: 100%"
                />
              </el-form-item>
              
              <el-form-item label="曲线下面积权重">
                <el-input-number
                  v-model="params.auc_weight"
                  :min="0.1"
                  :max="5"
                  :step="0.1"
                  :precision="1"
                  style="width: 100%"
                />
              </el-form-item>
              
              <el-form-item label="信噪比权重">
                <el-input-number
                  v-model="params.snr_weight"
                  :min="0.1"
                  :max="5"
                  :step="0.1"
                  :precision="1"
                  style="width: 100%"
                />
              </el-form-item>
            </template>
          </el-form>
          
          <el-row :gutter="10" style="margin-top: 20px;">
            <el-col :span="24">
              <el-button
                type="primary"
                :loading="analysisLoading"
                :disabled="!params.selectedFile"
                @click="startClustering"
                style="width: 100%;"
              >
                <el-icon><Cpu /></el-icon>
                开始聚类分析
              </el-button>
            </el-col>
          </el-row>
          
          <el-row :gutter="10" style="margin-top: 10px;" v-if="params.algorithm === 'kmeans'">
            <el-col :span="12">
              <el-button
                type="success"
                :loading="optimalKLoading"
                :disabled="!params.selectedFile"
                @click="findOptimalK"
                size="small"
                style="width: 100%;"
              >
                <el-icon><TrendCharts /></el-icon>
                寻找最佳K值
              </el-button>
            </el-col>
            <el-col :span="12">
              <el-button
                type="warning"
                :loading="compareKLoading"
                :disabled="!params.selectedFile"
                @click="compareKValues"
                size="small"
                style="width: 100%;"
              >
                <el-icon><PieChart /></el-icon>
                比较K值效果
              </el-button>
            </el-col>
          </el-row>
        </div>
      </el-col>
      
      <!-- 右侧结果展示 -->
      <el-col :xs="24" :sm="24" :md="16" :lg="18">
        <!-- 聚类结果概览 -->
        <div v-if="clusteringResult" class="result-overview card">
          <h3 class="section-title">
            <el-icon><PieChart /></el-icon>
            聚类结果概览
            <el-tag style="margin-left: 10px;" type="info">
              {{ clusteringResult.algorithm }} | K={{ clusteringResult.n_clusters }}
            </el-tag>
          </h3>
          
          <el-row :gutter="20">
            <el-col :span="6" v-for="(summary, index) in clusteringResult.cluster_summary" :key="index">
              <div class="cluster-card">
                <div class="cluster-header">
                  <span class="cluster-title">簇 {{ summary.Cluster }}</span>
                  <el-tag :color="getClusterColor(index)" class="cluster-tag">
                    {{ summary.Count }} 个事件 ({{ summary.Percentage }})
                  </el-tag>
                </div>
                <div class="cluster-stats">
                  <div class="stat-item" v-if="summary.amplitude_mean !== undefined">
                    <span class="stat-label">平均振幅:</span>
                    <span class="stat-value">{{ summary.amplitude_mean?.toFixed(2) || 'N/A' }}</span>
                  </div>
                  <div class="stat-item" v-if="summary.duration_mean !== undefined">
                    <span class="stat-label">平均持续时间:</span>
                    <span class="stat-value">{{ summary.duration_mean?.toFixed(2) || 'N/A' }}s</span>
                  </div>
                  <div class="stat-item" v-if="summary.rise_time_mean !== undefined">
                    <span class="stat-label">平均上升时间:</span>
                    <span class="stat-value">{{ summary.rise_time_mean?.toFixed(2) || 'N/A' }}s</span>
                  </div>
                  <div class="stat-item" v-if="summary.decay_time_mean !== undefined">
                    <span class="stat-label">平均衰减时间:</span>
                    <span class="stat-value">{{ summary.decay_time_mean?.toFixed(2) || 'N/A' }}s</span>
                  </div>
                </div>
              </div>
            </el-col>
          </el-row>
        </div>
        
        <!-- 最佳K值分析结果 -->
        <div v-if="optimalKResult" class="optimal-k-result card">
          <h3 class="section-title">
            <el-icon><TrendCharts /></el-icon>
            最佳K值分析结果
          </h3>
          
          <el-row :gutter="20">
            <el-col :span="8">
              <div class="analysis-summary">
                <div class="summary-item">
                  <span class="summary-label">推荐K值:</span>
                  <span class="summary-value optimal-k">{{ optimalKResult.optimal_k }}</span>
                </div>
                <div class="summary-item">
                  <span class="summary-label">搜索范围:</span>
                  <span class="summary-value">{{ optimalKResult.k_range[0] }} - {{ optimalKResult.k_range[optimalKResult.k_range.length - 1] }}</span>
                </div>
                <div class="summary-item">
                  <span class="summary-label">最高轮廓系数:</span>
                  <span class="summary-value">{{ Math.max(...optimalKResult.silhouette_scores).toFixed(3) }}</span>
                </div>
              </div>
            </el-col>
            <el-col :span="16">
              <div class="plot-container">
                <img :src="optimalKResult.optimal_k_plot" alt="最佳K值分析" class="result-image" />
              </div>
            </el-col>
          </el-row>
        </div>
        
        <!-- K值比较结果 -->
        <div v-if="compareKResult" class="compare-k-result card">
          <h3 class="section-title">
            <el-icon><PieChart /></el-icon>
            K值比较分析结果
          </h3>
          
          <el-row :gutter="20">
            <el-col :span="8">
              <div class="analysis-summary">
                <div class="summary-item">
                  <span class="summary-label">最佳K值:</span>
                  <span class="summary-value optimal-k">{{ compareKResult.best_k }}</span>
                </div>
                <div class="summary-item">
                  <span class="summary-label">比较的K值:</span>
                  <span class="summary-value">{{ compareKResult.k_values.join(', ') }}</span>
                </div>
                <div class="k-scores">
                  <div v-for="(score, k) in compareKResult.silhouette_scores" :key="k" class="score-item">
                    <span>K={{ k }}:</span>
                    <span :class="{ 'best-score': k == compareKResult.best_k }">{{ score.toFixed(3) }}</span>
                  </div>
                </div>
              </div>
            </el-col>
            <el-col :span="16">
              <div class="plot-container">
                <img :src="compareKResult.comparison_plot" alt="K值比较" class="result-image" />
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
        <div v-if="clusteringResult?.feature_distribution_plot" class="feature-distribution card">
          <h3 class="section-title">
            <el-icon><DataLine /></el-icon>
            特征分布分析
          </h3>
          
          <div class="plot-container">
            <img :src="clusteringResult.feature_distribution_plot" alt="特征分布" class="result-image" />
          </div>
        </div>
        
        <!-- 雷达图 -->
        <div v-if="clusteringResult?.radar_plot" class="radar-chart card">
          <h3 class="section-title">
            <el-icon><PieChart /></el-icon>
            聚类特征雷达图
          </h3>
          
          <div class="plot-container">
            <img :src="clusteringResult.radar_plot" alt="雷达图" class="result-image" />
          </div>
        </div>
        
        <!-- 下载区域 -->
        <div v-if="clusteringResult || optimalKResult || compareKResult" class="download-section card">
          <h3 class="section-title">
            <el-icon><Download /></el-icon>
            结果下载与操作
          </h3>
          
          <el-row :gutter="20">
            <el-col :span="8" v-if="clusteringResult">
              <el-button
                type="success"
                @click="downloadClusteredData"
                style="width: 100%"
              >
                <el-icon><Download /></el-icon>
                下载聚类结果数据
              </el-button>
            </el-col>
            <el-col :span="8" v-if="optimalKResult">
              <el-button
                type="warning"
                @click="applyOptimalK"
                style="width: 100%"
              >
                <el-icon><Check /></el-icon>
                应用推荐K值
              </el-button>
            </el-col>
            <el-col :span="8" v-if="compareKResult">
              <el-button
                type="info"
                @click="applyBestK"
                style="width: 100%"
              >
                <el-icon><Check /></el-icon>
                应用最佳K值
              </el-button>
            </el-col>
            <el-col :span="8">
              <el-button
                type="primary"
                @click="refreshResultFiles"
                style="width: 100%"
              >
                <el-icon><Refresh /></el-icon>
                刷新文件列表
              </el-button>
            </el-col>
            <el-col :span="8">
              <el-button
                type="danger"
                @click="clearResults"
                style="width: 100%"
              >
                <el-icon><Delete /></el-icon>
                清空结果
              </el-button>
            </el-col>
          </el-row>
        </div>
        
        <!-- 空状态 -->
        <div v-if="!clusteringResult && !optimalKResult && !compareKResult && !analysisLoading && !optimalKLoading && !compareKLoading" class="empty-state card">
          <el-empty 
            :description="resultFiles.length === 0 ? '暂无数据文件，请先进行事件提取' : '请选择数据文件并开始聚类分析'"
            :image-size="120"
          >
            <template v-if="resultFiles.length === 0">
              <el-button type="primary" @click="router.push('/extraction')">
                <el-icon><Upload /></el-icon>
                前往事件提取
              </el-button>
              <el-button type="default" @click="refreshResultFiles" style="margin-left: 10px;">
                <el-icon><Refresh /></el-icon>
                刷新文件列表
              </el-button>
            </template>
            <template v-else>
              <el-button type="primary" @click="refreshResultFiles">
                <el-icon><Refresh /></el-icon>
                刷新文件列表
              </el-button>
            </template>
          </el-empty>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  PieChart,
  Setting,
  TrendCharts,
  DataLine,
  Download,
  Refresh,
  Cpu,
  Check,
  Delete,
  Upload
} from '@element-plus/icons-vue'
import { clusteringAPI, downloadAPI } from '@/api'

// 路由实例
const router = useRouter()

// 响应式数据
const resultFiles = ref([])
const filesLoading = ref(false)
const analysisLoading = ref(false)
const optimalKLoading = ref(false)
const compareKLoading = ref(false)
const clusteringResult = ref(null)
const optimalKResult = ref(null)
const compareKResult = ref(null)

// 参数配置
const params = reactive({
  selectedFile: '',
  algorithm: 'kmeans',
  k: 3,
  auto_k: false,
  auto_k_min: 2,
  auto_k_max: 10,
  dbscan_eps: 0.5,
  dbscan_min_samples: 5,
  reduction_method: 'pca',
  use_weights: false,
  amplitude_weight: 1.5,
  duration_weight: 1.5,
  fwhm_weight: 1.0,
  rise_time_weight: 0.6,
  decay_time_weight: 0.6,
  auc_weight: 1.0,
  snr_weight: 1.0
})

// 生命周期
onMounted(() => {
  loadResultFiles()
})

// 加载结果文件列表
const loadResultFiles = async () => {
  filesLoading.value = true
  try {
    const response = await clusteringAPI.getResultFiles()
    if (response.success) {
      resultFiles.value = response.files || []
      if (resultFiles.value.length === 0) {
        ElMessage.info('暂无可用的数据文件，请先进行事件提取')
      }
    } else {
      ElMessage.error('获取文件列表失败')
      resultFiles.value = []
    }
  } catch (error) {
    console.error('加载文件列表失败:', error)
    ElMessage.error('无法连接到服务器，请检查后端服务是否启动')
    resultFiles.value = []
  } finally {
    filesLoading.value = false
  }
}

// 刷新文件列表
const refreshResultFiles = async () => {
  await loadResultFiles()
}

// 文件选择变化
const handleFileChange = () => {
  clearResults()
}

// 算法变化
const handleAlgorithmChange = () => {
  clearResults()
}

// 自动K值变化
const handleAutoKChange = () => {
  if (params.auto_k) {
    optimalKResult.value = null
    compareKResult.value = null
  }
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
      algorithm: params.algorithm,
      reduction_method: params.reduction_method,
      auto_k: params.auto_k
    }
    
    // 添加K值相关参数
    if (params.algorithm === 'kmeans') {
      if (!params.auto_k) {
        requestData.k = params.k
      }
      requestData.auto_k_range = `${params.auto_k_min},${params.auto_k_max}`
    } else if (params.algorithm === 'dbscan') {
      requestData.dbscan_eps = params.dbscan_eps
      requestData.dbscan_min_samples = params.dbscan_min_samples
    }
    
    // 如果启用权重，添加权重参数
    if (params.use_weights) {
      requestData.feature_weights = {
        amplitude: params.amplitude_weight,
        duration: params.duration_weight,
        fwhm: params.fwhm_weight,
        rise_time: params.rise_time_weight,
        decay_time: params.decay_time_weight,
        auc: params.auc_weight,
        snr: params.snr_weight
      }
    }
    
    const response = await clusteringAPI.analyze(requestData)
    if (response.success) {
      clusteringResult.value = response
      ElMessage.success('聚类分析完成')
    }
  } catch (error) {
    console.error('聚类分析失败:', error)
    ElMessage.error('聚类分析失败，请检查数据文件和参数设置')
  } finally {
    analysisLoading.value = false
  }
}

// 寻找最佳K值
const findOptimalK = async () => {
  if (!params.selectedFile) {
    ElMessage.warning('请选择数据文件')
    return
  }
  
  optimalKLoading.value = true
  try {
    const requestData = {
      filename: params.selectedFile,
      max_k: params.auto_k_max
    }
    
    // 如果启用权重，添加权重参数
    if (params.use_weights) {
      requestData.feature_weights = {
        amplitude: params.amplitude_weight,
        duration: params.duration_weight,
        fwhm: params.fwhm_weight,
        rise_time: params.rise_time_weight,
        decay_time: params.decay_time_weight,
        auc: params.auc_weight,
        snr: params.snr_weight
      }
    }
    
    const response = await clusteringAPI.findOptimalK(requestData)
    if (response.success) {
      optimalKResult.value = response
      ElMessage.success(`推荐使用K=${response.optimal_k}进行聚类`)
    }
  } catch (error) {
    console.error('最佳K值分析失败:', error)
    ElMessage.error('最佳K值分析失败')
  } finally {
    optimalKLoading.value = false
  }
}

// 比较K值效果
const compareKValues = async () => {
  if (!params.selectedFile) {
    ElMessage.warning('请选择数据文件')
    return
  }
  
  compareKLoading.value = true
  try {
    const k_values = Array.from(
      { length: params.auto_k_max - params.auto_k_min + 1 }, 
      (_, i) => params.auto_k_min + i
    ).join(',')
    
    const requestData = {
      filename: params.selectedFile,
      k_values: k_values
    }
    
    // 如果启用权重，添加权重参数
    if (params.use_weights) {
      requestData.feature_weights = {
        amplitude: params.amplitude_weight,
        duration: params.duration_weight,
        fwhm: params.fwhm_weight,
        rise_time: params.rise_time_weight,
        decay_time: params.decay_time_weight,
        auc: params.auc_weight,
        snr: params.snr_weight
      }
    }
    
    const response = await clusteringAPI.compareK(requestData)
    if (response.success) {
      compareKResult.value = response
      ElMessage.success(`K值比较完成，推荐使用K=${response.best_k}`)
    }
  } catch (error) {
    console.error('K值比较失败:', error)
    ElMessage.error('K值比较失败')
  } finally {
    compareKLoading.value = false
  }
}

// 应用推荐的K值
const applyOptimalK = () => {
  if (optimalKResult.value) {
    params.k = optimalKResult.value.optimal_k
    params.auto_k = false
    ElMessage.success(`已应用推荐K值: ${optimalKResult.value.optimal_k}`)
  }
}

// 应用最佳K值
const applyBestK = () => {
  if (compareKResult.value) {
    params.k = compareKResult.value.best_k
    params.auto_k = false
    ElMessage.success(`已应用最佳K值: ${compareKResult.value.best_k}`)
  }
}

// 清空结果
const clearResults = () => {
  clusteringResult.value = null
  optimalKResult.value = null
  compareKResult.value = null
}

// 下载聚类结果
const downloadClusteredData = async () => {
  if (!clusteringResult.value?.output_file) {
    ElMessage.error('没有可下载的聚类结果文件')
    return
  }
  
  try {
    const response = await downloadAPI.downloadFile(clusteringResult.value.output_file)
    
    // 创建下载链接
    const blob = new Blob([response], {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = clusteringResult.value.output_file
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
    
    ElMessage.success('文件下载成功')
  } catch (error) {
    console.error('下载失败:', error)
    ElMessage.error('文件下载失败')
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
  padding: 20px;
}

.page-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
  color: #2c3e50;
  font-size: 24px;
  font-weight: 600;
}

.info-alert {
  margin-bottom: 20px;
}

.card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  border: 1px solid #ebeef5;
  margin-bottom: 20px;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 20px;
  color: #303133;
  font-size: 18px;
  font-weight: bold;
}

.params-panel {
  position: sticky;
  top: 20px;
  height: fit-content;
}

.result-image {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.param-help {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}

/* 统一的表单项样式 */
:deep(.el-form) {
  margin-bottom: 0;
}

:deep(.el-form-item) {
  margin-bottom: 18px;
}

:deep(.el-form-item__label) {
  font-weight: 500;
  color: #606266;
}

:deep(.el-input-number) {
  width: 100%;
}

:deep(.el-select) {
  width: 100%;
}

:deep(.el-input) {
  width: 100%;
}

:deep(.el-slider) {
  width: 100%;
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

/* 新增样式 */
.analysis-summary {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 20px;
  height: 100%;
}

.summary-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 1px solid #ebeef5;
}

.summary-item:last-child {
  border-bottom: none;
  margin-bottom: 0;
}

.summary-label {
  font-weight: 500;
  color: #606266;
}

.summary-value {
  font-weight: bold;
  color: #2c3e50;
}

.optimal-k {
  color: #67c23a;
  font-size: 18px;
}

.k-scores {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #ebeef5;
}

.score-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 14px;
}

.best-score {
  color: #67c23a;
  font-weight: bold;
}

.optimal-k-result, .compare-k-result {
  margin-bottom: 20px;
}

.radar-chart {
  margin-bottom: 20px;
}

.no-files-tip {
  margin-top: 10px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .cluster-card {
    margin-bottom: 15px;
  }
  
  .feature-plot {
    margin-bottom: 30px;
  }
  
  .analysis-summary {
    margin-bottom: 20px;
  }
}
</style>