<template>
  <div class="principal-neuron">
    <!-- 页面标题 -->
    <div class="page-header card">
      <h1 class="page-title">
        <el-icon><DataAnalysis /></el-icon>
        主神经元分析
      </h1>
      <p class="page-description">
        分析神经元活动的效应大小，识别关键神经元，生成活动图和动画可视化
      </p>
    </div>

    <!-- 文件上传区域 -->
    <div class="upload-section card">
      <h2 class="section-title">
        <el-icon><Upload /></el-icon>
        数据上传
      </h2>
      
      <el-upload
        ref="uploadRef"
        class="upload-demo"
        drag
        :auto-upload="false"
        :on-change="handleFileChange"
        :file-list="fileList"
        accept=".csv,.xlsx,.xls"
        multiple
      >
        <el-icon class="el-icon--upload"><upload-filled /></el-icon>
        <div class="el-upload__text">
          拖拽文件到此处，或<em>点击上传</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            支持 CSV、Excel 格式的神经元效应大小数据文件
          </div>
        </template>
      </el-upload>
    </div>

    <!-- 分析参数配置 -->
    <div class="config-section card" v-if="fileList.length > 0">
      <h2 class="section-title">
        <el-icon><Setting /></el-icon>
        分析参数
      </h2>
      
      <el-form :model="analysisConfig" label-width="120px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="效应大小阈值">
              <el-input-number
                v-model="analysisConfig.effectSizeThreshold"
                :min="0"
                :max="5"
                :step="0.1"
                :precision="2"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="显著性阈值">
              <el-input-number
                v-model="analysisConfig.significanceThreshold"
                :min="0"
                :max="1"
                :step="0.01"
                :precision="3"
              />
            </el-form-item>
          </el-col>
        </el-row>
        
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item label="动画帧数">
              <el-input-number
                v-model="analysisConfig.animationFrames"
                :min="10"
                :max="200"
                :step="10"
              />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="动画速度(ms)">
              <el-input-number
                v-model="analysisConfig.animationSpeed"
                :min="50"
                :max="2000"
                :step="50"
              />
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
    </div>

    <!-- 分析操作按钮 -->
    <div class="action-section card" v-if="fileList.length > 0">
      <el-button-group>
        <el-button 
          type="primary" 
          @click="runEffectSizeAnalysis"
          :loading="loading.effectSize"
          :disabled="isAnyLoading"
        >
          <el-icon><DataAnalysis /></el-icon>
          效应大小分析
        </el-button>
        
        <el-button 
          type="success" 
          @click="generateActivityPlot"
          :loading="loading.activityPlot"
          :disabled="isAnyLoading"
        >
          <el-icon><PictureRounded /></el-icon>
          生成活动图
        </el-button>
        
        <el-button 
          type="warning" 
          @click="generateAnimation"
          :loading="loading.animation"
          :disabled="isAnyLoading"
        >
          <el-icon><VideoPlay /></el-icon>
          生成动画
        </el-button>
        
        <el-button 
          type="info" 
          @click="analyzeSharedNeurons"
          :loading="loading.sharedNeurons"
          :disabled="isAnyLoading"
        >
          <el-icon><Connection /></el-icon>
          共享神经元分析
        </el-button>
      </el-button-group>
    </div>

    <!-- 结果展示区域 -->
    <div class="results-section" v-if="hasResults">
      <!-- 效应大小分析结果 -->
      <div class="result-card card" v-if="results.effectSize">
        <h3 class="result-title">
          <el-icon><DataAnalysis /></el-icon>
          效应大小分析结果
        </h3>
        <div class="result-content">
          <el-descriptions :column="2" border>
            <el-descriptions-item label="总神经元数量">{{ results.effectSize.total_neurons }}</el-descriptions-item>
            <el-descriptions-item label="显著神经元数量">{{ results.effectSize.significant_neurons }}</el-descriptions-item>
            <el-descriptions-item label="平均效应大小">{{ results.effectSize.mean_effect_size?.toFixed(4) }}</el-descriptions-item>
            <el-descriptions-item label="最大效应大小">{{ results.effectSize.max_effect_size?.toFixed(4) }}</el-descriptions-item>
          </el-descriptions>
          
          <div class="download-section" v-if="results.effectSize.csv_path">
            <el-button type="primary" @click="downloadFile(results.effectSize.csv_path)">
              <el-icon><Download /></el-icon>
              下载分析结果
            </el-button>
          </div>
        </div>
      </div>

      <!-- 活动图结果 -->
      <div class="result-card card" v-if="results.activityPlot">
        <h3 class="result-title">
          <el-icon><PictureRounded /></el-icon>
          神经元活动图
        </h3>
        <div class="result-content">
          <div class="image-container" v-if="results.activityPlot.plot_path">
            <img :src="getImageUrl(results.activityPlot.plot_path)" alt="神经元活动图" class="result-image" />
          </div>
          <div class="download-section">
            <el-button type="primary" @click="downloadFile(results.activityPlot.plot_path)">
              <el-icon><Download /></el-icon>
              下载活动图
            </el-button>
          </div>
        </div>
      </div>

      <!-- 动画结果 -->
      <div class="result-card card" v-if="results.animation">
        <h3 class="result-title">
          <el-icon><VideoPlay /></el-icon>
          神经元活动动画
        </h3>
        <div class="result-content">
          <div class="animation-info">
            <el-descriptions :column="2" border>
              <el-descriptions-item label="动画帧数">{{ results.animation.frames }}</el-descriptions-item>
              <el-descriptions-item label="动画时长">{{ results.animation.duration }}秒</el-descriptions-item>
            </el-descriptions>
          </div>
          <div class="download-section">
            <el-button type="primary" @click="downloadFile(results.animation.animation_path)">
              <el-icon><Download /></el-icon>
              下载动画文件
            </el-button>
          </div>
        </div>
      </div>

      <!-- 共享神经元分析结果 -->
      <div class="result-card card" v-if="results.sharedNeurons">
        <h3 class="result-title">
          <el-icon><Connection /></el-icon>
          共享神经元分析
        </h3>
        <div class="result-content">
          <el-descriptions :column="2" border>
            <el-descriptions-item label="共享神经元数量">{{ results.sharedNeurons.shared_count }}</el-descriptions-item>
            <el-descriptions-item label="总比较对数">{{ results.sharedNeurons.total_comparisons }}</el-descriptions-item>
          </el-descriptions>
          
          <div class="image-container" v-if="results.sharedNeurons.plot_path">
            <img :src="getImageUrl(results.sharedNeurons.plot_path)" alt="共享神经元图" class="result-image" />
          </div>
          
          <div class="download-section">
            <el-button type="primary" @click="downloadFile(results.sharedNeurons.plot_path)">
              <el-icon><Download /></el-icon>
              下载共享神经元图
            </el-button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  DataAnalysis,
  Upload,
  Setting,
  PictureRounded,
  VideoPlay,
  Connection,
  Download,
  UploadFilled
} from '@element-plus/icons-vue'
import api from '../api'

export default {
  name: 'PrincipalNeuron',
  components: {
    DataAnalysis,
    Upload,
    Setting,
    PictureRounded,
    VideoPlay,
    Connection,
    Download,
    UploadFilled
  },
  setup() {
    const fileList = ref([])
    const uploadRef = ref(null)
    
    const analysisConfig = reactive({
      effectSizeThreshold: 0.5,
      significanceThreshold: 0.05,
      animationFrames: 50,
      animationSpeed: 200
    })
    
    const loading = reactive({
      effectSize: false,
      activityPlot: false,
      animation: false,
      sharedNeurons: false
    })
    
    const results = reactive({
      effectSize: null,
      activityPlot: null,
      animation: null,
      sharedNeurons: null
    })
    
    const isAnyLoading = computed(() => {
      return Object.values(loading).some(val => val)
    })
    
    const hasResults = computed(() => {
      return Object.values(results).some(val => val !== null)
    })
    
    const handleFileChange = (file, newFileList) => {
      // 验证文件类型
      const allowedTypes = ['.csv', '.xlsx', '.xls']
      const fileExtension = '.' + file.name.split('.').pop().toLowerCase()
      
      if (!allowedTypes.includes(fileExtension)) {
        ElMessage.error('只支持 CSV 和 Excel 格式的文件')
        return false
      }
      
      // 验证文件大小 (50MB)
      if (file.size > 50 * 1024 * 1024) {
        ElMessage.error('文件大小不能超过 50MB')
        return false
      }
      
      // 更新文件列表
      fileList.value = newFileList
      ElMessage.success(`已添加文件: ${file.name}`)
    }
    
    const runEffectSizeAnalysis = async () => {
      if (fileList.value.length === 0) {
        ElMessage.warning('请先上传数据文件')
        return
      }
      
      loading.effectSize = true
      
      try {
        const formData = new FormData()
        fileList.value.forEach(file => {
          formData.append('files', file.raw)
        })
        formData.append('effect_size_threshold', analysisConfig.effectSizeThreshold)
        formData.append('significance_threshold', analysisConfig.significanceThreshold)
        
        const response = await api.post('/principal-neuron/effect-size', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        
        results.effectSize = response.data
        ElMessage.success('效应大小分析完成')
      } catch (error) {
        console.error('效应大小分析失败:', error)
        ElMessage.error(error.response?.data?.detail || '效应大小分析失败')
      } finally {
        loading.effectSize = false
      }
    }
    
    const generateActivityPlot = async () => {
      if (fileList.value.length === 0) {
        ElMessage.warning('请先上传数据文件')
        return
      }
      
      loading.activityPlot = true
      
      try {
        const formData = new FormData()
        fileList.value.forEach(file => {
          formData.append('files', file.raw)
        })
        formData.append('effect_size_threshold', analysisConfig.effectSizeThreshold)
        
        const response = await api.post('/principal-neuron/activity-plot', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        
        results.activityPlot = response.data
        ElMessage.success('活动图生成完成')
      } catch (error) {
        console.error('活动图生成失败:', error)
        ElMessage.error(error.response?.data?.detail || '活动图生成失败')
      } finally {
        loading.activityPlot = false
      }
    }
    
    const generateAnimation = async () => {
      if (fileList.value.length === 0) {
        ElMessage.warning('请先上传数据文件')
        return
      }
      
      loading.animation = true
      
      try {
        const formData = new FormData()
        fileList.value.forEach(file => {
          formData.append('files', file.raw)
        })
        formData.append('frames', analysisConfig.animationFrames)
        formData.append('interval', analysisConfig.animationSpeed)
        
        const response = await api.post('/principal-neuron/animation', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        
        results.animation = response.data
        ElMessage.success('动画生成完成')
      } catch (error) {
        console.error('动画生成失败:', error)
        ElMessage.error(error.response?.data?.detail || '动画生成失败')
      } finally {
        loading.animation = false
      }
    }
    
    const analyzeSharedNeurons = async () => {
      if (fileList.value.length < 2) {
        ElMessage.warning('共享神经元分析需要至少上传2个文件')
        return
      }
      
      loading.sharedNeurons = true
      
      try {
        const formData = new FormData()
        fileList.value.forEach(file => {
          formData.append('files', file.raw)
        })
        formData.append('effect_size_threshold', analysisConfig.effectSizeThreshold)
        
        const response = await api.post('/principal-neuron/shared-neurons', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        
        results.sharedNeurons = response.data
        ElMessage.success('共享神经元分析完成')
      } catch (error) {
        console.error('共享神经元分析失败:', error)
        ElMessage.error(error.response?.data?.detail || '共享神经元分析失败')
      } finally {
        loading.sharedNeurons = false
      }
    }
    
    const downloadFile = async (filePath) => {
      try {
        const response = await api.get(`/download/${encodeURIComponent(filePath)}`, {
          responseType: 'blob'
        })
        
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', filePath.split('/').pop())
        document.body.appendChild(link)
        link.click()
        link.remove()
        window.URL.revokeObjectURL(url)
        
        ElMessage.success('文件下载成功')
      } catch (error) {
        console.error('文件下载失败:', error)
        ElMessage.error('文件下载失败')
      }
    }
    
    const getImageUrl = (imagePath) => {
      return `http://localhost:8000/download/${encodeURIComponent(imagePath)}`
    }
    
    return {
      fileList,
      uploadRef,
      analysisConfig,
      loading,
      results,
      isAnyLoading,
      hasResults,
      handleFileChange,
      runEffectSizeAnalysis,
      generateActivityPlot,
      generateAnimation,
      analyzeSharedNeurons,
      downloadFile,
      getImageUrl
    }
  }
}
</script>

<style scoped>
.principal-neuron {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  padding: 24px;
  margin-bottom: 20px;
}

.page-header {
  text-align: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.page-title {
  font-size: 28px;
  margin: 0 0 10px 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.page-description {
  font-size: 16px;
  margin: 0;
  opacity: 0.9;
}

.section-title {
  font-size: 20px;
  margin: 0 0 20px 0;
  display: flex;
  align-items: center;
  gap: 8px;
  color: #333;
}

.upload-demo {
  width: 100%;
}

.action-section {
  text-align: center;
}

.el-button-group .el-button {
  margin: 0 5px;
}

.results-section {
  margin-top: 20px;
}

.result-card {
  border-left: 4px solid #409eff;
}

.result-title {
  font-size: 18px;
  margin: 0 0 15px 0;
  display: flex;
  align-items: center;
  gap: 8px;
  color: #333;
}

.result-content {
  margin-top: 15px;
}

.image-container {
  text-align: center;
  margin: 20px 0;
}

.result-image {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.download-section {
  margin-top: 15px;
  text-align: center;
}

.animation-info {
  margin-bottom: 15px;
}

@media (max-width: 768px) {
  .principal-neuron {
    padding: 10px;
  }
  
  .card {
    padding: 16px;
  }
  
  .page-title {
    font-size: 24px;
  }
  
  .el-button-group .el-button {
    margin: 5px 2px;
  }
}
</style>