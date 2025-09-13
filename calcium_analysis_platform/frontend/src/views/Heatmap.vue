<template>
  <div class="heatmap">
    <h1 class="page-title">
      <el-icon><TrendCharts /></el-icon>
      热力图分析
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
        1. <strong>数据上传</strong>: 上传包含钙信号数据和行为标签的 Excel 文件。<br>
        2. <strong>行为分析</strong>: 选择起始和结束行为，设置分析参数。<br>
        3. <strong>热力图生成</strong>: 生成特定行为前后时间窗口的神经元活动热力图。
      </template>
    </el-alert>

    <el-row :gutter="20">
      <!-- 左侧参数面板 -->
      <el-col :xs="24" :sm="24" :md="8" :lg="6">
        <div class="params-panel card">
          <h3 class="section-title">
            <el-icon><Setting /></el-icon>
            分析参数
          </h3>
          
          <el-form :model="params" label-width="120px" size="small">
            <el-form-item label="起始行为">
              <el-select
                v-model="params.start_behavior"
                placeholder="选择起始行为"
                style="width: 100%"
              >
                <el-option
                  v-for="behavior in behaviorOptions"
                  :key="behavior.value"
                  :label="behavior.label"
                  :value="behavior.value"
                />
              </el-select>
              <div class="param-help">分析从此行为开始</div>
            </el-form-item>
            
            <el-form-item label="结束行为">
              <el-select
                v-model="params.end_behavior"
                placeholder="选择结束行为"
                style="width: 100%"
              >
                <el-option
                  v-for="behavior in behaviorOptions"
                  :key="behavior.value"
                  :label="behavior.label"
                  :value="behavior.value"
                />
              </el-select>
              <div class="param-help">分析到此行为结束</div>
            </el-form-item>
            
            <el-form-item label="行为前时间">
              <el-input-number
                v-model="params.pre_behavior_time"
                :min="1"
                :max="60"
                :step="1"
                :precision="1"
                style="width: 100%"
              />
              <div class="param-help">单位：秒</div>
            </el-form-item>
            
            <el-form-item label="采样频率">
              <el-input-number
                v-model="params.sampling_rate"
                :min="0.1"
                :max="100"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
              <div class="param-help">单位：Hz</div>
            </el-form-item>
            
            <el-form-item label="最小持续时间">
              <el-input-number
                v-model="params.min_behavior_duration"
                :min="0.1"
                :max="10"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
              <div class="param-help">单位：秒</div>
            </el-form-item>
            
            <el-form-item label="神经元排序">
              <el-select
                v-model="params.sorting_method"
                style="width: 100%"
              >
                <el-option label="全局排序" value="global" />
                <el-option label="局部排序" value="local" />
                <el-option label="首图排序" value="first" />
                <el-option label="自定义排序" value="custom" />
              </el-select>
              <div class="param-help">神经元在热力图中的排列方式</div>
            </el-form-item>
            
            <!-- 整体热力图参数 -->
            <el-divider content-position="left">整体热力图参数</el-divider>
            
            <el-form-item label="时间范围开始">
              <el-input-number
                v-model="overallParams.stamp_min"
                :min="0"
                :step="1"
                :precision="2"
                style="width: 100%"
                placeholder="留空表示从头开始"
              />
              <div class="param-help">开始时间戳（秒）</div>
            </el-form-item>
            
            <el-form-item label="时间范围结束">
              <el-input-number
                v-model="overallParams.stamp_max"
                :min="0"
                :step="1"
                :precision="2"
                style="width: 100%"
                placeholder="留空表示到结尾"
              />
              <div class="param-help">结束时间戳（秒）</div>
            </el-form-item>
            
            <el-form-item label="排序方式">
              <el-select
                v-model="overallParams.sort_method"
                style="width: 100%"
              >
                <el-option label="按峰值时间排序" value="peak" />
                <el-option label="按钙波时间排序" value="calcium_wave" />
              </el-select>
              <div class="param-help">神经元排序算法</div>
            </el-form-item>
            
            <el-form-item label="钙波阈值">
              <el-input-number
                v-model="overallParams.calcium_wave_threshold"
                :min="0.1"
                :max="5.0"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
              <div class="param-help">标准差的倍数</div>
            </el-form-item>
          </el-form>
        </div>
      </el-col>
      
      <!-- 右侧主要内容区域 -->
      <el-col :xs="24" :sm="24" :md="16" :lg="18">
        <!-- 文件上传区域 -->
        <div class="upload-section card">
          <h3 class="section-title">
            <el-icon><Upload /></el-icon>
            数据文件上传
          </h3>
          
          <el-upload
            ref="uploadRef"
            :on-change="handleFileChange"
            :on-remove="handleFileRemove"
            :before-upload="() => false"
            accept=".xlsx,.xls"
            drag
            :limit="1"
          >
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              将文件拖到此处，或<em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                支持 Excel 文件格式 (.xlsx, .xls)，文件应包含钙信号数据和行为标签
              </div>
            </template>
          </el-upload>
        </div>
        
        <!-- 行为检测结果 -->
        <div v-if="behaviorEvents.length > 0" class="behavior-events card">
          <h3 class="section-title">
            <el-icon><View /></el-icon>
            检测到的行为事件
          </h3>
          
          <el-table :data="behaviorEvents" style="width: 100%" size="small">
            <el-table-column prop="index" label="序号" width="60" />
            <el-table-column prop="start_behavior" label="起始行为" />
            <el-table-column prop="end_behavior" label="结束行为" />
            <el-table-column prop="start_time" label="开始时间(s)" width="100" />
            <el-table-column prop="end_time" label="结束时间(s)" width="100" />
            <el-table-column prop="duration" label="持续时间(s)" width="100" />
          </el-table>
        </div>
        
        <!-- 分析控制区域 -->
        <div class="analysis-section card">
          <h3 class="section-title">
            <el-icon><DataAnalysis /></el-icon>
            热力图分析
          </h3>
          
          <div class="analysis-controls">
            <el-button
              type="primary"
              :loading="analysisLoading"
              :disabled="!hasFile || !params.start_behavior || !params.end_behavior"
              @click="startAnalysis"
            >
              <el-icon><VideoPlay /></el-icon>
              行为热力图分析
            </el-button>
            
            <el-button
              type="warning"
              :loading="overallAnalysisLoading"
              :disabled="!hasFile"
              @click="startOverallAnalysis"
            >
              <el-icon><TrendCharts /></el-icon>
              整体热力图分析
            </el-button>
            
            <el-button
              type="info"
              @click="debugButtonState"
              size="small"
            >
              调试按钮状态
            </el-button>
            
            <el-button
              v-if="analysisResult"
              type="success"
              @click="downloadResult"
            >
              <el-icon><Download /></el-icon>
              下载结果
            </el-button>
          </div>
        </div>
        
        <!-- 分析结果展示 -->
        <div v-if="analysisResult" class="result-section card">
          <h3 class="section-title">
            <el-icon><PictureRounded /></el-icon>
            分析结果
          </h3>
          
          <div class="result-content">
            <div class="result-summary">
              <el-descriptions :column="3" border>
                <el-descriptions-item label="分析文件">{{ analysisResult.filename }}</el-descriptions-item>
                <el-descriptions-item label="行为配对数">{{ analysisResult.behavior_pairs_count }}</el-descriptions-item>
                <el-descriptions-item label="神经元数量">{{ analysisResult.neuron_count }}</el-descriptions-item>
                <el-descriptions-item label="起始行为">{{ analysisResult.start_behavior }}</el-descriptions-item>
                <el-descriptions-item label="结束行为">{{ analysisResult.end_behavior }}</el-descriptions-item>
                <el-descriptions-item label="分析状态">{{ analysisResult.status }}</el-descriptions-item>
              </el-descriptions>
            </div>
            
            <div v-if="analysisResult.heatmap_images" class="heatmap-gallery">
              <h4>生成的热力图</h4>
              <el-row :gutter="10">
                <el-col
                  v-for="(image, index) in analysisResult.heatmap_images"
                  :key="index"
                  :xs="24" :sm="12" :md="8" :lg="6"
                >
                  <div class="heatmap-item">
                    <img 
                      :src="image.url" 
                      :alt="image.title" 
                      class="heatmap-image" 
                      @click="openHeatmapModal(image, index)"
                    />
                    <div class="heatmap-title">{{ image.title }}</div>
                    <div class="heatmap-behavior-labels">
                      <el-tag 
                        v-if="!image.title.includes('平均')"
                        type="primary" 
                        size="small"
                        class="behavior-tag"
                      >
                        {{ analysisResult.start_behavior }}
                      </el-tag>
                      <el-icon v-if="!image.title.includes('平均')" class="arrow-icon"><ArrowRight /></el-icon>
                      <el-tag 
                        v-if="!image.title.includes('平均')"
                        type="success" 
                        size="small"
                        class="behavior-tag"
                      >
                        {{ analysisResult.end_behavior }}
                      </el-tag>
                      <el-tag 
                        v-else
                        type="warning" 
                        size="small"
                        class="behavior-tag"
                      >
                        平均热力图
                      </el-tag>
                    </div>
                  </div>
                </el-col>
              </el-row>
            </div>
          </div>
        </div>
        
        <!-- 整体热力图结果展示 -->
        <div v-if="overallAnalysisResult" class="result-section card">
          <h3 class="section-title">
            <el-icon><TrendCharts /></el-icon>
            整体热力图分析结果
          </h3>
          
          <div class="result-content">
            <div class="result-summary">
               <el-descriptions :column="3" border>
                 <el-descriptions-item label="分析文件">{{ overallAnalysisResult.filename }}</el-descriptions-item>
                 <el-descriptions-item label="神经元数量">{{ overallAnalysisResult.analysis_info?.neuron_count || 0 }}</el-descriptions-item>
                 <el-descriptions-item label="排序方式">{{ overallAnalysisResult.config?.sort_method || overallAnalysisResult.analysis_info?.sort_method }}</el-descriptions-item>
                 <el-descriptions-item label="钙波阈值">{{ overallAnalysisResult.config?.calcium_wave_threshold }}</el-descriptions-item>
                 <el-descriptions-item label="时间范围">{{ formatTimeRange(overallAnalysisResult.analysis_info?.time_range) }}</el-descriptions-item>
                 <el-descriptions-item label="分析状态">{{ overallAnalysisResult.success ? '成功' : '失败' }}</el-descriptions-item>
               </el-descriptions>
             </div>
            
            <div v-if="overallAnalysisResult.heatmap_image" class="overall-heatmap-display">
              <h4>整体热力图</h4>
              <div class="overall-heatmap-container">
                <img 
                  :src="overallAnalysisResult.heatmap_image" 
                  alt="整体热力图" 
                  class="overall-heatmap-image" 
                  @click="openOverallHeatmapModal"
                />
                <div class="overall-heatmap-info">
                  <el-tag type="info" size="small">点击查看大图</el-tag>
                </div>
              </div>
            </div>
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- 热力图放大模态框 -->
    <el-dialog
      v-model="heatmapModalVisible"
      :title="selectedHeatmap?.title || '热力图详情'"
      width="80%"
      center
      :before-close="closeHeatmapModal"
    >
      <div class="modal-heatmap-container">
        <img 
          v-if="selectedHeatmap"
          :src="selectedHeatmap.url" 
          :alt="selectedHeatmap.title" 
          class="modal-heatmap-image"
        />
        <div class="modal-behavior-info">
          <el-descriptions :column="2" border>
            <el-descriptions-item label="热力图类型">
              {{ selectedHeatmap?.title || '' }}
            </el-descriptions-item>
            <el-descriptions-item label="行为配对" v-if="!selectedHeatmap?.title.includes('平均')">
              <el-tag type="primary" size="small">{{ analysisResult?.start_behavior }}</el-tag>
              <el-icon class="mx-2"><ArrowRight /></el-icon>
              <el-tag type="success" size="small">{{ analysisResult?.end_behavior }}</el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="神经元数量">
              {{ analysisResult?.neuron_count || 0 }}
            </el-descriptions-item>
            <el-descriptions-item label="分析参数">
              前置时间: {{ params.pre_behavior_time }}s, 最小持续: {{ params.min_behavior_duration }}s
            </el-descriptions-item>
          </el-descriptions>
        </div>
      </div>
    </el-dialog>
    
    <!-- 整体热力图放大模态框 -->
    <el-dialog
      v-model="overallHeatmapModalVisible"
      title="整体热力图详情"
      width="90%"
      center
      :before-close="closeOverallHeatmapModal"
    >
      <div class="modal-heatmap-container">
        <img 
          v-if="overallAnalysisResult?.heatmap_image"
          :src="overallAnalysisResult.heatmap_image" 
          alt="整体热力图" 
          class="modal-heatmap-image"
        />
        <div class="modal-behavior-info">
          <el-descriptions :column="2" border>
             <el-descriptions-item label="热力图类型">
               整体热力图
             </el-descriptions-item>
             <el-descriptions-item label="排序方式">
               {{ overallAnalysisResult?.config?.sort_method || overallAnalysisResult?.analysis_info?.sort_method || '' }}
             </el-descriptions-item>
             <el-descriptions-item label="神经元数量">
               {{ overallAnalysisResult?.analysis_info?.neuron_count || 0 }}
             </el-descriptions-item>
             <el-descriptions-item label="钙波阈值">
               {{ overallAnalysisResult?.config?.calcium_wave_threshold || 0 }}
             </el-descriptions-item>
             <el-descriptions-item label="时间范围">
               {{ formatTimeRange(overallAnalysisResult?.analysis_info?.time_range) }}
             </el-descriptions-item>
             <el-descriptions-item label="分析参数">
               最小突出度: {{ overallAnalysisResult?.config?.min_prominence || overallParams.min_prominence }}, 最小上升率: {{ overallAnalysisResult?.config?.min_rise_rate || overallParams.min_rise_rate }}
             </el-descriptions-item>
           </el-descriptions>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, watch, computed } from 'vue'
import {
  TrendCharts,
  Setting,
  Upload,
  UploadFilled,
  View,
  DataAnalysis,
  VideoPlay,
  Download,
  PictureRounded,
  ArrowRight
} from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'

// 响应式数据
const uploadRef = ref()
const fileList = ref([])
const uploadedFile = ref(null) // 独立的文件状态
const hasFile = computed(() => uploadedFile.value !== null)
const analysisLoading = ref(false)
const behaviorEvents = ref([])
const analysisResult = ref(null)

// 整体热力图相关
const overallAnalysisLoading = ref(false)
const overallAnalysisResult = ref(null)

// 热力图模态框相关
const heatmapModalVisible = ref(false)
const selectedHeatmap = ref(null)
const selectedHeatmapIndex = ref(-1)

// 整体热力图模态框相关
const overallHeatmapModalVisible = ref(false)

// 行为选项（这些可以从上传的文件中动态获取）
const behaviorOptions = [
  { label: 'Explore', value: 'Explore' },
  { label: 'Water', value: 'Water' },
  { label: 'Smell-feed', value: 'Smell-feed' },
  { label: 'Eat-feed', value: 'Eat-feed' },
  { label: 'Get-feed', value: 'Get-feed' },
  { label: 'Groom', value: 'Groom' },
  { label: 'Smell-Get-seeds', value: 'Smell-Get-seeds' },
  { label: 'Get-seeds', value: 'Get-seeds' },
  { label: 'Crack-seeds-shells', value: 'Crack-seeds-shells' },
  { label: 'Eat-seed-kernels', value: 'Eat-seed-kernels' },
  { label: 'Smell-water', value: 'Smell-water' },
  { label: 'Find-seeds', value: 'Find-seeds' },
  { label: 'Grab-seeds', value: 'Grab-seeds' },
  { label: 'Explore-search-seeds', value: 'Explore-search-seeds' }
]

// 参数配置
const params = reactive({
  start_behavior: 'Crack-seeds-shells',
  end_behavior: 'Eat-seed-kernels',
  pre_behavior_time: 10.0,
  sampling_rate: 4.8,
  min_behavior_duration: 1.0,
  sorting_method: 'first'
})

// 整体热力图参数配置
const overallParams = reactive({
  stamp_min: null,
  stamp_max: null,
  sort_method: 'peak',
  calcium_wave_threshold: 1.5,
  min_prominence: 1.0,
  min_rise_rate: 0.1,
  max_fall_rate: 0.05
})

// 监听参数变化
watch(params, (newParams) => {
  console.log('参数变化:', newParams)
  console.log('按钮禁用条件检查:', {
    hasFile: hasFile.value,
    uploadedFile: uploadedFile.value,
    startBehavior: newParams.start_behavior,
    endBehavior: newParams.end_behavior,
    shouldDisable: !hasFile.value || !newParams.start_behavior || !newParams.end_behavior
  })
}, { deep: true })

// 页面加载时检查初始状态
onMounted(() => {
  console.log('页面加载完成，初始状态检查:', {
    hasFile: hasFile.value,
    uploadedFile: uploadedFile.value,
    startBehavior: params.start_behavior,
    endBehavior: params.end_behavior,
    behaviorOptions: behaviorOptions.length,
    shouldDisable: !hasFile.value || !params.start_behavior || !params.end_behavior
  })
})

// 文件上传处理
const handleFileChange = async (file, files) => {
  console.log('handleFileChange被调用:', { file, files, fileStatus: file?.status })
  console.log('当前uploadedFile:', uploadedFile.value)
  
  // 重置状态
  analysisResult.value = null
  behaviorEvents.value = []
  
  if (file) {
    console.log('文件状态:', file.status)
    console.log('设置uploadedFile为:', file)
    // 使用独立的文件状态，不依赖status
    uploadedFile.value = file
    fileList.value = [file] // 保持兼容性
    console.log('设置后uploadedFile:', uploadedFile.value)
    console.log('hasFile状态:', hasFile.value)
    
    // 检查按钮状态
    console.log('文件上传后按钮状态检查:', {
      hasFile: hasFile.value,
      startBehavior: params.start_behavior,
      endBehavior: params.end_behavior,
      shouldDisable: !hasFile.value || !params.start_behavior || !params.end_behavior
    })
    
    // 检测行为事件
    console.log('开始检测行为事件，hasFile:', hasFile.value)
    await detectBehaviorEvents(file)
    console.log('检测行为事件完成，hasFile:', hasFile.value)
  } else {
    console.log('清空文件状态，原因: 无有效文件')
    uploadedFile.value = null
    fileList.value = []
  }
}

// 文件移除处理
const handleFileRemove = (file, files) => {
  console.log('handleFileRemove被调用:', { file, files })
  console.log('files数组详情:', { length: files.length, isArray: Array.isArray(files), content: files })
  console.log('调用堆栈:', new Error().stack)
  
  // 完全禁用文件移除逻辑，避免意外清空状态
  console.log('禁用文件移除逻辑，保持所有状态不变')
  // 不执行任何清空操作
  
  console.log('文件移除后hasFile状态:', hasFile.value)
}

// 检测行为事件
const detectBehaviorEvents = async (file) => {
  try {
    console.log('开始检测行为事件，文件:', file.name)
    console.log('文件对象:', file)
    
    // 创建FormData对象
    const formData = new FormData()
    // 使用file.raw或file本身，取决于Element Plus的文件对象结构
    const fileToUpload = file.raw || file
    formData.append('file', fileToUpload)
    
    console.log('发送行为事件检测请求到后端...')
    
    // 调用后端API检测行为事件
    const response = await fetch('http://localhost:8000/api/behavior/detect', {
      method: 'POST',
      body: formData
    })
    
    console.log('行为事件检测API响应状态:', response.status)
    
    if (!response.ok) {
      const errorText = await response.text()
      console.error('行为事件检测API错误:', errorText)
      throw new Error(`API请求失败: ${response.status} - ${errorText}`)
    }
    
    const result = await response.json()
    console.log('行为事件检测结果:', result)
    
    behaviorEvents.value = result.behavior_events || []
    
    ElMessage.success(`检测到 ${behaviorEvents.value.length} 个行为事件配对`)
  } catch (error) {
    console.error('行为事件检测失败:', error)
    ElMessage.error('行为事件检测失败: ' + (error.message || '未知错误'))
  }
}

// 开始分析
const startAnalysis = async () => {
  if (fileList.value.length === 0) {
    ElMessage.warning('请先上传数据文件')
    return
  }
  
  if (!params.start_behavior || !params.end_behavior) {
    ElMessage.warning('请选择起始和结束行为')
    return
  }
  
  analysisLoading.value = true
  
  try {
      // 创建FormData对象
      const formData = new FormData()
      formData.append('file', fileList.value[0].raw)
      formData.append('start_behavior', params.start_behavior)
      formData.append('end_behavior', params.end_behavior)
      formData.append('pre_behavior_time', params.pre_behavior_time.toString())
      formData.append('min_duration', params.min_behavior_duration.toString())
      formData.append('sampling_rate', params.sampling_rate.toString())
      
      console.log('发送请求参数:', {
        file: fileList.value[0].raw.name,
        start_behavior: params.start_behavior,
        end_behavior: params.end_behavior,
        pre_behavior_time: params.pre_behavior_time,
        min_duration: params.min_behavior_duration,
        sampling_rate: params.sampling_rate
      })
      
      // 调用后端API
       const response = await fetch('http://localhost:8000/api/heatmap/analyze', {
          method: 'POST',
          body: formData
        })
      
      console.log('API响应状态:', response.status)
      
      if (!response.ok) {
        const errorData = await response.json()
        console.error('API错误响应:', errorData)
        throw new Error(errorData.detail || '分析失败')
      }
      
      const result = await response.json()
      console.log('API成功响应:', result)
      console.log('热力图数据:', result.heatmap_images)
      console.log('热力图数量:', result.heatmap_images ? result.heatmap_images.length : 0)
      
      analysisResult.value = result
      
      if (result.heatmap_images && result.heatmap_images.length > 0) {
        ElMessage.success(`热力图分析完成！生成了 ${result.heatmap_images.length} 张热力图`)
      } else {
        ElMessage.warning('热力图分析完成，但没有生成图像')
        console.warn('没有生成热力图图像，可能的原因：', {
          hasImages: !!result.heatmap_images,
          imageCount: result.heatmap_images ? result.heatmap_images.length : 0,
          behaviorPairs: result.behavior_pairs_count
        })
      }
  } catch (error) {
    console.error('分析失败:', error)
    ElMessage.error('分析失败: ' + (error.message || '未知错误'))
  } finally {
    analysisLoading.value = false
  }
}

// 开始整体热力图分析
const startOverallAnalysis = async () => {
  if (fileList.value.length === 0) {
    ElMessage.warning('请先上传数据文件')
    return
  }
  
  overallAnalysisLoading.value = true
  
  try {
    // 创建FormData对象
    const formData = new FormData()
    formData.append('file', fileList.value[0].raw)
    
    // 添加整体热力图参数
    if (overallParams.stamp_min !== null) {
      formData.append('stamp_min', overallParams.stamp_min.toString())
    }
    if (overallParams.stamp_max !== null) {
      formData.append('stamp_max', overallParams.stamp_max.toString())
    }
    formData.append('sort_method', overallParams.sort_method)
    formData.append('calcium_wave_threshold', overallParams.calcium_wave_threshold.toString())
    formData.append('min_prominence', overallParams.min_prominence.toString())
    formData.append('min_rise_rate', overallParams.min_rise_rate.toString())
    formData.append('max_fall_rate', overallParams.max_fall_rate.toString())
    
    console.log('发送整体热力图分析请求参数:', {
      file: fileList.value[0].raw.name,
      stamp_min: overallParams.stamp_min,
      stamp_max: overallParams.stamp_max,
      sort_method: overallParams.sort_method,
      calcium_wave_threshold: overallParams.calcium_wave_threshold,
      min_prominence: overallParams.min_prominence,
      min_rise_rate: overallParams.min_rise_rate,
      max_fall_rate: overallParams.max_fall_rate
    })
    
    // 调用后端API
    const response = await fetch('http://localhost:8000/api/heatmap/overall', {
      method: 'POST',
      body: formData
    })
    
    console.log('整体热力图API响应状态:', response.status)
    
    if (!response.ok) {
      const errorData = await response.json()
      console.error('整体热力图API错误响应:', errorData)
      throw new Error(errorData.detail || '整体热力图分析失败')
    }
    
    const result = await response.json()
    console.log('整体热力图API成功响应:', result)
    
    overallAnalysisResult.value = result
    
    if (result.heatmap_image) {
      ElMessage.success('整体热力图分析完成！')
    } else {
      ElMessage.warning('整体热力图分析完成，但没有生成图像')
    }
  } catch (error) {
    console.error('整体热力图分析失败:', error)
    ElMessage.error('整体热力图分析失败: ' + (error.message || '未知错误'))
  } finally {
    overallAnalysisLoading.value = false
  }
}

// 调试按钮状态
const debugButtonState = () => {
  console.log('=== 调试按钮状态 ===')
  console.log('uploadedFile:', uploadedFile.value)
  console.log('hasFile:', hasFile.value)
  console.log('fileList兼容性:', fileList.value)
  console.log('参数状态:', params)
  console.log('按钮禁用条件:', {
    noFile: !hasFile.value,
    noStartBehavior: !params.start_behavior,
    noEndBehavior: !params.end_behavior,
    shouldDisable: !hasFile.value || !params.start_behavior || !params.end_behavior
  })
  
  ElMessage.info(`文件状态: ${hasFile.value ? '已上传' : '未上传'}, 开始行为: ${params.start_behavior}, 结束行为: ${params.end_behavior}`)
}

// 下载结果
const downloadResult = () => {
  if (!analysisResult.value) {
    ElMessage.warning('没有可下载的结果')
    return
  }
  
  // 这里应该调用后端API下载结果文件
  ElMessage.info('下载功能开发中...')
}

// 打开热力图模态框
const openHeatmapModal = (heatmap, index) => {
  selectedHeatmap.value = heatmap
  selectedHeatmapIndex.value = index
  heatmapModalVisible.value = true
}

// 关闭热力图模态框
const closeHeatmapModal = () => {
  heatmapModalVisible.value = false
  selectedHeatmap.value = null
  selectedHeatmapIndex.value = -1
}

// 打开整体热力图模态框
const openOverallHeatmapModal = () => {
  overallHeatmapModalVisible.value = true
}

// 关闭整体热力图模态框
const closeOverallHeatmapModal = () => {
  overallHeatmapModalVisible.value = false
}

// 格式化时间范围
const formatTimeRange = (timeRange) => {
  if (!timeRange || !timeRange.min || !timeRange.max) {
    return '未知'
  }
  return `${timeRange.min.toFixed(2)} - ${timeRange.max.toFixed(2)}`
}
</script>

<style scoped>
.heatmap {
  max-width: 1400px;
  margin: 0 auto;
}

.info-alert {
  margin-bottom: 20px;
}

.params-panel {
  height: fit-content;
  position: sticky;
  top: 20px;
}

.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 20px;
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
}

.param-help {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}

.upload-section {
  margin-bottom: 20px;
}

.behavior-events {
  margin-bottom: 20px;
}

.analysis-section {
  margin-bottom: 20px;
}

.analysis-controls {
  display: flex;
  gap: 10px;
  align-items: center;
}

.result-section {
  margin-bottom: 20px;
}

.result-summary {
  margin-bottom: 20px;
}

.heatmap-gallery h4 {
  margin-bottom: 15px;
  color: #2c3e50;
}

.heatmap-item {
  text-align: center;
  margin-bottom: 15px;
}

.heatmap-image {
  width: 100%;
  max-width: 300px;
  height: auto;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.heatmap-image:hover {
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
  border-color: #409eff;
}

.heatmap-title {
  margin-top: 8px;
  font-size: 14px;
  color: #606266;
  font-weight: 500;
}

.heatmap-behavior-labels {
  margin-top: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
}

.behavior-tag {
  font-size: 12px;
}

.arrow-icon {
  color: #909399;
  font-size: 14px;
}

/* 模态框样式 */
.modal-heatmap-container {
  text-align: center;
}

.modal-heatmap-image {
  max-width: 100%;
  max-height: 70vh;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  margin-bottom: 20px;
}

.modal-behavior-info {
  margin-top: 20px;
}

.mx-2 {
  margin: 0 8px;
}

/* 整体热力图样式 */
.overall-heatmap-display h4 {
  margin-bottom: 15px;
  color: #2c3e50;
}

.overall-heatmap-container {
  text-align: center;
  margin-bottom: 20px;
}

.overall-heatmap-image {
  max-width: 100%;
  height: auto;
  border: 1px solid #dcdfe6;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.overall-heatmap-image:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  border-color: #409eff;
  transform: scale(1.02);
}

.overall-heatmap-info {
  margin-top: 10px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .params-panel {
    position: static;
    margin-bottom: 20px;
  }
  
  .analysis-controls {
    flex-direction: column;
    align-items: stretch;
  }
  
  .analysis-controls .el-button {
    width: 100%;
    margin-bottom: 10px;
  }
}
</style>