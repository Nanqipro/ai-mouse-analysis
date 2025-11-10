<template>
  <div class="extraction">
    <h1 class="page-title">
      <el-icon><DataAnalysis /></el-icon>
      钙事件提取
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
        1. <strong>参数调试</strong>: 上传一个文件，通过可视化预览找到最佳参数。<br>
        2. <strong>批量提取</strong>: 使用找到的参数处理所有上传的文件。<br>
        <strong>数据格式要求</strong>: Excel文件需包含神经元列（如n4, n5, n6等）和可选的behavior列，与element_extraction.py格式一致。
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
            <el-form-item label="采样频率 (Hz)">
              <el-input-number
                v-model="params.fs"
                :min="0.1"
                :max="100"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
              <div class="param-help">默认为 4.8Hz</div>
            </el-form-item>
            
            <el-form-item label="最小持续时间">
              <el-input-number
                v-model="params.min_duration_frames"
                :min="1"
                :max="100"
                style="width: 100%"
              />
              <div class="param-help">单位：帧</div>
            </el-form-item>
            
            <el-form-item label="最大持续时间">
              <el-input-number
                v-model="params.max_duration_frames"
                :min="50"
                :max="2000"
                :step="10"
                style="width: 100%"
              />
              <div class="param-help">单位：帧</div>
            </el-form-item>
            
            <el-form-item label="最小信噪比">
              <el-input-number
                v-model="params.min_snr"
                :min="1"
                :max="10"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
            </el-form-item>
            
            <el-form-item label="平滑窗口">
              <el-input-number
                v-model="params.smooth_window"
                :min="3"
                :max="101"
                :step="2"
                style="width: 100%"
              />
              <div class="param-help">单位：帧（奇数）</div>
            </el-form-item>
            
            <el-form-item label="峰值最小距离">
              <el-input-number
                v-model="params.peak_distance_frames"
                :min="1"
                :max="100"
                style="width: 100%"
              />
              <div class="param-help">单位：帧</div>
            </el-form-item>
            
            <el-form-item label="过滤强度">
              <el-input-number
                v-model="params.filter_strength"
                :min="0.5"
                :max="2"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
            </el-form-item>
          </el-form>
        </div>
      </el-col>
      
      <!-- 右侧主要内容 -->
      <el-col :xs="24" :sm="24" :md="16" :lg="18">
        <!-- 文件上传区域 -->
        <div class="upload-section card">
          <h3 class="section-title">
            <el-icon><Upload /></el-icon>
            文件上传
          </h3>
          
          <el-upload
            ref="uploadRef"
            class="upload-demo"
            drag
            :auto-upload="false"
            :multiple="true"
            accept=".xlsx,.xls"
            :on-change="handleFileChange"
            :file-list="fileList"
          >
            <el-icon class="el-icon--upload"><upload-filled /></el-icon>
            <div class="el-upload__text">
              将 Excel 文件拖拽到此处，或<em>点击上传</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                支持 .xlsx 和 .xls 格式，需包含神经元列（n4, n5, n6...）和可选的behavior列
              </div>
            </template>
          </el-upload>
        </div>
        
        <!-- 事件提取区域（合并自动和交互） -->
        <div v-if="fileList.length > 0" class="extraction-section card">
          <h3 class="section-title">
            <el-icon><TrendCharts /></el-icon>
            事件提取
          </h3>
          
          <div v-if="neuronColumns.length > 0" class="extraction-controls">
            <el-row :gutter="20" align="middle">
              <el-col :span="12">
                <el-select
                  v-model="selectedNeuronInteractive"
                  placeholder="选择神经元"
                  style="width: 100%"
                  @change="handleNeuronChange"
                >
                  <el-option
                    v-for="neuron in neuronColumns"
                    :key="neuron"
                    :label="neuron"
                    :value="neuron"
                  />
                </el-select>
              </el-col>
              <el-col :span="12">
                <el-button
                  type="primary"
                  :loading="interactiveLoading || autoDetectionLoading"
                  :disabled="!selectedNeuronInteractive"
                  @click="reloadExtraction"
                  style="width: 100%"
                >
                  <el-icon><TrendCharts /></el-icon>
                  重新提取事件
                </el-button>
              </el-col>
            </el-row>
          </div>
          
          <!-- 交互式图表 -->
          <div v-if="interactiveData" class="interactive-chart">
            <div class="chart-header">
              <h4>钙信号时序图</h4>
              <p class="chart-instruction">
                点击两次选择时间范围进行分析
              </p>
            </div>
            
            <div class="chart-instructions">
              <el-alert
                title="使用说明"
                type="info"
                :closable="false"
                show-icon
              >
                <template #default>
                  1. <strong>选择范围</strong>：点击选择起始时间点，再次点击选择结束时间点，然后点击"提取选定范围的事件"按钮<br/>
                  2. <strong>缩放</strong>：使用鼠标滚轮或工具栏进行缩放<br/>
                  3. <strong>坐标显示</strong>：鼠标靠近图表时显示横纵坐标
                </template>
              </el-alert>
            </div>
            
            <div ref="chartContainer" class="chart-container"></div>
            <div v-if="mouseCoords" class="mouse-coords">
              <el-tag type="info" size="small">
                时间: {{ mouseCoords.time.toFixed(2) }}s | 信号强度: {{ mouseCoords.value.toFixed(4) }}
              </el-tag>
            </div>
            
            <div class="chart-controls">
              <el-button
                type="primary"
                @click="resetSelection"
                :disabled="!selectedTimeRange"
                style="width: 100%; margin-bottom: 10px"
              >
                <el-icon><Delete /></el-icon>
                重置选择
              </el-button>
            </div>
            
            <div v-if="selectedTimeRange" class="time-range-info">
              <el-alert
                :title="`已选择时间范围: ${selectedTimeRange.start.toFixed(2)}s - ${selectedTimeRange.end.toFixed(2)}s`"
                type="info"
                :closable="false"
                show-icon
              />
              
              <el-button
                type="primary"
                :loading="manualExtractLoading"
                @click="extractSelectedRange"
                style="margin-top: 10px; width: 100%"
              >
                <el-icon><TrendCharts /></el-icon>
                提取选定范围的事件
              </el-button>
            </div>
          </div>
        </div>
        
        <!-- 事件特征表格 -->
        <div v-if="hasAnyFeatures" class="feature-management card">
          <h3 class="section-title">
            <el-icon><Collection /></el-icon>
            事件特征列表（按发生时间排序）
          </h3>
          
          <!-- 特征统计信息 -->
          <div class="feature-stats" style="margin-bottom: 20px;">
            <el-descriptions :column="2" size="small" border>
              <el-descriptions-item label="总事件数">
                <el-tag type="primary">{{ totalFeaturesCount }}</el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="自动检测">
                <el-tag type="info">{{ totalFeaturesCount - manualFeaturesCount }}</el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="手动添加">
                <el-tag type="success">{{ manualFeaturesCount }}</el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="平均振幅">
                <el-tag>{{ averageAmplitude.toFixed(3) }}</el-tag>
              </el-descriptions-item>
            </el-descriptions>
          </div>
          
          <!-- 事件特征表格 -->
          <div class="features-table">
            <el-table 
              :data="allFeaturesSorted" 
              stripe 
              size="small" 
              max-height="400"
              @row-click="handleRowClick"
            >
              <el-table-column label="事件索引" width="100">
                <template #default="scope">
                  <el-tag type="info" size="small">#{{ scope.$index + 1 }}</el-tag>
                </template>
              </el-table-column>
              <el-table-column label="来源" width="80">
                <template #default="scope">
                  <el-tag 
                    :type="scope.row.is_manual ? 'success' : 'primary'" 
                    size="small"
                  >
                    {{ scope.row.is_manual ? '手动' : '自动' }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column prop="peak_time" label="峰值时间(s)" width="120">
                <template #default="scope">
                  {{
                    (() => {
                      let peakTime = null
                      if (scope.row.peak_time !== undefined && scope.row.peak_time !== null && !isNaN(scope.row.peak_time)) {
                        peakTime = scope.row.peak_time
                      } else if (scope.row.peak !== undefined && scope.row.peak !== null && !isNaN(scope.row.peak) && params.fs && params.fs > 0) {
                        peakTime = scope.row.peak / params.fs
                      } else if (scope.row.peak_idx !== undefined && scope.row.peak_idx !== null && !isNaN(scope.row.peak_idx) && params.fs && params.fs > 0) {
                        peakTime = scope.row.peak_idx / params.fs
                      }
                      return peakTime !== null && !isNaN(peakTime) ? peakTime.toFixed(2) : 'N/A'
                    })()
                  }}
                </template>
              </el-table-column>
              <el-table-column prop="amplitude" label="振幅" width="100">
                <template #default="scope">
                  {{ scope.row.amplitude.toFixed(3) }}
                </template>
              </el-table-column>
              <el-table-column prop="duration" label="持续时间(s)" width="120">
                <template #default="scope">
                  {{ scope.row.duration.toFixed(2) }}
                </template>
              </el-table-column>
              <el-table-column prop="fwhm" label="半高宽(s)" width="100">
                <template #default="scope">
                  {{ scope.row.fwhm ? scope.row.fwhm.toFixed(2) : 'N/A' }}
                </template>
              </el-table-column>
              <el-table-column prop="snr" label="信噪比" width="100">
                <template #default="scope">
                  {{ scope.row.snr.toFixed(2) }}
                </template>
              </el-table-column>
              <el-table-column label="操作" width="150" fixed="right">
                <template #default="scope">
                  <el-button 
                    type="primary" 
                    size="small" 
                    @click.stop="editFeature(scope.row, scope.$index)"
                  >
                    编辑
                  </el-button>
                  <el-button 
                    type="danger" 
                    size="small" 
                    @click.stop="deleteFeature(scope.$index)"
                  >
                    删除
                  </el-button>
                </template>
              </el-table-column>
            </el-table>
          </div>
          
          <!-- 操作按钮区域 -->
          <div class="feature-actions" style="margin-top: 15px;">
            <el-row :gutter="10">
              <el-col :span="12">
                <el-button
                  type="warning"
                  :loading="deduplicateLoading"
                  @click="manualDeduplicate"
                  style="width: 100%; margin-bottom: 10px;"
                >
                  <el-icon><Delete /></el-icon>
                  手动去重
                </el-button>
              </el-col>
              <el-col :span="12">
                <el-button
                  type="primary"
                  :loading="savePreviewLoading"
                  @click="saveCombinedResult"
                  style="width: 100%; margin-bottom: 10px;"
                >
                  <el-icon><Download /></el-icon>
                  保存结果
                </el-button>
              </el-col>
            </el-row>
          </div>
          
          <!-- 编辑对话框 -->
          <el-dialog
            v-model="editDialogVisible"
            title="编辑事件特征"
            width="600px"
          >
            <el-form :model="editingFeature" label-width="120px" size="small">
              <el-alert
                title="提示：修改时不会进行校验，请确保输入正确的值"
                type="info"
                :closable="false"
                show-icon
                style="margin-bottom: 15px"
              />
              <el-form-item label="峰值时间(s)">
                <el-input-number
                  v-model="editingFeature.peak_time"
                  :min="0"
                  :precision="2"
                  :step="0.1"
                  style="width: 100%"
                />
              </el-form-item>
              <el-form-item label="起始时间(s)">
                <el-input-number
                  v-model="editingFeature.start_time"
                  :min="0"
                  :precision="2"
                  :step="0.1"
                  style="width: 100%"
                />
              </el-form-item>
              <el-form-item label="结束时间(s)">
                <el-input-number
                  v-model="editingFeature.end_time"
                  :min="0"
                  :precision="2"
                  :step="0.1"
                  style="width: 100%"
                />
              </el-form-item>
              <el-form-item label="振幅">
                <el-input-number
                  v-model="editingFeature.amplitude"
                  :min="0"
                  :precision="3"
                  :step="0.001"
                  style="width: 100%"
                />
              </el-form-item>
              <el-form-item label="持续时间(s)">
                <el-input-number
                  v-model="editingFeature.duration"
                  :min="0"
                  :precision="2"
                  :step="0.1"
                  style="width: 100%"
                />
              </el-form-item>
              <el-form-item label="半高宽(s)" v-if="editingFeature.fwhm !== undefined">
                <el-input-number
                  v-model="editingFeature.fwhm"
                  :min="0"
                  :precision="2"
                  :step="0.1"
                  style="width: 100%"
                />
              </el-form-item>
              <el-form-item label="信噪比" v-if="editingFeature.snr !== undefined">
                <el-input-number
                  v-model="editingFeature.snr"
                  :min="0"
                  :precision="2"
                  :step="0.1"
                  style="width: 100%"
                />
              </el-form-item>
            </el-form>
            <template #footer>
              <el-button @click="editDialogVisible = false">取消</el-button>
              <el-button type="primary" @click="saveEditFeature">保存</el-button>
            </template>
          </el-dialog>
        </div>
        
        <!-- 批量处理区域 -->
        <div v-if="fileList.length > 0" class="batch-section card">
          <h3 class="section-title">
            <el-icon><Operation /></el-icon>
            批量提取事件
          </h3>
          
          <el-button
            type="success"
            size="large"
            :loading="batchLoading"
            @click="startBatchProcessing"
            style="width: 100%"
          >
            <el-icon><Cpu /></el-icon>
            开始批量处理所有上传的文件
          </el-button>
          
          <!-- 批量处理结果 -->
          <div v-if="batchResult" class="batch-result">
            <el-alert
              title="分析完成！"
              type="success"
              :closable="false"
              show-icon
            >
              <template #default>
                批量分析已完成，结果文件：{{ batchResult.result_file }}
              </template>
            </el-alert>
            
            <el-button
              type="primary"
              @click="downloadResult"
              style="margin-top: 15px; width: 100%"
            >
              <el-icon><Download /></el-icon>
              下载结果文件
            </el-button>
            
            <el-button
              type="success"
              @click="$router.push('/clustering')"
              style="margin-top: 10px; width: 100%"
            >
              <el-icon><Right /></el-icon>
              前往聚类分析
            </el-button>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, computed, nextTick, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  DataAnalysis,
  Setting,
  Upload,
  UploadFilled,
  View,
  TrendCharts,
  Operation,
  Cpu,
  Download,
  Right,
  Select,
  Collection,
  Delete
} from '@element-plus/icons-vue'
import { extractionAPI, downloadAPI } from '@/api'
import * as echarts from 'echarts'

// 路由实例
const router = useRouter()

// 响应式数据
const uploadRef = ref()
const fileList = ref([])
const neuronColumns = ref([])
const batchLoading = ref(false)
const batchResult = ref(null)

// 事件提取相关（合并自动和交互）
const selectedNeuronInteractive = ref('')
const interactiveLoading = ref(false)
const autoDetectionLoading = ref(false)
const interactiveData = ref(null)
const interactiveResult = ref(null)
const selectedTimeRange = ref(null)
const manualExtractLoading = ref(false)
const clickCount = ref(0)
const startTime = ref(null)
const mouseCoords = ref(null)

// 特征管理相关
const deduplicateLoading = ref(false)
const savePreviewLoading = ref(false)
const chartContainer = ref()
const editDialogVisible = ref(false)
const editingFeature = ref({})
const editingIndex = ref(-1)

// 所有事件特征列表（合并自动和手动）
const allFeatures = ref([])

let chartInstance = null

// 计算属性
const hasAnyFeatures = computed(() => {
  return allFeatures.value.length > 0
})

const totalFeaturesCount = computed(() => {
  return allFeatures.value.length
})

// 移除单独的autoFeaturesCount和interactiveFeaturesCount，统一使用allFeatures

const manualFeaturesCount = computed(() => {
  return allFeatures.value.filter(f => f.is_manual).length
})

// 按峰值时间排序的所有特征
const allFeaturesSorted = computed(() => {
  return [...allFeatures.value].sort((a, b) => {
    // 获取峰值时间的辅助函数
    const getPeakTime = (feature) => {
      if (feature.peak_time !== undefined && feature.peak_time !== null && !isNaN(feature.peak_time)) {
        return feature.peak_time
      } else if (feature.peak !== undefined && feature.peak !== null && !isNaN(feature.peak) && params.fs && params.fs > 0) {
        return feature.peak / params.fs
      } else if (feature.peak_idx !== undefined && feature.peak_idx !== null && !isNaN(feature.peak_idx) && params.fs && params.fs > 0) {
        return feature.peak_idx / params.fs
      }
      return 0
    }
    
    const timeA = getPeakTime(a)
    const timeB = getPeakTime(b)
    return timeA - timeB
  })
})

const averageAmplitude = computed(() => {
  if (allFeatures.value.length === 0) return 0
  return allFeatures.value.reduce((sum, f) => sum + f.amplitude, 0) / allFeatures.value.length
})

// 参数配置
const params = reactive({
  fs: 4.8,
  min_duration_frames: 12,
  max_duration_frames: 800,
  min_snr: 3.5,
  smooth_window: 31,
  peak_distance_frames: 24,
  filter_strength: 1.0
})

// 文件变化处理
const handleFileChange = async (file, files) => {
  fileList.value = files
  // 重置相关状态
  neuronColumns.value = []
  selectedNeuronInteractive.value = ''
  interactiveResult.value = null
  interactiveData.value = null
  selectedTimeRange.value = null
  batchResult.value = null
  allFeatures.value = []
  
  // 清理图表
  if (chartInstance) {
    chartInstance.dispose()
    chartInstance = null
  }
  
  // 如果有文件，尝试获取神经元列并自动提取事件
  if (files.length > 0) {
    await loadNeuronColumns(files[0])
    // 自动选择第一个神经元并开始检测
    if (neuronColumns.value.length > 0) {
      selectedNeuronInteractive.value = neuronColumns.value[0]
      // 先加载图表，再提取事件（这样图表初始化时就能显示事件）
      await loadInteractiveChart()
      // 等待图表初始化完成
      await nextTick()
      await startAutoDetection()
    }
  }
}

// 加载神经元列
const loadNeuronColumns = async (file) => {
  try {
    const formData = new FormData()
    formData.append('file', file.raw)
    formData.append('fs', params.fs)
    formData.append('min_duration_frames', params.min_duration_frames)
    formData.append('max_duration_frames', params.max_duration_frames)
    formData.append('min_snr', params.min_snr)
    formData.append('smooth_window', params.smooth_window)
    formData.append('peak_distance_frames', params.peak_distance_frames)
    formData.append('filter_strength', params.filter_strength)
    formData.append('neuron_id', 'temp') // 临时值，只为获取列名
    
    const response = await extractionAPI.preview(formData)
    if (response.success && response.neuron_columns) {
      neuronColumns.value = response.neuron_columns
      if (neuronColumns.value.length > 0) {
        selectedNeuron.value = neuronColumns.value[0]
        selectedNeuronInteractive.value = neuronColumns.value[0]
      }
    }
  } catch (error) {
    console.error('加载神经元列失败:', error)
  }
}

// 开始自动检测（合并到统一流程）
const startAutoDetection = async () => {
  if (!selectedNeuronInteractive.value || fileList.value.length === 0) {
    return
  }
  
  autoDetectionLoading.value = true
  try {
    const formData = new FormData()
    formData.append('file', fileList.value[0].raw)
    formData.append('fs', params.fs)
    formData.append('min_duration_frames', params.min_duration_frames)
    formData.append('max_duration_frames', params.max_duration_frames)
    formData.append('min_snr', params.min_snr)
    formData.append('smooth_window', params.smooth_window)
    formData.append('peak_distance_frames', params.peak_distance_frames)
    formData.append('filter_strength', params.filter_strength)
    formData.append('neuron_id', selectedNeuronInteractive.value)
    
    const response = await extractionAPI.preview(formData)
    console.log('自动检测响应:', response)
    if (response.success) {
      // 将自动检测的特征添加到总列表，标记为自动
      if (response.features && response.features.length > 0) {
        console.log('收到特征数据:', response.features.length, '个')
        console.log('第一个特征示例:', response.features[0])
        
        const autoFeatures = response.features.map(f => {
          // 确保所有时间字段都存在，避免 NaN
          const getPeakTime = () => {
            if (f.peak_time !== undefined && f.peak_time !== null && !isNaN(f.peak_time)) {
              return f.peak_time
            } else if (f.peak !== undefined && f.peak !== null && !isNaN(f.peak) && params.fs && params.fs > 0) {
              return f.peak / params.fs
            } else if (f.peak_idx !== undefined && f.peak_idx !== null && !isNaN(f.peak_idx) && params.fs && params.fs > 0) {
              return f.peak_idx / params.fs
            }
            return null
          }
          
          const getStartTime = () => {
            if (f.start_time !== undefined && f.start_time !== null && !isNaN(f.start_time)) {
              return f.start_time
            } else if (f.start !== undefined && f.start !== null && !isNaN(f.start) && params.fs && params.fs > 0) {
              return f.start / params.fs
            } else if (f.start_idx !== undefined && f.start_idx !== null && !isNaN(f.start_idx) && params.fs && params.fs > 0) {
              return f.start_idx / params.fs
            }
            return null
          }
          
          const getEndTime = () => {
            if (f.end_time !== undefined && f.end_time !== null && !isNaN(f.end_time)) {
              return f.end_time
            } else if (f.end !== undefined && f.end !== null && !isNaN(f.end) && params.fs && params.fs > 0) {
              return f.end / params.fs
            } else if (f.end_idx !== undefined && f.end_idx !== null && !isNaN(f.end_idx) && params.fs && params.fs > 0) {
              return f.end_idx / params.fs
            }
            return null
          }
          
          const feature = {
            ...f,
            is_manual: false,
            peak_time: getPeakTime(),
            start_time: getStartTime(),
            end_time: getEndTime(),
            peak_idx: f.peak_idx !== undefined && !isNaN(f.peak_idx) ? f.peak_idx : (f.peak !== undefined && !isNaN(f.peak) ? f.peak : null),
            start_idx: f.start_idx !== undefined && !isNaN(f.start_idx) ? f.start_idx : (f.start !== undefined && !isNaN(f.start) ? f.start : null),
            end_idx: f.end_idx !== undefined && !isNaN(f.end_idx) ? f.end_idx : (f.end !== undefined && !isNaN(f.end) ? f.end : null),
            peak_value: f.peak_value !== undefined && !isNaN(f.peak_value) ? f.peak_value : null
          }
          console.log('处理后的特征:', feature)
          return feature
        })
        
        // 合并到总列表，避免重复
        let addedCount = 0
        autoFeatures.forEach(f => {
          const exists = allFeatures.value.some(existing => {
            const getTime = (feat) => {
              if (feat.peak_time !== undefined && feat.peak_time !== null && !isNaN(feat.peak_time)) {
                return feat.peak_time
              } else if (feat.peak !== undefined && feat.peak !== null && !isNaN(feat.peak) && params.fs && params.fs > 0) {
                return feat.peak / params.fs
              } else if (feat.peak_idx !== undefined && feat.peak_idx !== null && !isNaN(feat.peak_idx) && params.fs && params.fs > 0) {
                return feat.peak_idx / params.fs
              }
              return null
            }
            const existingTime = getTime(existing)
            const newTime = getTime(f)
            if (existingTime === null || newTime === null || isNaN(existingTime) || isNaN(newTime)) return false
            return Math.abs(existingTime - newTime) < 0.1
          })
          if (!exists) {
            allFeatures.value.push(f)
            addedCount++
          }
        })
        
        console.log(`添加了 ${addedCount} 个新特征，当前总数: ${allFeatures.value.length}`)
        
        // 重新初始化图表以显示新事件
        if (interactiveData.value && chartInstance) {
          await nextTick()
          initChart()
        } else if (interactiveData.value) {
          // 如果图表还没初始化，等待一下再初始化
          await nextTick()
          await nextTick()
          initChart()
        }
      }
      ElMessage.success(`自动检测完成，检测到 ${response.features?.length || 0} 个事件`)
    } else {
      ElMessage.error('自动检测失败: ' + (response.message || '未知错误'))
    }
  } catch (error) {
    console.error('自动检测失败:', error)
    ElMessage.error('自动检测失败: ' + (error.response?.data?.detail || error.message || '网络错误'))
  } finally {
    autoDetectionLoading.value = false
  }
}

// 加载交互式图表
const loadInteractiveChart = async () => {
  if (!selectedNeuronInteractive.value || fileList.value.length === 0) {
    ElMessage.warning('请选择神经元')
    return
  }
  
  interactiveLoading.value = true
  try {
    const formData = new FormData()
    formData.append('file', fileList.value[0].raw)
    formData.append('fs', params.fs)
    formData.append('neuron_id', selectedNeuronInteractive.value)
    
    const response = await extractionAPI.getInteractiveData(formData)
    if (response.success) {
      interactiveData.value = response.data
      await nextTick()
      initChart()
    } else {
      ElMessage.error('获取数据失败')
    }
  } catch (error) {
    console.error('加载数据失败:', error)
    ElMessage.error('加载数据失败: ' + (error.response?.data?.detail || error.message || '网络错误'))
  } finally {
    interactiveLoading.value = false
  }
}

// 处理神经元切换
const handleNeuronChange = async () => {
  // 清空当前事件列表
  allFeatures.value = []
  selectedTimeRange.value = null
  clickCount.value = 0
  startTime.value = null
  
  // 清理图表
  if (chartInstance) {
    chartInstance.dispose()
    chartInstance = null
  }
  
  // 加载新神经元的数据
  if (selectedNeuronInteractive.value) {
    await loadInteractiveChart()
    await startAutoDetection()
  }
}

// 重新提取事件
const reloadExtraction = async () => {
  // 清空当前事件列表
  allFeatures.value = []
  selectedTimeRange.value = null
  clickCount.value = 0
  startTime.value = null
  
  // 重新加载图表和提取事件
  await loadInteractiveChart()
  await startAutoDetection()
}

// 开始批量处理
const startBatchProcessing = async () => {
  if (fileList.value.length === 0) {
    ElMessage.warning('请先上传文件')
    return
  }
  
  batchLoading.value = true
  try {
    const formData = new FormData()
    
    // 添加所有文件
    fileList.value.forEach(file => {
      formData.append('files', file.raw)
    })
    
    // 添加参数
    formData.append('fs', params.fs)
    formData.append('min_duration_frames', params.min_duration_frames)
    formData.append('max_duration_frames', params.max_duration_frames)
    formData.append('min_snr', params.min_snr)
    formData.append('smooth_window', params.smooth_window)
    formData.append('peak_distance_frames', params.peak_distance_frames)
    formData.append('filter_strength', params.filter_strength)
    
    const response = await extractionAPI.batchExtraction(formData)
    if (response.success) {
      batchResult.value = response
      ElMessage.success('批量处理完成')
    }
  } catch (error) {
    console.error('批量处理失败:', error)
  } finally {
    batchLoading.value = false
  }
}

// 下载结果
const downloadResult = async () => {
  if (!batchResult.value?.result_file) {
    ElMessage.error('没有可下载的结果文件')
    return
  }
  
  try {
    const response = await downloadAPI.downloadFile(batchResult.value.result_file)
    
    // 创建下载链接
    const blob = new Blob([response], {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = batchResult.value.result_file
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

// 保存合并结果
const saveCombinedResult = async () => {
  if (!hasAnyFeatures.value) {
    ElMessage.warning('没有可保存的特征数据')
    return
  }
  
  if (!fileList.value.length) {
    ElMessage.warning('请确保已上传文件')
    return
  }
  
  savePreviewLoading.value = true
  try {
    // 使用allFeatures列表
    const featuresToSave = allFeatures.value.map(f => ({
      ...f,
      isManualExtracted: f.is_manual || false
    }))
    
    // 构建要保存的数据
    const saveData = {
      filename: fileList.value[0].name,
      neuron: selectedNeuronInteractive.value,
      features: featuresToSave,
      params: params,
      total_features: featuresToSave.length,
      auto_features: allFeatures.value.filter(f => !f.is_manual).length,
      manual_features: allFeatures.value.filter(f => f.is_manual).length
    }
    
    // 调用后端API保存合并结果
    const formData = new FormData()
    formData.append('data', JSON.stringify(saveData))
    
    const response = await extractionAPI.savePreviewResult(formData)
    
    if (response.success) {
      ElMessage.success(`合并结果已保存: ${response.filename}`)
    } else {
      ElMessage.error('保存失败: ' + (response.message || '未知错误'))
    }
  } catch (error) {
    console.error('保存失败:', error)
    ElMessage.error('保存失败: ' + (error.response?.data?.detail || error.message || '网络错误'))
  } finally {
    savePreviewLoading.value = false
  }
}

// 跳转到聚类分析
const goToClustering = () => {
  if (!hasAnyFeatures.value) {
    ElMessage.warning('当前没有特征数据，建议先进行事件提取')
    return
  }
  
  router.push('/clustering')
  ElMessage.success('已跳转到聚类分析页面')
}


// 初始化图表
const initChart = () => {
  console.log('initChart被调用，chartContainer:', chartContainer.value)
  console.log('interactiveData:', interactiveData.value)
  
  if (!chartContainer.value || !interactiveData.value) {
    console.log('缺少必要条件，退出initChart')
    return
  }
  
  // 检查数据结构
  if (!interactiveData.value.time || !interactiveData.value.data) {
    console.error('数据结构错误:', {
      hasTime: !!interactiveData.value.time,
      hasData: !!interactiveData.value.data,
      keys: Object.keys(interactiveData.value)
    })
    ElMessage.error('数据格式错误，请检查后端API返回')
    return
  }
  
  // 销毁现有图表
  if (chartInstance) {
    chartInstance.dispose()
  }
  
  chartInstance = echarts.init(chartContainer.value)
  
  // 准备数据 - 将时间轴和数据组合成坐标点
  const chartData = interactiveData.value.time.map((time, index) => [
    parseFloat(time),
    interactiveData.value.data[index]
  ])
  
  console.log('图表数据准备完成，数据点数量:', chartData.length)
  
  // 准备已检测的事件标记
  const markPoints = []
  const markLines = []
  const markAreas = []
  
  console.log('准备标记钙波，allFeatures数量:', allFeatures.value.length)
  if (allFeatures.value.length > 0) {
    console.log('第一个特征示例:', allFeatures.value[0])
  }
  
  // 添加已检测的事件到图表
  allFeatures.value.forEach((feature, idx) => {
    // 兼容多种时间字段格式
    let peakTime = null
    let startTime = null
    let endTime = null
    
    if (feature.peak_time !== undefined && feature.peak_time !== null) {
      peakTime = feature.peak_time
    } else if (feature.peak !== undefined && feature.peak !== null) {
      peakTime = feature.peak / params.fs
    } else if (feature.peak_idx !== undefined && feature.peak_idx !== null) {
      peakTime = feature.peak_idx / params.fs
    }
    
    if (feature.start_time !== undefined && feature.start_time !== null) {
      startTime = feature.start_time
    } else if (feature.start !== undefined && feature.start !== null) {
      startTime = feature.start / params.fs
    } else if (feature.start_idx !== undefined && feature.start_idx !== null) {
      startTime = feature.start_idx / params.fs
    }
    
    if (feature.end_time !== undefined && feature.end_time !== null) {
      endTime = feature.end_time
    } else if (feature.end !== undefined && feature.end !== null) {
      endTime = feature.end / params.fs
    } else if (feature.end_idx !== undefined && feature.end_idx !== null) {
      endTime = feature.end_idx / params.fs
    }
    
    // 验证时间值
    if (peakTime === null || startTime === null || endTime === null) {
      console.warn(`事件特征 #${idx + 1} 缺少时间信息:`, {
        peakTime, startTime, endTime, 
        feature_keys: Object.keys(feature),
        feature: feature
      })
      return
    }
    
    // 获取峰值对应的数据值
    const peakIdx = Math.round(peakTime * params.fs)
    let peakValue = null
    
    if (feature.peak_value !== undefined && feature.peak_value !== null) {
      peakValue = feature.peak_value
    } else if (peakIdx >= 0 && peakIdx < interactiveData.value.data.length) {
      peakValue = interactiveData.value.data[peakIdx]
    }
    
    if (peakValue === null) {
      console.warn(`无法获取峰值数据 #${idx + 1}:`, { 
        peakTime, peakIdx, 
        dataLength: interactiveData.value.data.length,
        feature: feature
      })
      return
    }
    
    console.log(`✓ 添加钙波标记 #${idx + 1}:`, { 
      peakTime: peakTime.toFixed(2), 
      startTime: startTime.toFixed(2), 
      endTime: endTime.toFixed(2), 
      peakValue: peakValue.toFixed(4), 
      is_manual: feature.is_manual 
    })
    
    // 标记峰值点（更大更明显）
    markPoints.push({
      coord: [peakTime, peakValue],
      symbol: 'circle',
      symbolSize: 12,
      itemStyle: {
        color: feature.is_manual ? '#67C23A' : '#409EFF',
        borderColor: '#fff',
        borderWidth: 2
      },
      label: {
        show: true,
        formatter: `#${idx + 1}`,
        position: 'top',
        fontSize: 10,
        fontWeight: 'bold',
        color: feature.is_manual ? '#67C23A' : '#409EFF'
      }
    })
    
    // 标记起始和结束线（更明显）
    markLines.push(
      {
        xAxis: startTime,
        lineStyle: { 
          color: feature.is_manual ? '#67C23A' : '#409EFF', 
          width: 2, 
          type: 'dashed',
          opacity: 0.8
        },
        label: {
          show: true,
          formatter: '起始',
          position: 'insideEndTop',
          fontSize: 9,
          color: feature.is_manual ? '#67C23A' : '#409EFF'
        }
      },
      {
        xAxis: endTime,
        lineStyle: { 
          color: feature.is_manual ? '#67C23A' : '#409EFF', 
          width: 2, 
          type: 'dashed',
          opacity: 0.8
        },
        label: {
          show: true,
          formatter: '结束',
          position: 'insideEndTop',
          fontSize: 9,
          color: feature.is_manual ? '#67C23A' : '#409EFF'
        }
      }
    )
    
    // 标记事件区域（更明显的背景色）
    markAreas.push([
      {
        xAxis: startTime,
        itemStyle: { 
          color: feature.is_manual ? 'rgba(103, 194, 58, 0.25)' : 'rgba(64, 158, 255, 0.25)',
          borderColor: feature.is_manual ? '#67C23A' : '#409EFF',
          borderWidth: 1,
          borderType: 'solid'
        },
        label: {
          show: true,
          formatter: feature.is_manual ? '手动' : '自动',
          position: 'inside',
          fontSize: 10,
          fontWeight: 'bold',
          color: feature.is_manual ? '#67C23A' : '#409EFF'
        }
      },
      {
        xAxis: endTime
      }
    ])
  })
  
  console.log('标记准备完成:', {
    markPoints: markPoints.length,
    markLines: markLines.length,
    markAreas: markAreas.length
  })
  
  const option = {
    title: {
      text: `神经元 ${interactiveData.value.neuron_id} 钙信号`,
      left: 'center'
    },
    tooltip: {
      trigger: 'axis',
      formatter: function(params) {
        const time = params[0].value[0]
        const value = params[0].value[1]
        return `时间: ${time.toFixed(2)}s<br/>信号强度: ${value.toFixed(4)}`
      },
      axisPointer: {
        type: 'cross',
        label: {
          backgroundColor: '#6a7985'
        }
      }
    },
    toolbox: {
      show: true,
      feature: {
        dataZoom: {
          yAxisIndex: 'none',
          title: {
            zoom: '区域缩放',
            back: '区域缩放还原'
          }
        },
        restore: {
          title: '重置'
        },
        saveAsImage: {
          title: '保存图片'
        }
      }
    },
    dataZoom: [
      {
        type: 'inside',
        start: 0,
        end: 100,
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
        moveOnMouseWheel: false
      },
      {
        type: 'slider',
        start: 0,
        end: 100,
        height: 20,
        showDataShadow: true,
        showDetail: true
      }
    ],
    xAxis: {
      type: 'value',
      name: '时间 (s)',
      nameLocation: 'middle',
      nameGap: 30
    },
    yAxis: {
      type: 'value',
      name: '荧光强度',
      nameLocation: 'middle',
      nameGap: 50
    },
    series: [{
      name: '钙信号',
      type: 'line',
      data: chartData,
      symbol: 'circle',
      symbolSize: 0,
      lineStyle: {
        width: 1
      },
      large: false,
      sampling: 'none',
      triggerLineEvent: true,
      markPoint: {
        data: markPoints,
        symbol: 'circle',
        symbolSize: 12,
        label: {
          show: true,
          position: 'top',
          fontSize: 10,
          fontWeight: 'bold'
        },
        itemStyle: {
          borderWidth: 2
        }
      },
      markLine: {
        data: markLines,
        symbol: ['none', 'none'],
        label: {
          show: true,
          position: 'insideEndTop',
          fontSize: 9
        },
        lineStyle: {
          width: 2,
          type: 'dashed',
          opacity: 0.8
        }
      },
      markArea: {
        data: markAreas,
        itemStyle: {
          borderWidth: 1,
          borderType: 'solid'
        },
        label: {
          show: true,
          position: 'inside',
          fontSize: 10,
          fontWeight: 'bold'
        },
        emphasis: {
          itemStyle: {
            opacity: 0.3
          }
        }
      }
    }]
  }
  
  // 使用 notMerge=false 来合并配置，确保标记能正确显示
  chartInstance.setOption(option, false)
  
  console.log('图表配置已应用，标记数量:', {
    points: markPoints.length,
    lines: markLines.length,
    areas: markAreas.length,
    allFeaturesCount: allFeatures.value.length
  })
  
  // 强制刷新图表
  chartInstance.resize()
  
  // 移除之前的事件监听器，避免重复绑定
  chartInstance.getZr().off('mousemove')
  chartInstance.getZr().off('mouseout')
  chartInstance.getZr().off('click')
  
  // 监听鼠标移动，显示坐标
  chartInstance.getZr().on('mousemove', function(event) {
    const pointInPixel = [event.offsetX, event.offsetY]
    const pointInGrid = chartInstance.convertFromPixel('grid', pointInPixel)
    
    if (pointInGrid && pointInGrid[0] !== null && pointInGrid[0] !== undefined) {
      const time = pointInGrid[0]
      const value = pointInGrid[1]
      
      // 找到最接近的数据点
      const timeIndex = Math.round(time * params.fs)
      if (timeIndex >= 0 && timeIndex < interactiveData.value.data.length) {
        mouseCoords.value = {
          time: time,
          value: interactiveData.value.data[timeIndex]
        }
      }
    }
  })
  
  chartInstance.getZr().on('mouseout', function() {
    mouseCoords.value = null
  })
  
  // 监听图表点击事件 - 优先检测峰值，否则进行范围选择
  chartInstance.getZr().on('click', async function(event) {
    console.log('图表区域点击事件触发:', event)
    
    // 将像素坐标转换为数据坐标
    const pointInPixel = [event.offsetX, event.offsetY]
    const pointInGrid = chartInstance.convertFromPixel('grid', pointInPixel)
    
    if (pointInGrid && pointInGrid[0] !== null && pointInGrid[0] !== undefined) {
      const clickedTime = pointInGrid[0]
      const clickedValue = pointInGrid[1]
      
      // 范围选择模式：两次点击选择时间范围
      console.log('点击的时间点:', clickedTime, '当前点击计数:', clickCount.value)
      
      if (clickCount.value === 0) {
        // 第一次点击，设置起始时间
        startTime.value = clickedTime
        clickCount.value = 1
        ElMessage.info(`已选择起始时间: ${clickedTime.toFixed(2)}s，请点击选择结束时间`)
        
        // 在图表上标记起始点
        updateChartMarkLines([{
          xAxis: clickedTime,
          lineStyle: { color: '#67C23A', width: 2 },
          label: { 
            show: true,
            formatter: '起始点',
            position: 'insideEndTop'
          }
        }])
      } else {
        // 第二次点击，设置结束时间
        const endTime = clickedTime
        
        if (endTime <= startTime.value) {
          ElMessage.warning('结束时间必须大于起始时间，请重新选择')
          clickCount.value = 0
          startTime.value = null
          return
        }
        
        selectedTimeRange.value = {
          start: parseFloat(startTime.value),
          end: parseFloat(endTime)
        }
        clickCount.value = 0
        startTime.value = null
        
        ElMessage.success(`已选择时间范围: ${selectedTimeRange.value.start.toFixed(2)}s - ${selectedTimeRange.value.end.toFixed(2)}s`)
        
        // 在图表上标记起始点和结束点，以及选择区域
         updateChartMarkLines([
           {
             xAxis: selectedTimeRange.value.start,
             lineStyle: { color: '#67C23A', width: 2 },
             label: { 
               show: true,
               formatter: '起始点',
               position: 'insideEndTop'
             }
           },
           {
             xAxis: selectedTimeRange.value.end,
             lineStyle: { color: '#F56C6C', width: 2 },
             label: { 
               show: true,
               formatter: '结束点',
               position: 'insideEndTop'
             }
           }
         ])
        
        // 添加选择区域高亮
        updateChartMarkArea({
          xAxis: selectedTimeRange.value.start
        }, {
          xAxis: selectedTimeRange.value.end
        })
      }
    }
   })
}

// 更新图表标记线
const updateChartMarkLines = (markLines) => {
  console.log('更新标记线:', markLines)
  if (chartInstance) {
    const option = {
      series: [{
        markLine: {
          data: markLines,
          symbol: 'none',
          label: {
            show: true,
            position: 'end',
            color: '#333'
          },
          lineStyle: {
            type: 'solid'
          }
        }
      }]
    }
    chartInstance.setOption(option, false, true)
    console.log('标记线已更新')
  }
}

// 更新图表标记区域
const updateChartMarkArea = (start, end) => {
  console.log('更新标记区域:', start, end)
  if (chartInstance) {
    const option = {
      series: [{
        markArea: {
          data: [[
            start,
            end
          ]],
          itemStyle: {
            color: 'rgba(103, 194, 58, 0.2)'
          }
        }
      }]
    }
    chartInstance.setOption(option, false, true)
    console.log('标记区域已更新')
  }
}

// 重置选择
const resetSelection = () => {
  selectedTimeRange.value = null
  clickCount.value = 0
  startTime.value = null
  
  // 清除图表上的标记
  if (chartInstance) {
    const option = {
      series: [{
        markLine: {
          data: []
        },
        markArea: {
          data: []
        }
      }]
    }
    chartInstance.setOption(option, false, true)
  }
  
  ElMessage.success('已重置选择，请重新点击选择时间范围')
}

// 编辑事件特征
const editFeature = (feature, index) => {
  editingFeature.value = { ...feature }
  editingIndex.value = index
  editDialogVisible.value = true
}

// 删除事件特征
const deleteFeature = (index) => {
  ElMessageBox.confirm(
    '确定要删除这个事件特征吗？',
    '确认删除',
    {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    }
  ).then(() => {
    // 从排序后的列表获取要删除的特征
    const featureToDelete = allFeaturesSorted.value[index]
    // 在原始列表中查找并删除
    const getPeakTime = (feat) => {
      if (feat.peak_time !== undefined && feat.peak_time !== null && !isNaN(feat.peak_time)) {
        return feat.peak_time
      } else if (feat.peak !== undefined && feat.peak !== null && !isNaN(feat.peak) && params.fs && params.fs > 0) {
        return feat.peak / params.fs
      } else if (feat.peak_idx !== undefined && feat.peak_idx !== null && !isNaN(feat.peak_idx) && params.fs && params.fs > 0) {
        return feat.peak_idx / params.fs
      }
      return null
    }
    
    const originalIndex = allFeatures.value.findIndex(f => {
      const timeA = getPeakTime(f)
      const timeB = getPeakTime(featureToDelete)
      if (timeA === null || timeB === null || isNaN(timeA) || isNaN(timeB)) return false
      return Math.abs(timeA - timeB) < 0.01
    })
    if (originalIndex >= 0) {
      allFeatures.value.splice(originalIndex, 1)
      ElMessage.success('已删除事件特征')
      // 重新初始化图表
      if (interactiveData.value) {
        nextTick(() => {
          initChart()
        })
      }
    }
  }).catch(() => {
    // 用户取消
  })
}

// 保存编辑的事件特征
const saveEditFeature = () => {
  if (editingIndex.value >= 0 && editingIndex.value < allFeaturesSorted.value.length) {
    // 从排序后的列表获取要编辑的特征
    const featureToEdit = allFeaturesSorted.value[editingIndex.value]
    // 在原始列表中查找并更新
    const getPeakTime = (feat) => {
      if (feat.peak_time !== undefined && feat.peak_time !== null && !isNaN(feat.peak_time)) {
        return feat.peak_time
      } else if (feat.peak !== undefined && feat.peak !== null && !isNaN(feat.peak) && params.fs && params.fs > 0) {
        return feat.peak / params.fs
      } else if (feat.peak_idx !== undefined && feat.peak_idx !== null && !isNaN(feat.peak_idx) && params.fs && params.fs > 0) {
        return feat.peak_idx / params.fs
      }
      return null
    }
    
    const originalIndex = allFeatures.value.findIndex(f => {
      const timeA = getPeakTime(f)
      const timeB = getPeakTime(featureToEdit)
      if (timeA === null || timeB === null || isNaN(timeA) || isNaN(timeB)) return false
      return Math.abs(timeA - timeB) < 0.01
    })
    if (originalIndex >= 0) {
      // 更新特征（不进行校验，直接保存）
      // 确保时间索引也更新
      const updatedFeature = {
        ...allFeatures.value[originalIndex],
        ...editingFeature.value,
        is_manual: true  // 编辑后标记为手动
      }
      
      // 如果修改了时间，需要更新索引
      if (editingFeature.value.peak_time !== undefined) {
        updatedFeature.peak_time = editingFeature.value.peak_time
        if (updatedFeature.peak_idx !== undefined) {
          updatedFeature.peak_idx = Math.round(editingFeature.value.peak_time * params.fs)
        }
      }
      if (editingFeature.value.start_time !== undefined) {
        updatedFeature.start_time = editingFeature.value.start_time
        if (updatedFeature.start_idx !== undefined) {
          updatedFeature.start_idx = Math.round(editingFeature.value.start_time * params.fs)
        }
      }
      if (editingFeature.value.end_time !== undefined) {
        updatedFeature.end_time = editingFeature.value.end_time
        if (updatedFeature.end_idx !== undefined) {
          updatedFeature.end_idx = Math.round(editingFeature.value.end_time * params.fs)
        }
      }
      
      allFeatures.value[originalIndex] = updatedFeature
      ElMessage.success('事件特征已更新')
      editDialogVisible.value = false
      // 重新初始化图表
      if (interactiveData.value) {
        nextTick(() => {
          initChart()
        })
      }
    }
  }
}

// 处理表格行点击
const handleRowClick = (row) => {
  // 可以在这里实现跳转到对应时间点的功能
  console.log('点击了事件:', row)
}

// 提取选定范围的事件
const extractSelectedRange = async () => {
  if (!selectedTimeRange.value || !selectedNeuronInteractive.value || fileList.value.length === 0) {
    ElMessage.warning('请先选择时间范围')
    return
  }
  
  manualExtractLoading.value = true
  try {
    const formData = new FormData()
    formData.append('file', fileList.value[0].raw)
    formData.append('fs', params.fs)
    formData.append('neuron_id', selectedNeuronInteractive.value)
    formData.append('start_time', selectedTimeRange.value.start)
    formData.append('end_time', selectedTimeRange.value.end)
    formData.append('min_duration_frames', params.min_duration_frames)
    formData.append('max_duration_frames', params.max_duration_frames)
    formData.append('min_snr', params.min_snr)
    formData.append('smooth_window', params.smooth_window)
    formData.append('peak_distance_frames', params.peak_distance_frames)
    formData.append('filter_strength', params.filter_strength)
    
    console.log('发送手动提取请求...')
    const response = await extractionAPI.manualExtract(formData)
    console.log('手动提取响应:', response)
    console.log('响应中的plot字段:', response.plot)
    console.log('响应中的transients字段:', response.transients)
    console.log('响应中的success字段:', response.success)
    
    if (response.success) {
      console.log('提取结果:', response)
      
      if (response.transients && response.transients.length > 0) {
        // 为新特征添加手动提取标识
        const newFeatures = response.transients.map(feature => ({
          ...feature,
          is_manual: true,
          peak_time: feature.peak_time || (feature.peak / params.fs),
          start_time: feature.start_time || (feature.start / params.fs),
          end_time: feature.end_time || (feature.end / params.fs)
        }))
        
        // 创建去重函数，基于多个特征进行更精确的去重
        const isDuplicate = (feature1, feature2) => {
          // 时间容差：0.1秒
          const timeTolerance = 0.1
          // 振幅容差：相对误差5%
          const amplitudeTolerance = Math.max(0.01, Math.abs(feature1.amplitude) * 0.05)
          // 持续时间容差：0.05秒
          const durationTolerance = 0.05
          
          const getStartTime = (feat) => {
            if (feat.start_time !== undefined && feat.start_time !== null && !isNaN(feat.start_time)) {
              return feat.start_time
            } else if (feat.start !== undefined && feat.start !== null && !isNaN(feat.start) && params.fs && params.fs > 0) {
              return feat.start / params.fs
            } else if (feat.start_idx !== undefined && feat.start_idx !== null && !isNaN(feat.start_idx) && params.fs && params.fs > 0) {
              return feat.start_idx / params.fs
            }
            return null
          }
          
          const time1 = getStartTime(feature1)
          const time2 = getStartTime(feature2)
          const timeMatch = (time1 !== null && time2 !== null && !isNaN(time1) && !isNaN(time2)) ? 
                           Math.abs(time1 - time2) < timeTolerance : false
          const amplitudeMatch = Math.abs(feature1.amplitude - feature2.amplitude) < amplitudeTolerance
          const durationMatch = Math.abs(feature1.duration - feature2.duration) < durationTolerance
          
          // 如果有峰值时间，也进行比较
          const getPeakTime = (feat) => {
            if (feat.peak_time !== undefined && feat.peak_time !== null && !isNaN(feat.peak_time)) {
              return feat.peak_time
            } else if (feat.peak !== undefined && feat.peak !== null && !isNaN(feat.peak) && params.fs && params.fs > 0) {
              return feat.peak / params.fs
            } else if (feat.peak_idx !== undefined && feat.peak_idx !== null && !isNaN(feat.peak_idx) && params.fs && params.fs > 0) {
              return feat.peak_idx / params.fs
            }
            return null
          }
          
          const peakTime1 = getPeakTime(feature1)
          const peakTime2 = getPeakTime(feature2)
          const peakTimeMatch = (peakTime1 !== null && peakTime2 !== null && !isNaN(peakTime1) && !isNaN(peakTime2)) ? 
                                Math.abs(peakTime1 - peakTime2) < timeTolerance : false
          
          return timeMatch && amplitudeMatch && durationMatch && peakTimeMatch
        }
        
        // 过滤重复的特征
        const uniqueNewFeatures = newFeatures.filter(newFeature => {
          const isDuplicateFeature = allFeatures.value.some(existingFeature => isDuplicate(newFeature, existingFeature))
          if (isDuplicateFeature) {
            console.log('发现重复特征，已过滤:', {
              start_time: newFeature.start_time,
              amplitude: newFeature.amplitude,
              duration: newFeature.duration
            })
          }
          return !isDuplicateFeature
        })
        
        // 添加到总列表
        uniqueNewFeatures.forEach(f => {
          allFeatures.value.push(f)
        })
        
        // 更新交互式结果（用于显示）
        if (!interactiveResult.value) {
          interactiveResult.value = { features: [], plot: null }
        }
        interactiveResult.value.features = [...interactiveResult.value.features, ...uniqueNewFeatures]
        if (response.plot) {
          interactiveResult.value.plot = response.plot
        }
        
        console.log('检测到事件特征:', response.transients.length, '个，其中', uniqueNewFeatures.length, '个为新特征')
        ElMessage.success(`交互式提取完成，检测到 ${response.transients.length} 个事件特征，其中 ${uniqueNewFeatures.length} 个为新特征`)
        
        // 重新初始化图表以显示新事件
        if (interactiveData.value) {
          await nextTick()
          initChart()
        }
      } else {
        console.log('没有检测到事件特征')
        ElMessage.info('在选定时间范围内未检测到事件特征')
      }
    } else {
      console.error('手动提取失败:', response)
      ElMessage.error('手动提取失败: ' + (response.message || '未知错误'))
    }
  } catch (error) {
    console.error('手动提取失败:', error)
    console.error('错误详情:', error.response?.data || error.message)
    ElMessage.error('手动提取请求失败: ' + (error.response?.data?.message || error.message || '网络错误'))
  } finally {
    manualExtractLoading.value = false
  }
}


 
 // 组件挂载和卸载
onMounted(() => {
  // 监听窗口大小变化
  window.addEventListener('resize', handleResize)
  
  // 处理ResizeObserver错误
  const resizeObserverErrorHandler = (e) => {
    if (e.message === 'ResizeObserver loop completed with undelivered notifications.') {
      e.stopImmediatePropagation()
    }
  }
  window.addEventListener('error', resizeObserverErrorHandler)
})

// 手动去重特征列表
const manualDeduplicate = async () => {
  if (!hasAnyFeatures.value) {
    ElMessage.warning('没有特征数据需要去重')
    return
  }
  
  deduplicateLoading.value = true
  
  try {
    const originalCount = allFeatures.value.length
    
    // 创建去重函数（与extractSelectedRange中的相同）
    const isDuplicate = (feature1, feature2) => {
      // 时间容差：0.1秒
      const timeTolerance = 0.1
      // 振幅容差：相对误差5%
      const amplitudeTolerance = Math.max(0.01, Math.abs(feature1.amplitude) * 0.05)
      // 持续时间容差：0.05秒
      const durationTolerance = 0.05
      
      const timeMatch = Math.abs((feature1.start_time || 0) - (feature2.start_time || 0)) < timeTolerance
      const amplitudeMatch = Math.abs(feature1.amplitude - feature2.amplitude) < amplitudeTolerance
      const durationMatch = Math.abs(feature1.duration - feature2.duration) < durationTolerance
      
      // 如果有峰值时间，也进行比较
      let peakTimeMatch = true
      if (feature1.peak_time !== undefined && feature2.peak_time !== undefined) {
        peakTimeMatch = Math.abs(feature1.peak_time - feature2.peak_time) < timeTolerance
      }
      
      return timeMatch && amplitudeMatch && durationMatch && peakTimeMatch
    }
    
    // 执行去重
    const uniqueFeatures = []
    let duplicateCount = 0
    
    allFeatures.value.forEach((feature, index) => {
      const isDuplicateFeature = uniqueFeatures.some(existingFeature => isDuplicate(feature, existingFeature))
      if (!isDuplicateFeature) {
        uniqueFeatures.push(feature)
      } else {
        duplicateCount++
        console.log(`去重：移除第${index + 1}个特征（重复）:`, {
          start_time: feature.start_time || (feature.start / params.fs),
          amplitude: feature.amplitude,
          duration: feature.duration
        })
      }
    })
    
    // 更新allFeatures列表
    allFeatures.value = uniqueFeatures
    
    const removedCount = originalCount - uniqueFeatures.length
    
    if (removedCount > 0) {
      ElMessage.success(`去重完成！原有 ${originalCount} 个特征，移除 ${removedCount} 个重复特征，剩余 ${uniqueFeatures.length} 个特征`)
      console.log(`手动去重完成：${originalCount} -> ${uniqueFeatures.length}，移除 ${removedCount} 个重复特征`)
    } else {
      ElMessage.info(`未发现重复特征，当前共有 ${uniqueFeatures.length} 个特征`)
      console.log('手动去重完成：未发现重复特征')
    }
    
  } catch (error) {
    console.error('去重操作失败:', error)
    ElMessage.error('去重操作失败: ' + (error.message || '未知错误'))
  } finally {
    deduplicateLoading.value = false
  }
}

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  if (chartInstance) {
    chartInstance.dispose()
  }
})

// 处理窗口大小变化
const handleResize = () => {
  if (chartInstance) {
    // 使用防抖来避免频繁调用resize
    clearTimeout(resizeTimer)
    resizeTimer = setTimeout(() => {
      chartInstance.resize()
    }, 100)
  }
}

// 防抖定时器
let resizeTimer = null
</script>

<style scoped>
.extraction {
  max-width: 1400px;
  margin: 0 auto;
}

.info-alert {
  margin-bottom: 20px;
}

.chart-instructions {
  margin-bottom: 15px;
}

.chart-container {
  width: 100%;
  height: 400px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  margin-bottom: 15px;
}

.time-range-info {
  margin-top: 15px;
}



.result-image {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  margin-top: 10px;
}

.card {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  border: 1px solid #e4e7ed;
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
  height: fit-content;
  position: sticky;
  top: 20px;
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

.upload-section {
  margin-bottom: 20px;
}

.extraction-section {
  margin-bottom: 20px;
}

.auto-controls {
  margin-bottom: 20px;
}

.interactive-controls {
  margin-bottom: 20px;
}

.auto-result {
  margin-top: 20px;
}

.interactive-result {
  margin-top: 20px;
}

.preview-image {
  margin-bottom: 20px;
  text-align: center;
}

.preview-table {
  width: 100%;
}

.preview-table h4 {
  margin-bottom: 10px;
  color: #2c3e50;
}

.preview-table .el-table {
  width: 100% !important;
}

.no-events {
  margin-top: 20px;
}

.batch-section {
  margin-bottom: 20px;
}

.batch-result {
  margin-top: 20px;
}

/* 交互式图表样式 */
.interactive-chart {
  margin-top: 20px;
}

.chart-header {
  margin-bottom: 15px;
}

.chart-header h4 {
  margin-bottom: 5px;
  color: #2c3e50;
}

.chart-instruction {
  color: #606266;
  font-size: 14px;
  margin: 0;
}

.chart-container {
  width: 100%;
  height: 400px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  margin-bottom: 15px;
}

.time-range-info {
  margin-top: 15px;
}





.mouse-coords {
  margin-top: 10px;
  margin-bottom: 10px;
  text-align: center;
  position: sticky;
  top: 0;
  background: white;
  z-index: 10;
  padding: 5px;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .preview-controls .el-col {
    margin-bottom: 10px;
  }
  
  .chart-container {
    height: 300px;
  }
  

}
</style>