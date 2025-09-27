<template>
  <div class="trace">
    <h1 class="page-title">
      <el-icon><DataLine /></el-icon>
      神经元活动Trace图
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
        1. <strong>上传数据文件</strong>: 支持Excel格式的神经元活动数据。<br>
        2. <strong>配置参数</strong>: 设置时间范围、排序方式、显示参数等。<br>
        3. <strong>生成Trace图</strong>: 可视化神经元活动的时间序列模式。
      </template>
    </el-alert>

    <el-row :gutter="20">
      <!-- 左侧参数面板 -->
      <el-col :xs="24" :sm="24" :md="10" :lg="8">
        <div class="params-panel card">
          <h3 class="section-title">
            <el-icon><Setting /></el-icon>
            Trace图参数
          </h3>
          
          <el-form :model="params" label-width="120px" size="small">

            <!-- 时间范围设置 -->
            <el-form-item label="时间范围">
              <el-row :gutter="10">
                <el-col :span="12">
                  <el-input-number
                    v-model="params.stamp_min"
                    :min="0"
                    :precision="1"
                    placeholder="开始时间"
                    style="width: 100%"
                  />
                </el-col>
                <el-col :span="12">
                  <el-input-number
                    v-model="params.stamp_max"
                    :min="0"
                    :precision="1"
                    placeholder="结束时间"
                    style="width: 100%"
                  />
                </el-col>
              </el-row>
              <div class="param-help">留空表示使用全部时间范围</div>
            </el-form-item>

            <!-- 排序方式 -->
            <el-form-item label="排序方式">
              <el-select v-model="params.sort_method" style="width: 100%" @change="handleSortMethodChange">
                <el-option label="原始顺序" value="original" />
                <el-option label="峰值时间排序" value="peak" />
                <el-option label="钙波时间排序" value="calcium_wave" />
                <el-option label="自定义顺序" value="custom" />
              </el-select>
              <div class="param-help">选择神经元的排序方式</div>
            </el-form-item>

            <!-- 自定义神经元顺序 -->
            <el-form-item v-if="params.sort_method === 'custom'" label="自定义顺序">
              <el-input
                v-model="customOrderText"
                type="textarea"
                :rows="4"
                placeholder="请输入神经元ID，用逗号分隔，例如：n1,n2,n3"
                @blur="updateCustomOrder"
              />
              <div class="param-help">用逗号分隔神经元ID</div>
            </el-form-item>

            <!-- 显示参数 -->
            <el-divider content-position="left">显示参数</el-divider>
            
            <el-form-item label="最大神经元数">
              <el-input-number
                v-model="params.max_neurons"
                :min="10"
                :max="200"
                :step="10"
                style="width: 100%"
              />
              <div class="param-help">限制显示的神经元数量</div>
            </el-form-item>

            <el-form-item label="垂直偏移">
              <el-input-number
                v-model="params.trace_offset"
                :min="20"
                :max="200"
                :step="10"
                :precision="1"
                style="width: 100%"
              />
              <div class="param-help">神经元间的垂直间距</div>
            </el-form-item>

            <el-form-item label="信号缩放">
              <el-input-number
                v-model="params.scaling_factor"
                :min="10"
                :max="200"
                :step="10"
                :precision="1"
                style="width: 100%"
              />
              <div class="param-help">信号振幅的缩放因子</div>
            </el-form-item>

            <el-form-item label="线条宽度">
              <el-input-number
                v-model="params.line_width"
                :min="0.5"
                :max="5.0"
                :step="0.5"
                :precision="1"
                style="width: 100%"
              />
              <div class="param-help">Trace线条的宽度</div>
            </el-form-item>

            <el-form-item label="线条透明度">
              <el-slider
                v-model="params.trace_alpha"
                :min="0.1"
                :max="1.0"
                :step="0.1"
                :precision="1"
                show-input
                :show-input-controls="false"
              />
              <div class="param-help">Trace线条的透明度</div>
            </el-form-item>

            <!-- 采样率 -->
            <el-form-item label="采样率">
              <el-input-number
                v-model="params.sampling_rate"
                :min="1.0"
                :max="20.0"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
              <div class="param-help">数据采样频率 (Hz)</div>
            </el-form-item>

            <!-- 钙波检测参数 -->
            <el-divider content-position="left">钙波检测参数</el-divider>
            
            <el-form-item label="钙波阈值">
              <el-input-number
                v-model="params.calcium_wave_threshold"
                :min="0.5"
                :max="5.0"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
              <div class="param-help">钙波检测的阈值倍数</div>
            </el-form-item>

            <el-form-item label="最小突出度">
              <el-input-number
                v-model="params.min_prominence"
                :min="0.1"
                :max="5.0"
                :step="0.1"
                :precision="1"
                style="width: 100%"
              />
              <div class="param-help">峰值的最小突出度</div>
            </el-form-item>

            <el-form-item label="最小上升速率">
              <el-input-number
                v-model="params.min_rise_rate"
                :min="0.01"
                :max="1.0"
                :step="0.01"
                :precision="2"
                style="width: 100%"
              />
              <div class="param-help">钙波的最小上升速率</div>
            </el-form-item>

            <el-form-item label="最大下降速率">
              <el-input-number
                v-model="params.max_fall_rate"
                :min="0.01"
                :max="1.0"
                :step="0.01"
                :precision="2"
                style="width: 100%"
              />
              <div class="param-help">钙波的最大下降速率</div>
            </el-form-item>

            <!-- 操作按钮 -->
            <el-form-item>
              <el-button
                type="primary"
                :loading="loading"
                :disabled="!selectedFile"
                @click="startTraceAnalysis"
                style="width: 100%"
              >
                <el-icon><DataLine /></el-icon>
                生成Trace图
              </el-button>
            </el-form-item>
          </el-form>
        </div>
      </el-col>

      <!-- 右侧结果展示 -->
      <el-col :xs="24" :sm="24" :md="14" :lg="16">
        <div class="results-panel card">
          <h3 class="section-title">
            <el-icon><Picture /></el-icon>
            Trace图结果
          </h3>

          <!-- 加载状态 -->
          <div v-if="loading" class="loading-container">
            <el-progress
              :percentage="progress"
              :status="progressStatus"
              :stroke-width="8"
            />
            <p class="loading-text">{{ loadingText }}</p>
          </div>

          <!-- 错误信息 -->
          <el-alert
            v-if="error"
            :title="error"
            type="error"
            :closable="true"
            @close="error = ''"
            class="error-alert"
          />

          <!-- 结果展示 -->
          <div v-if="result && !loading" class="result-container">
            <!-- 图像展示 -->
            <div class="image-container">
              <img
                :src="result.image"
                alt="Trace图"
                class="trace-image"
                @load="onImageLoad"
              />
            </div>

            <!-- 分析信息 -->
            <div class="info-container">
              <el-descriptions title="分析信息" :column="2" border>
                <el-descriptions-item label="排序方式">
                  {{ getSortMethodName(result.info.sort_method) }}
                </el-descriptions-item>
                <el-descriptions-item label="总神经元数">
                  {{ result.info.total_neurons }}
                </el-descriptions-item>
                <el-descriptions-item label="显示神经元数">
                  {{ result.info.displayed_neurons }}
                </el-descriptions-item>
                <el-descriptions-item label="时间范围">
                  {{ formatTimeRange(result.info.time_range) }}
                </el-descriptions-item>
                <el-descriptions-item label="持续时间">
                  {{ result.info.time_range.duration_seconds.toFixed(1) }} 秒
                </el-descriptions-item>
                <el-descriptions-item label="行为类型数">
                  {{ result.info.behavior_types.length }}
                </el-descriptions-item>
                <el-descriptions-item label="行为事件数">
                  {{ result.info.total_behavior_events }}
                </el-descriptions-item>
                <el-descriptions-item label="行为类型">
                  <el-tag
                    v-for="behavior in result.info.behavior_types"
                    :key="behavior"
                    size="small"
                    style="margin-right: 5px; margin-bottom: 5px;"
                  >
                    {{ behavior }}
                  </el-tag>
                </el-descriptions-item>
              </el-descriptions>
            </div>

            <!-- 峰值时间信息 -->
            <div v-if="Object.keys(result.info.peak_times).length > 0" class="peak-times-container">
              <h4>峰值时间信息</h4>
              <el-table
                :data="peakTimesData"
                size="small"
                max-height="300"
                style="width: 100%"
              >
                <el-table-column prop="neuron" label="神经元" width="100" />
                <el-table-column prop="peak_time" label="峰值时间(秒)" width="120" />
                <el-table-column prop="peak_stamp" label="峰值时间戳" width="120" />
              </el-table>
            </div>
          </div>

          <!-- 空状态 -->
          <div v-if="!result && !loading && !error" class="empty-state">
            <!-- 文件上传区域 -->
            <div class="upload-section">
              <h3 class="section-title">
                <el-icon><Upload /></el-icon>
                数据文件上传
              </h3>
              <el-upload
                ref="uploadRef"
                class="upload-demo"
                drag
                :auto-upload="false"
                :on-change="handleFileChange"
                :show-file-list="false"
                accept=".xlsx,.xls"
              >
                <el-icon class="el-icon--upload"><upload-filled /></el-icon>
                <div class="el-upload__text">
                  将 Excel 文件拖拽到此处，或<em>点击上传</em>
                </div>
                <template #tip>
                  <div class="el-upload__tip">
                    支持 .xlsx 和 .xls 格式，需包含神经元活动数据
                  </div>
                </template>
              </el-upload>
              <div v-if="selectedFile" class="file-info">
                <el-tag type="success" size="small">
                  {{ selectedFile.name }}
                </el-tag>
              </div>
            </div>
            
            <div class="empty-message">
              <el-empty description="请上传数据文件并配置参数，然后点击生成Trace图" />
            </div>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { TrendCharts, Setting, Picture, UploadFilled } from '@element-plus/icons-vue'
import { traceAPI } from '@/api'

// 响应式数据
const loading = ref(false)
const progress = ref(0)
const progressStatus = ref('')
const loadingText = ref('')
const error = ref('')
const result = ref(null)
const selectedFile = ref(null)
const uploadRef = ref(null)
const customOrderText = ref('')

// 参数配置
const params = reactive({
  stamp_min: null,
  stamp_max: null,
  sort_method: 'peak',
  custom_neuron_order: [],
  trace_offset: 60.0,
  scaling_factor: 80.0,
  max_neurons: 60,
  trace_alpha: 0.8,
  line_width: 2.0,
  sampling_rate: 4.8,
  calcium_wave_threshold: 1.5,
  min_prominence: 1.0,
  min_rise_rate: 0.1,
  max_fall_rate: 0.05
})

// 计算属性
const peakTimesData = computed(() => {
  if (!result.value || !result.value.info.peak_times) return []
  
  return Object.entries(result.value.info.peak_times).map(([neuron, time]) => ({
    neuron,
    peak_time: (time / params.sampling_rate).toFixed(2),
    peak_stamp: time.toFixed(1)
  }))
})

// 方法
const handleFileChange = (file) => {
  selectedFile.value = file.raw
  error.value = ''
  result.value = null
}

const handleSortMethodChange = () => {
  if (params.sort_method !== 'custom') {
    customOrderText.value = ''
    params.custom_neuron_order = []
  }
}

const updateCustomOrder = () => {
  if (params.sort_method === 'custom' && customOrderText.value.trim()) {
    params.custom_neuron_order = customOrderText.value
      .split(',')
      .map(item => item.trim())
      .filter(item => item)
  }
}

const getSortMethodName = (method) => {
  const names = {
    'original': '原始顺序',
    'peak': '峰值时间排序',
    'calcium_wave': '钙波时间排序',
    'custom': '自定义顺序'
  }
  return names[method] || method
}

const formatTimeRange = (timeRange) => {
  return `${timeRange.start_seconds.toFixed(1)}s - ${timeRange.end_seconds.toFixed(1)}s`
}

const onImageLoad = () => {
  ElMessage.success('Trace图加载完成')
}

const startTraceAnalysis = async () => {
  if (!selectedFile.value) {
    ElMessage.warning('请先选择数据文件')
    return
  }

  loading.value = true
  progress.value = 0
  progressStatus.value = ''
  error.value = ''
  result.value = null

  let progressInterval = null

  try {
    // 更新自定义顺序
    updateCustomOrder()

    // 创建FormData
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    
    // 添加参数
    if (params.stamp_min !== null) {
      formData.append('stamp_min', params.stamp_min)
    }
    if (params.stamp_max !== null) {
      formData.append('stamp_max', params.stamp_max)
    }
    formData.append('sort_method', params.sort_method)
    if (params.custom_neuron_order.length > 0) {
      formData.append('custom_neuron_order', JSON.stringify(params.custom_neuron_order))
    }
    formData.append('trace_offset', params.trace_offset)
    formData.append('scaling_factor', params.scaling_factor)
    formData.append('max_neurons', params.max_neurons)
    formData.append('trace_alpha', params.trace_alpha)
    formData.append('line_width', params.line_width)
    formData.append('sampling_rate', params.sampling_rate)
    formData.append('calcium_wave_threshold', params.calcium_wave_threshold)
    formData.append('min_prominence', params.min_prominence)
    formData.append('min_rise_rate', params.min_rise_rate)
    formData.append('max_fall_rate', params.max_fall_rate)

    // 模拟进度更新
    progressInterval = setInterval(() => {
      if (progress.value < 90) {
        progress.value += Math.random() * 10
        if (progress.value < 30) {
          loadingText.value = '正在加载数据...'
        } else if (progress.value < 60) {
          loadingText.value = '正在分析神经元活动...'
        } else if (progress.value < 90) {
          loadingText.value = '正在生成Trace图...'
        }
      }
    }, 200)

    // 调用API
    const response = await traceAPI.analyze(formData)
    
    if (progressInterval) {
      clearInterval(progressInterval)
    }
    progress.value = 100
    progressStatus.value = 'success'
    loadingText.value = '分析完成'

    if (response.success) {
      result.value = response
      ElMessage.success('Trace图生成成功')
    } else {
      throw new Error('分析失败')
    }

  } catch (err) {
    if (progressInterval) {
      clearInterval(progressInterval)
    }
    progress.value = 0
    progressStatus.value = 'exception'
    loadingText.value = '分析失败'
    error.value = err.message || 'Trace图分析失败'
    ElMessage.error('Trace图分析失败: ' + (err.message || '未知错误'))
  } finally {
    setTimeout(() => {
      loading.value = false
    }, 1000)
  }
}

// 生命周期
onMounted(() => {
  // 初始化
})
</script>

<style scoped>
.trace {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.page-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
  color: #303133;
  font-size: 28px;
  font-weight: bold;
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
  color: #303133;
  font-size: 18px;
  font-weight: bold;
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

.file-info {
  margin-top: 10px;
}

.loading-container {
  text-align: center;
  padding: 40px 20px;
}

.loading-text {
  margin-top: 15px;
  color: #606266;
  font-size: 14px;
}

.error-alert {
  margin-bottom: 20px;
}

.result-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.image-container {
  text-align: center;
  background: #f8f9fa;
  border-radius: 8px;
  padding: 20px;
  border: 1px solid #e4e7ed;
}

.trace-image {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
}

.info-container {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  border: 1px solid #e4e7ed;
}

.peak-times-container {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  border: 1px solid #e4e7ed;
}

.peak-times-container h4 {
  margin-bottom: 15px;
  color: #303133;
  font-size: 16px;
  font-weight: bold;
}

.empty-state {
  text-align: center;
  padding: 40px 20px;
}

.upload-section {
  margin-bottom: 20px;
}

.file-info {
  margin-top: 10px;
}

.empty-message {
  margin-top: 20px;
}

.card {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  border: 1px solid #e4e7ed;
}

.results-panel {
  min-height: 600px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .trace {
    padding: 10px;
  }
  
  .page-title {
    font-size: 24px;
  }
  
  .params-panel {
    position: static;
    margin-bottom: 20px;
  }
}
</style>
