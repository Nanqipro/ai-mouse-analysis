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
        2. <strong>批量提取</strong>: 使用找到的参数处理所有上传的文件。
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
                支持 .xlsx 和 .xls 格式，需包含 'dF' 工作表
              </div>
            </template>
          </el-upload>
        </div>
        
        <!-- 参数调试区域 -->
        <div v-if="fileList.length > 0" class="preview-section card">
          <h3 class="section-title">
            <el-icon><View /></el-icon>
            参数调试与单神经元可视化
          </h3>
          
          <div v-if="neuronColumns.length > 0" class="preview-controls">
            <el-row :gutter="20" align="middle">
              <el-col :span="12">
                <el-select
                  v-model="selectedNeuron"
                  placeholder="选择一个神经元进行预览"
                  style="width: 100%"
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
                  :loading="previewLoading"
                  :disabled="!selectedNeuron"
                  @click="generatePreview"
                  style="width: 100%"
                >
                  <el-icon><TrendCharts /></el-icon>
                  生成预览图
                </el-button>
              </el-col>
            </el-row>
          </div>
          
          <!-- 预览模式选择 -->
          <div v-if="neuronColumns.length > 0" class="preview-mode-selector">
            <el-radio-group v-model="previewMode" @change="onPreviewModeChange">
              <el-radio-button label="auto">自动检测</el-radio-button>
              <el-radio-button label="interactive">交互式选择</el-radio-button>
            </el-radio-group>
          </div>
          
          <!-- 预览结果 -->
          <div v-if="previewResult" class="preview-result">
            <div v-if="previewResult.plot" class="preview-image">
              <img :src="previewResult.plot" alt="预览图" class="result-image" />
            </div>
            
            <div v-if="previewResult.features && previewResult.features.length > 0" class="preview-table">
              <h4>检测到的事件特征</h4>
              <el-table :data="previewResult.features" stripe size="small" max-height="300">
                <el-table-column prop="amplitude" label="振幅" width="80" />
                <el-table-column prop="duration" label="持续时间" width="100">
                  <template #default="scope">
                    {{ scope.row.duration.toFixed(2) }}s
                  </template>
                </el-table-column>
                <el-table-column prop="fwhm" label="半高宽" width="80">
                  <template #default="scope">
                    {{ scope.row.fwhm ? scope.row.fwhm.toFixed(2) : 'N/A' }}s
                  </template>
                </el-table-column>
                <el-table-column prop="rise_time" label="上升时间" width="100">
                  <template #default="scope">
                    {{ scope.row.rise_time.toFixed(2) }}s
                  </template>
                </el-table-column>
                <el-table-column prop="decay_time" label="衰减时间" width="100">
                  <template #default="scope">
                    {{ scope.row.decay_time.toFixed(2) }}s
                  </template>
                </el-table-column>
                <el-table-column prop="auc" label="曲线下面积" width="120">
                  <template #default="scope">
                    {{ scope.row.auc.toFixed(2) }}
                  </template>
                </el-table-column>
                <el-table-column prop="snr" label="信噪比" width="80">
                  <template #default="scope">
                    {{ scope.row.snr.toFixed(2) }}
                  </template>
                </el-table-column>
              </el-table>
            </div>
            
            <div v-else class="no-events">
              <el-alert
                title="未检测到有效事件"
                type="warning"
                :closable="false"
                show-icon
              >
                请尝试调整参数后重新分析
              </el-alert>
            </div>
          </div>
          
          <!-- 交互式图表 -->
          <div v-if="previewMode === 'interactive' && interactiveData" class="interactive-chart">
            <div class="chart-header">
              <h4>交互式时间选择</h4>
              <p class="chart-instruction">
                在图表上拖拽选择时间范围，然后点击"提取选定范围"按钮进行分析
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
                  1. 使用图表下方的滑动条缩放到您想要分析的时间范围<br/>
                  2. 点击"确认当前显示范围"按钮来选择时间范围<br/>
                  3. 选择完成后，点击"提取选定范围的事件"按钮进行分析
                </template>
              </el-alert>
            </div>
            
            <div ref="chartContainer" class="chart-container"></div>
            
            <div class="chart-controls">
              <el-button
                type="primary"
                @click="confirmCurrentRange"
                style="width: 100%; margin-bottom: 10px"
              >
                <el-icon><Select /></el-icon>
                确认当前显示范围为选择范围
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
          
          <!-- 手动提取结果 -->
          <div v-if="manualExtractResult" class="manual-extract-result">
            <el-alert
              :title="manualExtractResult.message"
              type="success"
              :closable="false"
              show-icon
            />
            
            <div v-if="manualExtractResult.plot" class="preview-image">
              <img :src="manualExtractResult.plot" alt="手动提取结果" class="result-image" />
            </div>
            
            <div v-if="manualExtractResult.features && manualExtractResult.features.length > 0" class="preview-table">
              <h4>选定范围内的事件特征</h4>
              <el-table :data="manualExtractResult.features" stripe size="small" max-height="300">
                <el-table-column prop="amplitude" label="振幅" width="80" />
                <el-table-column prop="duration" label="持续时间" width="100">
                  <template #default="scope">
                    {{ scope.row.duration.toFixed(2) }}s
                  </template>
                </el-table-column>
                <el-table-column prop="start_time" label="开始时间" width="100">
                  <template #default="scope">
                    {{ scope.row.start_time ? scope.row.start_time.toFixed(2) : 'N/A' }}s
                  </template>
                </el-table-column>
                <el-table-column prop="peak_time" label="峰值时间" width="100">
                  <template #default="scope">
                    {{ scope.row.peak_time ? scope.row.peak_time.toFixed(2) : 'N/A' }}s
                  </template>
                </el-table-column>
                <el-table-column prop="end_time" label="结束时间" width="100">
                  <template #default="scope">
                    {{ scope.row.end_time ? scope.row.end_time.toFixed(2) : 'N/A' }}s
                  </template>
                </el-table-column>
                <el-table-column prop="snr" label="信噪比" width="80">
                  <template #default="scope">
                    {{ scope.row.snr.toFixed(2) }}
                  </template>
                </el-table-column>
              </el-table>
            </div>
          </div>
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
import { ref, reactive, nextTick, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
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
  Select
} from '@element-plus/icons-vue'
import { extractionAPI, downloadAPI } from '@/api'
import * as echarts from 'echarts'

// 响应式数据
const uploadRef = ref()
const fileList = ref([])
const neuronColumns = ref([])
const selectedNeuron = ref('')
const previewLoading = ref(false)
const batchLoading = ref(false)
const previewResult = ref(null)
const batchResult = ref(null)

// 交互式图表相关
const previewMode = ref('auto')
const chartContainer = ref()
const interactiveData = ref(null)
const selectedTimeRange = ref(null)
const manualExtractLoading = ref(false)
const manualExtractResult = ref(null)
let chartInstance = null

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
const handleFileChange = (file, files) => {
  fileList.value = files
  // 重置相关状态
  neuronColumns.value = []
  selectedNeuron.value = ''
  previewResult.value = null
  batchResult.value = null
  
  // 如果有文件，尝试获取神经元列
  if (files.length > 0) {
    loadNeuronColumns(files[0])
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
      }
    }
  } catch (error) {
    console.error('加载神经元列失败:', error)
  }
}

// 生成预览
const generatePreview = async () => {
  if (!selectedNeuron.value || fileList.value.length === 0) {
    ElMessage.warning('请选择神经元')
    return
  }
  
  previewLoading.value = true
  try {
    if (previewMode.value === 'auto') {
      // 自动检测模式
      const formData = new FormData()
      formData.append('file', fileList.value[0].raw)
      formData.append('fs', params.fs)
      formData.append('min_duration_frames', params.min_duration_frames)
      formData.append('max_duration_frames', params.max_duration_frames)
      formData.append('min_snr', params.min_snr)
      formData.append('smooth_window', params.smooth_window)
      formData.append('peak_distance_frames', params.peak_distance_frames)
      formData.append('filter_strength', params.filter_strength)
      formData.append('neuron_id', selectedNeuron.value)
      
      const response = await extractionAPI.preview(formData)
      if (response.success) {
        previewResult.value = response
        ElMessage.success('预览生成成功')
      }
    } else {
      // 交互式模式
      await loadInteractiveData()
      ElMessage.success('交互式图表加载成功')
    }
  } catch (error) {
    console.error('预览生成失败:', error)
  } finally {
    previewLoading.value = false
  }
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
  }
}

// 预览模式切换
const onPreviewModeChange = async (mode) => {
  if (mode === 'interactive' && selectedNeuron.value && fileList.value.length > 0) {
    await loadInteractiveData()
  } else {
    // 清理交互式数据
    interactiveData.value = null
    selectedTimeRange.value = null
    manualExtractResult.value = null
    if (chartInstance) {
      chartInstance.dispose()
      chartInstance = null
    }
  }
}

// 加载交互式数据
const loadInteractiveData = async () => {
  if (!selectedNeuron.value || fileList.value.length === 0) {
    return
  }
  
  try {
    const formData = new FormData()
    formData.append('file', fileList.value[0].raw)
    formData.append('fs', params.fs)
    formData.append('neuron_id', selectedNeuron.value)
    
    const response = await extractionAPI.getInteractiveData(formData)
    console.log('API响应:', response)
    if (response.success) {
      interactiveData.value = response.data
      console.log('设置的交互式数据:', interactiveData.value)
      await nextTick()
      initChart()
    } else {
      console.error('API返回失败:', response)
      ElMessage.error('获取交互式数据失败')
    }
  } catch (error) {
    console.error('加载交互式数据失败:', error)
  }
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
      }
    },
    toolbox: {
      show: true,
      feature: {
        dataZoom: {
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
    dataZoom: [{
      type: 'slider',
      show: true,
      xAxisIndex: [0],
      start: 0,
      end: 100,
      bottom: 10,
      height: 20,
      handleIcon: 'M10.7,11.9v-1.3H9.3v1.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4v1.3h1.3v-1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23.1h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
      handleSize: '80%',
      handleStyle: {
        color: '#fff',
        shadowBlur: 3,
        shadowColor: 'rgba(0, 0, 0, 0.6)',
        shadowOffsetX: 2,
        shadowOffsetY: 2
      }
    }],
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
      symbol: 'none',
      lineStyle: {
        width: 1
      }
    }]
  }
  
  chartInstance.setOption(option)
  
  // 监听dataZoom事件
  chartInstance.on('dataZoom', function(params) {
    console.log('dataZoom事件触发:', params)
    
    // 获取当前缩放的范围
    const option = chartInstance.getOption()
    const dataZoom = option.dataZoom[0]
    
    if (dataZoom && dataZoom.startValue !== undefined && dataZoom.endValue !== undefined) {
      const startTime = dataZoom.startValue
      const endTime = dataZoom.endValue
      
      selectedTimeRange.value = {
        start: parseFloat(startTime),
        end: parseFloat(endTime)
      }
      console.log('设置选择范围:', selectedTimeRange.value)
      ElMessage.success(`已选择时间范围: ${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s`)
    }
  })
  
  // 添加一个按钮来手动获取当前显示范围
  window.getCurrentTimeRange = function() {
    if (chartInstance) {
      const option = chartInstance.getOption()
      const xAxis = option.xAxis[0]
      const dataZoom = option.dataZoom[0]
      
      if (dataZoom && dataZoom.startValue !== undefined && dataZoom.endValue !== undefined) {
        const startTime = dataZoom.startValue
        const endTime = dataZoom.endValue
        
        selectedTimeRange.value = {
          start: parseFloat(startTime),
          end: parseFloat(endTime)
        }
        console.log('手动获取选择范围:', selectedTimeRange.value)
        ElMessage.success(`已选择时间范围: ${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s`)
      }
    }
  }
}

// 确认当前显示范围
const confirmCurrentRange = () => {
  if (chartInstance) {
    const option = chartInstance.getOption()
    const dataZoom = option.dataZoom[0]
    
    if (dataZoom && dataZoom.startValue !== undefined && dataZoom.endValue !== undefined) {
      const startTime = dataZoom.startValue
      const endTime = dataZoom.endValue
      
      selectedTimeRange.value = {
        start: parseFloat(startTime),
        end: parseFloat(endTime)
      }
      console.log('确认选择范围:', selectedTimeRange.value)
      ElMessage.success(`已选择时间范围: ${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s`)
    } else {
      ElMessage.warning('请先使用下方滑动条调整显示范围')
    }
  }
}

// 提取选定范围的事件
const extractSelectedRange = async () => {
  if (!selectedTimeRange.value || !selectedNeuron.value || fileList.value.length === 0) {
    ElMessage.warning('请先选择时间范围')
    return
  }
  
  manualExtractLoading.value = true
  try {
    const formData = new FormData()
    formData.append('file', fileList.value[0].raw)
    formData.append('fs', params.fs)
    formData.append('neuron_id', selectedNeuron.value)
    formData.append('start_time', selectedTimeRange.value.start)
    formData.append('end_time', selectedTimeRange.value.end)
    formData.append('min_duration_frames', params.min_duration_frames)
    formData.append('max_duration_frames', params.max_duration_frames)
    formData.append('min_snr', params.min_snr)
    formData.append('smooth_window', params.smooth_window)
    formData.append('peak_distance_frames', params.peak_distance_frames)
    formData.append('filter_strength', params.filter_strength)
    
    const response = await extractionAPI.manualExtract(formData)
    if (response.success) {
      manualExtractResult.value = response
      ElMessage.success('手动提取完成')
    }
  } catch (error) {
    console.error('手动提取失败:', error)
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

.manual-extract-result {
  margin-top: 20px;
}

.result-image {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  margin-top: 10px;
}

.param-help {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}

.upload-section {
  margin-bottom: 20px;
}

.preview-section {
  margin-bottom: 20px;
}

.preview-controls {
  margin-bottom: 20px;
}

.preview-result {
  margin-top: 20px;
}

.preview-image {
  margin-bottom: 20px;
  text-align: center;
}

.preview-table h4 {
  margin-bottom: 10px;
  color: #2c3e50;
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
.preview-mode-selector {
  margin-bottom: 20px;
  text-align: center;
}

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

.manual-extract-result {
  margin-top: 20px;
  padding: 15px;
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  background-color: #fafafa;
}

.manual-extract-result .preview-image {
  margin-top: 15px;
}

.manual-extract-result .preview-table {
  margin-top: 15px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .preview-controls .el-col {
    margin-bottom: 10px;
  }
  
  .chart-container {
    height: 300px;
  }
  
  .preview-mode-selector {
    margin-bottom: 15px;
  }
}
</style>