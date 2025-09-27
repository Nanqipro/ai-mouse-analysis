<template>
  <div class="neuron-analysis">
    <el-container>
      <!-- 页面标题 -->
      <el-header class="page-header">
        <h1 class="page-title">
          <el-icon><DataAnalysis /></el-icon>
          神经元分析
        </h1>
        <p class="page-description">
          两步式工作流程：效应量分析 → 位置标记 → 综合分析
        </p>
      </el-header>

      <el-container>
        <!-- 左侧工作流程面板 -->
        <el-aside :width="sidebarWidth" class="workflow-panel">
          <!-- 工作流程步骤指示器 -->
          <el-card class="workflow-card">
            <template #header>
              <div class="card-header">
                <el-icon><List /></el-icon>
                <span>工作流程</span>
              </div>
            </template>
            
            <el-steps :active="currentStep" direction="vertical" finish-status="success">
              <el-step 
                title="效应量分析" 
                description="计算神经元与行为的效应量"
                :status="getStepStatus(0)"
              />
              <el-step 
                title="位置标记" 
                description="标记神经元空间位置"
                :status="getStepStatus(1)"
              />
              <el-step 
                title="综合分析" 
                description="结合效应量和位置数据"
                :status="getStepStatus(2)"
              />
            </el-steps>
          </el-card>

          <!-- 当前步骤参数面板 -->
          <el-card class="parameter-card">
            <template #header>
              <div class="card-header">
                <el-icon><Setting /></el-icon>
                <span>{{ getCurrentStepTitle() }}</span>
              </div>
            </template>

            <!-- 步骤1：效应量分析 -->
            <div v-if="currentStep === 0" class="step-content">
              <el-form :model="effectSizeForm" label-width="120px" class="analysis-form">
                <el-form-item label="行为列名">
                  <el-input
                    v-model="effectSizeForm.behaviorColumn"
                    placeholder="留空则使用最后一列"
                    clearable
                  />
                </el-form-item>

                <el-form-item label="效应量阈值">
                  <el-slider
                    v-model="effectSizeForm.threshold"
                    :min="0.1"
                    :max="2.0"
                    :step="0.1"
                    show-input
                    :format-tooltip="formatThresholdTooltip"
                  />
                </el-form-item>

                <el-form-item>
                  <el-button
                    type="primary"
                    :loading="loading"
                    :disabled="!effectSizeForm.file"
                    @click="startEffectSizeAnalysis"
                    class="analysis-button"
                  >
                    <el-icon><DataAnalysis /></el-icon>
                    开始效应量分析
                  </el-button>
                </el-form-item>
              </el-form>
            </div>

            <!-- 步骤2：位置标记 -->
            <div v-if="currentStep === 1" class="step-content">
              <el-form :model="positionForm" label-width="120px" class="analysis-form">
                <el-form-item label="起始编号">
                  <el-input-number
                    v-model="positionForm.startNumber"
                    :min="1"
                    :max="1000"
                    controls-position="right"
                  />
                </el-form-item>

                <el-form-item>
                  <el-button
                    type="primary"
                    @click="startPositionMarking"
                    class="analysis-button"
                  >
                    <el-icon><Location /></el-icon>
                    开始位置标记
                  </el-button>
                </el-form-item>
              </el-form>
            </div>

            <!-- 步骤3：综合分析 -->
            <div v-if="currentStep === 2" class="step-content">
              <el-form :model="analysisForm" label-width="120px" class="analysis-form">
                <!-- 单行为空间分析参数 -->
                <div class="behavior-analysis-params">
                  <el-form-item label="选择行为">
                    <el-select v-model="analysisForm.selectedBehavior" placeholder="选择要分析的行为">
                      <el-option 
                        v-for="behavior in getAvailableBehaviors()" 
                        :key="behavior" 
                        :label="behavior" 
                        :value="behavior" 
                      />
                    </el-select>
                  </el-form-item>
                  
                  <el-form-item label="神经元类型">
                    <el-radio-group v-model="analysisForm.neuronType">
                      <el-radio label="all">所有神经元</el-radio>
                      <el-radio label="key">关键神经元</el-radio>
                    </el-radio-group>
                  </el-form-item>
                  
                  <el-form-item label="显示选项">
                    <el-checkbox-group v-model="analysisForm.displayOptions">
                      <el-checkbox label="show-effect-values">显示效应量数值</el-checkbox>
                      <el-checkbox label="show-neuron-ids">显示神经元ID</el-checkbox>
                      <el-checkbox label="show-colorbar">显示颜色条</el-checkbox>
                    </el-checkbox-group>
                  </el-form-item>
                </div>

                <el-form-item>
                  <el-button
                    type="primary"
                    :loading="loading"
                    :disabled="!canStartComprehensiveAnalysis"
                    @click="startComprehensiveAnalysis"
                    class="analysis-button"
                  >
                    <el-icon><DataAnalysis /></el-icon>
                    开始综合分析
                  </el-button>
                </el-form-item>
              </el-form>
            </div>

            <!-- 通用操作按钮 -->
            <div class="workflow-controls">
              <el-button @click="resetWorkflow" :disabled="loading">
                <el-icon><Refresh /></el-icon>
                重置工作流程
              </el-button>
              <el-button 
                v-if="currentStep > 0" 
                @click="goToPreviousStep" 
                :disabled="loading"
              >
                <el-icon><ArrowLeft /></el-icon>
                上一步
              </el-button>
              <el-button 
                v-if="currentStep < 2 && canProceedToNextStep" 
                @click="goToNextStep" 
                type="primary"
                :disabled="loading"
              >
                下一步
                <el-icon><ArrowRight /></el-icon>
              </el-button>
            </div>
          </el-card>
        </el-aside>

        <!-- 右侧结果面板 -->
        <el-main class="results-panel">
          <!-- 步骤1：数据文件上传 -->
          <el-card v-if="currentStep === 0 && !effectSizeResult" class="upload-card">
            <template #header>
              <div class="card-header">
                <el-icon><Upload /></el-icon>
                <span>数据文件上传</span>
              </div>
            </template>
            
            <div class="upload-section">
              <el-upload
                ref="uploadRef"
                class="upload-demo"
                drag
                :auto-upload="false"
                :on-change="handleFileChange"
                :before-remove="handleFileRemove"
                :limit="1"
                accept=".xlsx,.xls,.csv"
              >
                <el-icon class="el-icon--upload"><upload-filled /></el-icon>
                <div class="el-upload__text">
                  将 Excel/CSV 文件拖拽到此处，或<em>点击上传</em>
                </div>
                <template #tip>
                  <div class="el-upload__tip">
                    支持 .xlsx、.xls 和 .csv 格式，需包含神经元活动数据和行为标签
                  </div>
                </template>
              </el-upload>
            </div>
          </el-card>

          <!-- 步骤1：效应量分析结果 -->
          <div v-if="currentStep === 0 && effectSizeResult" class="step-results">
            <el-card class="result-card">
              <template #header>
                <div class="card-header">
                  <el-icon><DataAnalysis /></el-icon>
                  <span>效应量分析结果</span>
                </div>
              </template>
              
              <!-- 效应量分析摘要 -->
              <div class="summary-content">
                <el-row :gutter="20">
                  <el-col :span="6" v-for="(value, key) in getEffectSizeSummary()" :key="key">
                    <div class="summary-item">
                      <div class="summary-label">{{ getSummaryLabel(key) }}</div>
                      <div class="summary-value">{{ value }}</div>
                    </div>
                  </el-col>
                </el-row>
              </div>

              <!-- 效应量可视化 -->
              <div class="visualization-content">
                <el-tabs v-model="effectSizeActiveTab" type="card">
                  <el-tab-pane label="效应量矩阵" name="key-neurons">
                    <div class="matrix-container">
                      <div class="matrix-header">
                        <h4>行为类型 × 神经元ID 效应量矩阵</h4>
                        <p class="matrix-description">
                          显示每个神经元对每种行为的效应量值，超过阈值 {{ effectSizeForm.threshold }} 的数值将标红显示
                        </p>
                      </div>
                      <el-table 
                        :data="formatEffectSizesMatrix()" 
                        style="width: 100%" 
                        max-height="500"
                        border
                        stripe
                        class="effect-matrix-table"
                      >
                        <el-table-column prop="neuron" label="神经元ID" width="120" fixed="left">
                          <template #default="{ row }">
                            <span class="neuron-id">{{ row.neuron }}</span>
                          </template>
                        </el-table-column>
                        <el-table-column 
                          v-for="behavior in getBehaviors()" 
                          :key="behavior"
                          :prop="behavior" 
                          :label="behavior" 
                          width="140"
                          align="center"
                        >
                          <template #default="{ row }">
                            <span 
                              :class="getEffectSizeClass(row[behavior], behavior)"
                              class="effect-size-value"
                            >
                              {{ formatEffectSizeValue(row[behavior]) }}
                            </span>
                          </template>
                        </el-table-column>
                      </el-table>
                      
                      <!-- 图例说明 -->
                      <div class="matrix-legend">
                        <h5>图例说明：</h5>
                        <div class="legend-items">
                          <div class="legend-item">
                            <span class="legend-color high-effect"></span>
                            <span>高效应量 (≥ {{ effectSizeForm.threshold }})</span>
                          </div>
                          <div class="legend-item">
                            <span class="legend-color medium-effect"></span>
                            <span>中等效应量 (0.2 - {{ effectSizeForm.threshold }})</span>
                          </div>
                          <div class="legend-item">
                            <span class="legend-color low-effect"></span>
                            <span>低效应量 (< 0.2)</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </el-tab-pane>
                </el-tabs>
              </div>
            </el-card>
          </div>

          <!-- 步骤2：位置标记工具 -->
          <div v-if="currentStep === 1" class="step-results">
            <el-card class="result-card">
              <template #header>
                <div class="card-header">
                  <el-icon><Location /></el-icon>
                  <span>神经元位置标记</span>
                </div>
              </template>
              <PositionMarker 
                @positions-updated="handlePositionsUpdated"
                ref="positionMarkerRef"
              />
            </el-card>
          </div>

          <!-- 步骤3：综合分析结果 -->
          <div v-if="currentStep === 2 && comprehensiveResult" class="step-results">
            <el-card class="result-card">
              <template #header>
                <div class="card-header">
                  <el-icon><DataAnalysis /></el-icon>
                  <span>综合分析结果</span>
                </div>
              </template>
              
              <!-- 综合分析摘要 -->
              <div class="summary-content">
                <el-row :gutter="20">
                  <el-col :span="6" v-for="(value, key) in getComprehensiveSummary()" :key="key">
                    <div class="summary-item">
                      <div class="summary-label">{{ getSummaryLabel(key) }}</div>
                      <div class="summary-value">{{ value }}</div>
                    </div>
                  </el-col>
                </el-row>
              </div>

              <!-- 综合分析可视化 -->
              <div class="visualization-content">
                <div class="image-container">
                  <img 
                    v-if="comprehensiveResult.single_behavior_spatial_plot"
                    :src="comprehensiveResult.single_behavior_spatial_plot" 
                    alt="单行为空间分析图"
                    class="result-image"
                  />
                </div>
              </div>

              <!-- 详细数据 -->
              <div class="data-content">
                <el-tabs v-model="comprehensiveDataTab" type="card">
                  <!-- 关键神经元列表 -->
                  <el-tab-pane label="关键神经元" name="key-neurons">
                    <div v-for="(neurons, behavior) in getKeyNeuronsByBehavior()" :key="behavior">
                      <h4>{{ behavior }} ({{ neurons.length }} 个神经元)</h4>
                      <el-tag 
                        v-for="neuron in neurons" 
                        :key="neuron" 
                        class="neuron-tag"
                        type="success"
                      >
                        {{ neuron }}
                      </el-tag>
                    </div>
                  </el-tab-pane>

                  <!-- 位置数据 -->
                  <el-tab-pane label="位置数据" name="position-data">
                    <el-table :data="formatPositionData()" style="width: 100%" max-height="400">
                      <el-table-column prop="neuron_id" label="神经元ID" width="120" />
                      <el-table-column prop="x" label="X坐标" width="100" />
                      <el-table-column prop="y" label="Y坐标" width="100" />
                      <el-table-column prop="effect_size" label="平均效应量" width="120" />
                      <el-table-column prop="behaviors" label="相关行为" />
                    </el-table>
                  </el-tab-pane>

                </el-tabs>
              </div>
            </el-card>
          </div>
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script>
import { ref, reactive, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { 
  DataAnalysis, 
  Setting, 
  Upload, 
  UploadFilled, 
  Refresh, 
  Document, 
  Picture, 
  Grid,
  Location,
  List,
  ArrowLeft,
  ArrowRight
} from '@element-plus/icons-vue'
import { neuronAPI } from '@/api'
import PositionMarker from '@/components/PositionMarker.vue'

export default {
  name: 'NeuronAnalysis',
  components: {
    DataAnalysis,
    Setting,
    Upload,
    UploadFilled,
    Refresh,
    Document,
    Picture,
    Grid,
    Location,
    List,
    ArrowLeft,
    ArrowRight,
    PositionMarker
  },
  setup() {
    // 工作流程状态
    const currentStep = ref(0)
    const loading = ref(false)
    
    // 各步骤的结果数据
    const effectSizeResult = ref(null)
    const positionData = ref(null)
    const comprehensiveResult = ref(null)
    
    // 标签页状态
    const effectSizeActiveTab = ref('key-neurons')
    const comprehensiveDataTab = ref('key-neurons')
    
    // 文件上传引用
    const uploadRef = ref(null)
    const positionMarkerRef = ref(null)

    // 各步骤的表单数据
    const effectSizeForm = reactive({
      file: null,
      behaviorColumn: '',
      threshold: 0.5
    })

    const positionForm = reactive({
      startNumber: 1
    })

    const analysisForm = reactive({
      selectedBehavior: '',
      neuronType: 'key',
      displayOptions: ['show-colorbar', 'show-neuron-ids']
    })

    const sidebarWidth = computed(() => {
      return '400px'
    })

    // 工作流程控制方法
    const getStepStatus = (step) => {
      if (step < currentStep.value) return 'success'
      if (step === currentStep.value) return 'process'
      return 'wait'
    }

    const getCurrentStepTitle = () => {
      const titles = ['效应量分析参数', '位置标记参数', '综合分析参数']
      return titles[currentStep.value] || '参数设置'
    }

    const canProceedToNextStep = computed(() => {
      switch (currentStep.value) {
        case 0: return !!effectSizeResult.value
        case 1: return !!positionData.value
        default: return false
      }
    })

    const canStartComprehensiveAnalysis = computed(() => {
      return !!effectSizeResult.value && !!positionData.value
    })

    const goToNextStep = () => {
      if (canProceedToNextStep.value && currentStep.value < 2) {
        currentStep.value++
      }
    }

    const goToPreviousStep = () => {
      if (currentStep.value > 0) {
        currentStep.value--
      }
    }

    // 文件处理方法
    const handleFileChange = (file) => {
      effectSizeForm.file = file.raw
    }

    const handleFileRemove = () => {
      effectSizeForm.file = null
    }


    const formatThresholdTooltip = (value) => {
      if (value < 0.2) return '小效应'
      if (value < 0.5) return '中等效应'
      if (value < 0.8) return '大效应'
      return '极大效应'
    }

    // 数据格式化方法
    const getSummaryLabel = (key) => {
      const labels = {
        total_neurons: '总神经元数',
        total_behaviors: '总行为数',
        key_neurons_found: '关键神经元数',
        analysis_timestamp: '分析时间',
        total_positions: '标记位置数'
      }
      return labels[key] || key
    }

    const getEffectSizeSummary = () => {
      if (!effectSizeResult.value) return {}
      return effectSizeResult.value.statistics || {}
    }

    const getComprehensiveSummary = () => {
      if (!comprehensiveResult.value) return {}
      return comprehensiveResult.value.statistics || {}
    }

    const formatEffectSizesData = () => {
      if (!effectSizeResult.value?.effect_sizes) return []
      
      const effectSizes = effectSizeResult.value.effect_sizes
      return Object.keys(effectSizes).map(neuron => {
        const row = { neuron }
        Object.keys(effectSizes[neuron]).forEach(behavior => {
          row[behavior] = effectSizes[neuron][behavior].toFixed(3)
        })
        return row
      })
    }

    // 格式化效应量矩阵数据
    const formatEffectSizesMatrix = () => {
      if (!effectSizeResult.value?.effect_sizes) return []
      
      const effectSizes = effectSizeResult.value.effect_sizes
      return Object.keys(effectSizes).map(neuron => {
        const row = { neuron }
        Object.keys(effectSizes[neuron]).forEach(behavior => {
          row[behavior] = effectSizes[neuron][behavior]
        })
        return row
      })
    }

    // 格式化效应量数值显示
    const formatEffectSizeValue = (value) => {
      if (typeof value !== 'number') return 'N/A'
      return value.toFixed(3)
    }

    // 获取效应量样式类
    const getEffectSizeClass = (value, behavior) => {
      if (typeof value !== 'number') return 'effect-size-na'
      
      const absValue = Math.abs(value)
      if (absValue >= effectSizeForm.threshold) {
        return 'effect-size-high'
      } else if (absValue >= 0.2) {
        return 'effect-size-medium'
      } else {
        return 'effect-size-low'
      }
    }

    const getBehaviors = () => {
      if (!effectSizeResult.value?.effect_sizes) return []
      const effectSizes = effectSizeResult.value.effect_sizes
      const firstNeuron = Object.keys(effectSizes)[0]
      return firstNeuron ? Object.keys(effectSizes[firstNeuron]) : []
    }

    const getAvailableBehaviors = () => {
      return getBehaviors()
    }

    const getKeyNeuronsByBehavior = () => {
      if (!effectSizeResult.value?.key_neurons) return {}
      
      const keyNeurons = effectSizeResult.value.key_neurons
      const result = {}
      
      // 转换数据结构：从 {behavior: {neuron_ids: [...]}} 到 {behavior: [...]}
      for (const [behavior, neuronInfo] of Object.entries(keyNeurons)) {
        if (neuronInfo && neuronInfo.neuron_ids) {
          result[behavior] = neuronInfo.neuron_ids
        } else {
          result[behavior] = []
        }
      }
      
      return result
    }

    const formatPositionData = () => {
      if (!positionData.value?.positions) return []
      
      const positions = positionData.value.positions
      return Object.keys(positions).map(neuronId => {
        const pos = positions[neuronId]
        return {
          neuron_id: neuronId,
          x: pos.x?.toFixed(2) || 'N/A',
          y: pos.y?.toFixed(2) || 'N/A',
          effect_size: getAverageEffectSize(neuronId),
          behaviors: getRelatedBehaviors(neuronId)
        }
      })
    }

    const getAverageEffectSize = (neuronId) => {
      if (!effectSizeResult.value?.effect_sizes) return 'N/A'
      const effectSizes = effectSizeResult.value.effect_sizes[neuronId]
      if (!effectSizes) return 'N/A'
      const values = Object.values(effectSizes)
      return (values.reduce((a, b) => a + b, 0) / values.length).toFixed(3)
    }

    const getRelatedBehaviors = (neuronId) => {
      if (!effectSizeResult.value?.effect_sizes) return 'N/A'
      const effectSizes = effectSizeResult.value.effect_sizes[neuronId]
      if (!effectSizes) return 'N/A'
      return Object.keys(effectSizes).filter(behavior => 
        Math.abs(effectSizes[behavior]) > effectSizeForm.threshold
      ).join(', ')
    }

    // 步骤1：效应量分析
    const startEffectSizeAnalysis = async () => {
      if (!effectSizeForm.file) {
        ElMessage.warning('请先上传数据文件')
        return
      }

      loading.value = true
      let progressInterval = null

      try {
        // 创建FormData
        const formData = new FormData()
        formData.append('file', effectSizeForm.file)
        
        if (effectSizeForm.behaviorColumn) {
          formData.append('behavior_column', effectSizeForm.behaviorColumn)
        }
        
        formData.append('threshold', effectSizeForm.threshold.toString())

        // 显示进度
        progressInterval = setInterval(() => {
          ElMessage.info('效应量分析进行中，请稍候...')
        }, 5000)

        console.log('开始效应量分析API调用')
        const response = await neuronAPI.effectSize(formData)
        
        if (progressInterval) {
          clearInterval(progressInterval)
        }

        if (response.success) {
          effectSizeResult.value = response.result
          ElMessage.success('效应量分析完成！')
          // 自动进入下一步
          setTimeout(() => {
            currentStep.value = 1
          }, 1000)
        } else {
          throw new Error(response.message || '效应量分析失败')
        }

      } catch (error) {
        if (progressInterval) {
          clearInterval(progressInterval)
        }
        console.error('效应量分析错误:', error)
        ElMessage.error(`效应量分析失败: ${error.message}`)
      } finally {
        loading.value = false
      }
    }

    // 步骤2：位置标记
    const startPositionMarking = () => {
      // 这里可以添加图像处理逻辑
      ElMessage.success('位置标记工具已激活，请在图像上标记神经元位置')
    }

    // 步骤3：综合分析
    const startComprehensiveAnalysis = async () => {
      if (!canStartComprehensiveAnalysis.value) {
        ElMessage.warning('请先完成效应量分析和位置标记')
        return
      }

      loading.value = true
      let progressInterval = null

      try {
        // 创建FormData
        const formData = new FormData()
        
        // 添加效应量分析结果
        if (effectSizeResult.value) {
          formData.append('effect_size_data', JSON.stringify(effectSizeResult.value))
        }
        
        // 添加位置数据
        if (positionData.value) {
          formData.append('position_data', JSON.stringify(positionData.value))
        }
        
        // 添加分析参数
        formData.append('analysis_type', 'single-behavior-spatial')
        formData.append('selected_behavior', analysisForm.selectedBehavior)
        formData.append('neuron_type', analysisForm.neuronType)
        formData.append('display_options', JSON.stringify(analysisForm.displayOptions))

        // 显示进度
        progressInterval = setInterval(() => {
          ElMessage.info('综合分析进行中，请稍候...')
        }, 5000)

        console.log('开始综合分析API调用')
        const response = await neuronAPI.comprehensiveAnalysis(formData)
        
        if (progressInterval) {
          clearInterval(progressInterval)
        }

        if (response.success) {
          comprehensiveResult.value = response.result
          ElMessage.success('综合分析完成！')
        } else {
          throw new Error(response.message || '综合分析失败')
        }

      } catch (error) {
        if (progressInterval) {
          clearInterval(progressInterval)
        }
        console.error('综合分析错误:', error)
        ElMessage.error(`综合分析失败: ${error.message}`)
      } finally {
        loading.value = false
      }
    }

    // 位置标记相关
    const handlePositionsUpdated = (data) => {
      positionData.value = data
      console.log('位置数据已更新:', data)
      ElMessage.success(`已标记 ${data.total_neurons} 个神经元位置`)
    }
    
    const resetWorkflow = () => {
      // 重置工作流程状态
      currentStep.value = 0
      
      // 重置表单数据
      effectSizeForm.file = null
      effectSizeForm.behaviorColumn = ''
      effectSizeForm.threshold = 0.5
      
      positionForm.startNumber = 1
      
      analysisForm.selectedBehavior = ''
      analysisForm.neuronType = 'key'
      analysisForm.displayOptions = ['show-colorbar', 'show-neuron-ids']
      
      // 重置结果数据
      effectSizeResult.value = null
      positionData.value = null
      comprehensiveResult.value = null
      
      // 重置标签页
      effectSizeActiveTab.value = 'key-neurons'
      comprehensiveDataTab.value = 'key-neurons'
      
      // 清空文件上传
      if (uploadRef.value) {
        uploadRef.value.clearFiles()
      }
      
      ElMessage.success('工作流程已重置')
    }

    return {
      // 状态
      currentStep,
      loading,
      effectSizeResult,
      positionData,
      comprehensiveResult,
      
      // 标签页状态
      effectSizeActiveTab,
      comprehensiveDataTab,
      
      // 表单数据
      effectSizeForm,
      positionForm,
      analysisForm,
      
      // 引用
      uploadRef,
      positionMarkerRef,
      
      // 计算属性
      sidebarWidth,
      canProceedToNextStep,
      canStartComprehensiveAnalysis,
      
      // 方法
      getStepStatus,
      getCurrentStepTitle,
      goToNextStep,
      goToPreviousStep,
      handleFileChange,
      handleFileRemove,
      formatThresholdTooltip,
      getSummaryLabel,
      getEffectSizeSummary,
      getComprehensiveSummary,
      formatEffectSizesData,
      formatEffectSizesMatrix,
      formatEffectSizeValue,
      getEffectSizeClass,
      getBehaviors,
      getAvailableBehaviors,
      getKeyNeuronsByBehavior,
      formatPositionData,
      startEffectSizeAnalysis,
      startPositionMarking,
      startComprehensiveAnalysis,
      handlePositionsUpdated,
      resetWorkflow
    }
  }
}
</script>

<style scoped>
.neuron-analysis {
  min-height: 100vh;
  background-color: #f5f7fa;
}

.page-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  text-align: center;
  padding: 20px;
}

.page-title {
  margin: 0;
  font-size: 28px;
  font-weight: bold;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
}

.page-description {
  margin: 10px 0 0 0;
  font-size: 16px;
  opacity: 0.9;
}

.workflow-panel {
  background-color: white;
  border-right: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 20px;
}

.workflow-card {
  border: none;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.parameter-card {
  border: none;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  flex: 1;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: bold;
  color: #303133;
  font-size: 18px;
}

.step-content {
  padding: 10px 0;
}

.analysis-form {
  padding: 0 10px;
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

.analysis-button {
  width: 100%;
  height: 45px;
  font-size: 16px;
  font-weight: bold;
}

.workflow-controls {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.results-panel {
  padding: 20px;
  background-color: #f5f7fa;
}

.empty-result-card {
  height: 400px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.upload-card {
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
}

.upload-section {
  padding: 20px;
}

.step-results {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.result-card {
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.summary-content {
  padding: 10px 0;
}

.summary-item {
  text-align: center;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 6px;
  border: 1px solid #e9ecef;
}

.summary-label {
  font-size: 14px;
  color: #6c757d;
  margin-bottom: 8px;
}

.summary-value {
  font-size: 24px;
  font-weight: bold;
  color: #495057;
}

.visualization-content,
.data-content {
  padding: 10px 0;
}

.image-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
  align-items: center;
}

.result-image {
  max-width: 100%;
  height: auto;
  border-radius: 6px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.neuron-tag {
  margin: 2px;
}

/* 效应量矩阵样式 */
.matrix-container {
  padding: 20px 0;
}

.matrix-header {
  margin-bottom: 20px;
  text-align: center;
}

.matrix-header h4 {
  margin: 0 0 10px 0;
  color: #303133;
  font-size: 18px;
  font-weight: bold;
}

.matrix-description {
  margin: 0;
  color: #606266;
  font-size: 14px;
  line-height: 1.5;
}

.effect-matrix-table {
  margin-bottom: 20px;
}

.neuron-id {
  font-weight: bold;
  color: #409eff;
}

.effect-size-value {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-weight: bold;
  font-size: 13px;
  min-width: 60px;
  text-align: center;
}

/* 效应量样式类 */
.effect-size-high {
  background-color: #fef0f0;
  color: #f56c6c;
  border: 1px solid #fbc4c4;
}

.effect-size-medium {
  background-color: #fdf6ec;
  color: #e6a23c;
  border: 1px solid #f5dab1;
}

.effect-size-low {
  background-color: #f0f9ff;
  color: #909399;
  border: 1px solid #d3d4d6;
}

.effect-size-na {
  background-color: #f5f7fa;
  color: #c0c4cc;
  border: 1px solid #e4e7ed;
}

/* 图例样式 */
.matrix-legend {
  background-color: #f8f9fa;
  padding: 15px;
  border-radius: 6px;
  border: 1px solid #e9ecef;
}

.matrix-legend h5 {
  margin: 0 0 10px 0;
  color: #495057;
  font-size: 14px;
  font-weight: bold;
}

.legend-items {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #6c757d;
}

.legend-color {
  width: 16px;
  height: 16px;
  border-radius: 3px;
  border: 1px solid #ddd;
}

.legend-color.high-effect {
  background-color: #fef0f0;
  border-color: #fbc4c4;
}

.legend-color.medium-effect {
  background-color: #fdf6ec;
  border-color: #f5dab1;
}

.legend-color.low-effect {
  background-color: #f0f9ff;
  border-color: #d3d4d6;
}

/* 单行为空间分析参数样式 */
.behavior-analysis-params {
  background-color: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 6px;
  padding: 15px;
  margin: 10px 0;
}

.behavior-analysis-params .el-form-item {
  margin-bottom: 15px;
}

.behavior-analysis-params .el-form-item:last-child {
  margin-bottom: 0;
}

/* 工作流程步骤样式 */
:deep(.el-steps) {
  padding: 20px 0;
}

:deep(.el-step__title) {
  font-weight: bold;
  color: #303133;
}

:deep(.el-step__description) {
  color: #606266;
  font-size: 12px;
}

:deep(.el-step.is-process .el-step__title) {
  color: #409eff;
}

:deep(.el-step.is-success .el-step__title) {
  color: #67c23a;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .workflow-panel {
    width: 100% !important;
  }
  
  .page-title {
    font-size: 24px;
  }
  
  .summary-item {
    margin-bottom: 10px;
  }
  
  .workflow-controls {
    flex-direction: row;
    flex-wrap: wrap;
  }
  
  .legend-items {
    flex-direction: column;
    gap: 10px;
  }
  
  .effect-matrix-table {
    font-size: 12px;
  }
  
  .effect-size-value {
    padding: 2px 4px;
    font-size: 11px;
    min-width: 50px;
  }
}
</style>
