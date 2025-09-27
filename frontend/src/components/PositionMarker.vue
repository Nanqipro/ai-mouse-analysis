<template>
  <div class="position-marker">
    <div class="marker-header">
      <h3>神经元位置标记</h3>
      <div class="marker-controls">
        <el-button @click="clearAllMarks" type="danger" size="small">清空所有标记</el-button>
        <el-button @click="exportPositions" type="primary" size="small">导出位置数据</el-button>
      </div>
    </div>
    
    <div class="marker-content">
      <!-- 图片上传区域 -->
      <div class="image-upload-section">
        <el-upload
          ref="imageUploadRef"
          :auto-upload="false"
          :on-change="handleImageChange"
          :before-remove="handleImageRemove"
          :limit="1"
          accept="image/*"
          drag
        >
          <el-icon class="el-icon--upload"><upload-filled /></el-icon>
          <div class="el-upload__text">
            将图片拖到此处，或<em>点击上传</em>
          </div>
          <template #tip>
            <div class="el-upload__tip">
              支持JPG、PNG等图片格式，用于标记神经元位置
            </div>
          </template>
        </el-upload>
      </div>
      
      <!-- 图片显示和标记区域 -->
      <div v-if="backgroundImage" class="image-marker-container">
        <div class="marker-instructions">
          <p>点击图片上的位置来标记神经元，右键点击可删除标记</p>
          <p>当前已标记: {{ Object.keys(positions).length }} 个神经元</p>
        </div>
        
        <div class="image-container" ref="imageContainer">
          <img 
            ref="backgroundImg"
            :src="backgroundImage" 
            alt="背景图片"
            @click="handleImageClick"
            @contextmenu.prevent="handleRightClick"
            class="marker-image"
          />
          
          <!-- 标记点 -->
          <div 
            v-for="(pos, neuronId) in positions" 
            :key="neuronId"
            class="neuron-marker"
            :style="{
              left: pos.x + 'px',
              top: pos.y + 'px'
            }"
            @click.stop="selectMarker(neuronId)"
            @contextmenu.stop.prevent="removeMarker(neuronId)"
          >
            <div class="marker-label" :class="{ selected: selectedMarker === neuronId }">
              {{ formatNeuronId(neuronId) }}
            </div>
          </div>
        </div>
        
        <!-- 标记控制面板 -->
        <div class="marker-panel">
          <el-form :model="markerForm" label-width="100px" size="small">
            <el-form-item label="神经元ID">
              <el-input 
                v-model="markerForm.neuronId" 
                placeholder="输入神经元ID"
                @keyup.enter="addMarker"
              />
            </el-form-item>
            <el-form-item label="坐标">
              <el-input 
                v-model="markerForm.x" 
                placeholder="X坐标"
                style="width: 80px; margin-right: 10px;"
              />
              <el-input 
                v-model="markerForm.y" 
                placeholder="Y坐标"
                style="width: 80px;"
              />
            </el-form-item>
            <el-form-item>
              <el-button @click="addMarker" type="primary" size="small">添加标记</el-button>
              <el-button @click="clearForm" size="small">清空</el-button>
            </el-form-item>
          </el-form>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, nextTick } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'

export default {
  name: 'PositionMarker',
  components: {
    UploadFilled
  },
  emits: ['positions-updated'],
  setup(props, { emit }) {
    const imageUploadRef = ref(null)
    const imageContainer = ref(null)
    const backgroundImg = ref(null)
    
    const backgroundImage = ref(null)
    const positions = ref({})
    const selectedMarker = ref(null)
    
    const markerForm = reactive({
      neuronId: '',
      x: '',
      y: ''
    })
    
    // 处理图片上传
    const handleImageChange = (file) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        backgroundImage.value = e.target.result
        positions.value = {}
        selectedMarker.value = null
      }
      reader.readAsDataURL(file.raw)
    }
    
    const handleImageRemove = () => {
      backgroundImage.value = null
      positions.value = {}
      selectedMarker.value = null
    }
    
    // 处理图片点击
    const handleImageClick = (event) => {
      if (!backgroundImg.value) return
      
      const rect = backgroundImg.value.getBoundingClientRect()
      const x = event.clientX - rect.left
      const y = event.clientY - rect.top
      
      // 生成新的神经元ID
      const neuronId = `Neuron_${Object.keys(positions.value).length + 1}`
      
      // 添加标记
      positions.value[neuronId] = { x, y }
      selectedMarker.value = neuronId
      
      // 更新表单
      markerForm.neuronId = neuronId
      markerForm.x = Math.round(x)
      markerForm.y = Math.round(y)
      
      emitPositions()
    }
    
    // 处理右键点击
    const handleRightClick = (event) => {
      // 右键点击图片空白处，清空选择
      selectedMarker.value = null
    }
    
    // 选择标记
    const selectMarker = (neuronId) => {
      selectedMarker.value = neuronId
      const pos = positions.value[neuronId]
      markerForm.neuronId = neuronId
      markerForm.x = Math.round(pos.x)
      markerForm.y = Math.round(pos.y)
    }
    
    // 删除标记
    const removeMarker = (neuronId) => {
      delete positions.value[neuronId]
      if (selectedMarker.value === neuronId) {
        selectedMarker.value = null
        clearForm()
      }
      emitPositions()
    }
    
    // 添加标记
    const addMarker = () => {
      if (!markerForm.neuronId || !markerForm.x || !markerForm.y) {
        ElMessage.warning('请填写完整的标记信息')
        return
      }
      
      const x = parseFloat(markerForm.x)
      const y = parseFloat(markerForm.y)
      
      if (isNaN(x) || isNaN(y)) {
        ElMessage.warning('坐标必须是数字')
        return
      }
      
      positions.value[markerForm.neuronId] = { x, y }
      selectedMarker.value = markerForm.neuronId
      
      emitPositions()
    }
    
    // 清空表单
    const clearForm = () => {
      markerForm.neuronId = ''
      markerForm.x = ''
      markerForm.y = ''
    }
    
    // 清空所有标记
    const clearAllMarks = () => {
      positions.value = {}
      selectedMarker.value = null
      clearForm()
      emitPositions()
    }
    
    // 导出位置数据
    const exportPositions = () => {
      if (Object.keys(positions.value).length === 0) {
        ElMessage.warning('没有位置数据可导出')
        return
      }
      
      const data = {
        positions: positions.value,
        total_neurons: Object.keys(positions.value).length,
        export_time: new Date().toISOString()
      }
      
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'neuron_positions.json'
      a.click()
      URL.revokeObjectURL(url)
      
      ElMessage.success('位置数据已导出')
    }
    
    // 格式化神经元ID显示
    const formatNeuronId = (neuronId) => {
      // 将 Neuron_1, Neuron_2 等格式化为 n1, n2
      if (neuronId.startsWith('Neuron_')) {
        const number = neuronId.replace('Neuron_', '')
        return `n${number}`
      }
      return neuronId
    }
    
    // 发送位置数据更新事件
    const emitPositions = () => {
      emit('positions-updated', {
        positions: positions.value,
        total_neurons: Object.keys(positions.value).length
      })
    }
    
    return {
      imageUploadRef,
      imageContainer,
      backgroundImg,
      backgroundImage,
      positions,
      selectedMarker,
      markerForm,
      handleImageChange,
      handleImageRemove,
      handleImageClick,
      handleRightClick,
      selectMarker,
      removeMarker,
      addMarker,
      clearForm,
      clearAllMarks,
      exportPositions,
      formatNeuronId
    }
  }
}
</script>

<style scoped>
.position-marker {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 20px;
  background: white;
}

.marker-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid #e4e7ed;
}

.marker-header h3 {
  margin: 0;
  color: #303133;
}

.marker-controls {
  display: flex;
  gap: 10px;
}

.image-upload-section {
  margin-bottom: 20px;
}

.image-marker-container {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 20px;
  background: #fafafa;
}

.marker-instructions {
  margin-bottom: 15px;
  padding: 10px;
  background: #f0f9ff;
  border-radius: 4px;
  border-left: 4px solid #409eff;
}

.marker-instructions p {
  margin: 5px 0;
  color: #606266;
  font-size: 14px;
}

.image-container {
  position: relative;
  display: inline-block;
  margin-bottom: 20px;
  border: 2px solid #dcdfe6;
  border-radius: 8px;
  overflow: hidden;
  cursor: crosshair;
}

.marker-image {
  display: block;
  max-width: 100%;
  max-height: 500px;
  user-select: none;
}

.neuron-marker {
  position: absolute;
  transform: translate(-50%, -50%);
  cursor: pointer;
  z-index: 10;
}

.marker-label {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: rgba(245, 108, 108, 0.9);
  color: white;
  padding: 2px 4px;
  border-radius: 3px;
  font-size: 9px;
  font-weight: bold;
  white-space: nowrap;
  pointer-events: none;
  border: 1px solid rgba(255, 255, 255, 0.3);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
  transition: all 0.2s ease;
}

.marker-label.selected {
  background: rgba(64, 158, 255, 0.9);
  transform: translate(-50%, -50%) scale(1.1);
  box-shadow: 0 2px 6px rgba(64, 158, 255, 0.4);
  border-color: rgba(255, 255, 255, 0.5);
}

.marker-panel {
  background: white;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 20px;
}

.marker-panel .el-form-item {
  margin-bottom: 15px;
}
</style>
