import axios from 'axios'
import { ElMessage } from 'element-plus'

// 创建axios实例
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  timeout: 300000, // 5分钟超时，因为分析可能需要较长时间
  maxContentLength: 100 * 1024 * 1024, // 100MB
  maxBodyLength: 100 * 1024 * 1024, // 100MB
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
api.interceptors.request.use(
  config => {
    return config
  },
  error => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  response => {
    return response.data
  },
  error => {
    console.error('响应错误:', error)
    
    let message = '请求失败'
    if (error.response) {
      message = error.response.data?.detail || `请求失败 (${error.response.status})`
    } else if (error.request) {
      message = '网络连接失败，请检查后端服务是否启动'
    }
    
    ElMessage.error(message)
    return Promise.reject(error)
  }
)

// 事件提取相关API
export const extractionAPI = {
  // 预览单个神经元的提取结果
  preview(formData) {
    return api.post('/extraction/preview', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      maxContentLength: 100 * 1024 * 1024, // 100MB
      maxBodyLength: 100 * 1024 * 1024, // 100MB
      timeout: 300000 // 5分钟超时
    })
  },
  
  // 获取交互式图表数据
  getInteractiveData(formData) {
    return api.post('/extraction/interactive_data', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      maxContentLength: 100 * 1024 * 1024,
      maxBodyLength: 100 * 1024 * 1024,
      timeout: 300000
    })
  },
  
  // 基于用户选择的时间范围进行手动提取
  manualExtract(formData) {
    return api.post('/extraction/manual_extract', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      maxContentLength: 100 * 1024 * 1024,
      maxBodyLength: 100 * 1024 * 1024,
      timeout: 300000
    })
  },
  
  // 处理检测到的事件特征
  processFeatures(formData) {
    return api.post('/extraction/process_features', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  },
  
  // 批量提取
  batchExtraction(formData) {
    return api.post('/extraction/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  },
  
  // 保存预览结果
  savePreviewResult(formData) {
    return api.post('/extraction/save_preview', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  }
}

// 聚类分析相关API
export const clusteringAPI = {
  // 获取结果文件列表
  getResultFiles() {
    return api.get('/results/files')
  },
  
  // 执行综合聚类分析
  analyze(data) {
    const formData = new FormData()
    
    // 添加基本参数
    formData.append('filename', data.filename)
    if (data.k !== null && data.k !== undefined) {
      formData.append('k', data.k)
    }
    formData.append('algorithm', data.algorithm || 'kmeans')
    formData.append('reduction_method', data.reduction_method || 'pca')
    formData.append('auto_k', data.auto_k || false)
    formData.append('auto_k_range', data.auto_k_range || '2,10')
    
    // 添加特征权重（如果有）
    if (data.feature_weights) {
      formData.append('feature_weights', JSON.stringify(data.feature_weights))
    }
    
    // 添加DBSCAN参数
    if (data.algorithm === 'dbscan') {
      formData.append('dbscan_eps', data.dbscan_eps || 0.5)
      formData.append('dbscan_min_samples', data.dbscan_min_samples || 5)
    }
    
    return api.post('/clustering/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 600000 // 10分钟超时，聚类分析可能需要较长时间
    })
  },
  
  // 确定最佳K值
  findOptimalK(data) {
    const formData = new FormData()
    formData.append('filename', data.filename)
    formData.append('max_k', data.max_k || 10)
    
    if (data.feature_weights) {
      formData.append('feature_weights', JSON.stringify(data.feature_weights))
    }
    
    return api.post('/clustering/optimal_k', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 300000 // 5分钟超时
    })
  },
  
  // 比较不同K值的效果
  compareK(data) {
    const formData = new FormData()
    formData.append('filename', data.filename)
    formData.append('k_values', data.k_values || '2,3,4,5')
    
    if (data.feature_weights) {
      formData.append('feature_weights', JSON.stringify(data.feature_weights))
    }
    
    return api.post('/clustering/compare_k', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 600000 // 10分钟超时
    })
  }
}

// 文件下载API
export const downloadAPI = {
  // 下载文件
  downloadFile(filename) {
    return api.get(`/download/${filename}`, {
      responseType: 'blob'
    })
  }
}

export default api