import axios from 'axios'
import { ElMessage } from 'element-plus'

// 创建axios实例
const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 5分钟超时，因为分析可能需要较长时间
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
      }
    })
  },
  
  // 获取交互式图表数据
  getInteractiveData(formData) {
    return api.post('/extraction/interactive_data', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  },
  
  // 基于用户选择的时间范围进行手动提取
  manualExtract(formData) {
    return api.post('/extraction/manual_extract', formData, {
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
  }
}

// 聚类分析相关API
export const clusteringAPI = {
  // 获取结果文件列表
  getResultFiles() {
    return api.get('/results/files')
  },
  
  // 执行聚类分析
  analyze(data) {
    return api.post('/clustering/analyze', data, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
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