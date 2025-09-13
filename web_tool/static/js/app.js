// 神经元数据分析工具 - 前端JavaScript

class NeuronAnalysisTool {
    constructor() {
        this.selectedModule = null;
        this.currentTaskId = null;
        this.modules = {};
        this.statusCheckInterval = null;
        
        this.init();
    }
    
    init() {
        this.loadModules();
        this.setupEventListeners();
        this.setupFileUpload();
    }
    
    async loadModules() {
        try {
            const response = await fetch('/api/modules');
            this.modules = await response.json();
        } catch (error) {
            this.showToast('加载模块信息失败', 'error');
        }
    }
    
    setupEventListeners() {
        // 模块选择
        document.querySelectorAll('.module-card').forEach(card => {
            card.addEventListener('click', () => {
                this.selectModule(card.dataset.module);
            });
        });
        
        // 开始分析按钮
        document.getElementById('start-analysis').addEventListener('click', () => {
            this.startAnalysis();
        });
    }
    
    setupFileUpload() {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        
        // 拖拽上传
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });
        
        // 文件选择
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileUpload(e.target.files[0]);
            }
        });
    }
    
    selectModule(moduleName) {
        // 清除之前的选择
        document.querySelectorAll('.module-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        // 选择新模块
        document.querySelector(`[data-module="${moduleName}"]`).classList.add('selected');
        this.selectedModule = moduleName;
        
        // 显示模块详情
        this.showModuleDetails(moduleName);
        
        // 显示上传区域
        document.getElementById('upload-section').style.display = 'block';
        
        // 更新步骤状态
        this.updateStepStatus(1, 'completed');
        this.updateStepStatus(2, 'active');
    }
    
    showModuleDetails(moduleName) {
        const module = this.modules[moduleName];
        if (!module) return;
        
        const detailsSection = document.getElementById('module-details');
        document.getElementById('module-title').textContent = module.name;
        document.getElementById('module-description').textContent = module.description;
        
        // 输入要求
        const inputReqs = document.getElementById('input-requirements');
        inputReqs.innerHTML = '';
        module.required_columns.forEach(col => {
            const li = document.createElement('li');
            li.textContent = col === 'neuron_columns' ? '神经元数据列' : col;
            inputReqs.appendChild(li);
        });
        
        // 输出类型
        const outputTypes = document.getElementById('output-types');
        outputTypes.innerHTML = '';
        module.outputs.forEach(output => {
            const li = document.createElement('li');
            li.textContent = `${output.name} (${output.format.toUpperCase()})`;
            outputTypes.appendChild(li);
        });
        
        // 检查是否需要额外文件
        if (module.additional_files) {
            this.setupAdditionalFiles(module.additional_files);
        }
        
        detailsSection.style.display = 'block';
    }
    
    setupAdditionalFiles(additionalFiles) {
        const container = document.getElementById('additional-files-container');
        const section = document.getElementById('additional-files');
        
        container.innerHTML = '';
        
        Object.entries(additionalFiles).forEach(([fileType, config]) => {
            const div = document.createElement('div');
            div.className = 'mb-3';
            div.innerHTML = `
                <label class="form-label">${config.description}</label>
                <input type="file" class="form-control" 
                       data-file-type="${fileType}" 
                       accept=".${config.format}" 
                       ${config.required ? 'required' : ''}>
                <small class="text-muted">格式: ${config.format.toUpperCase()}</small>
            `;
            container.appendChild(div);
        });
        
        section.style.display = 'block';
    }
    
    async handleFileUpload(file) {
        if (!this.selectedModule) {
            this.showToast('请先选择分析模块', 'warning');
            return;
        }
        
        // 显示进度条
        document.querySelector('.progress-container').style.display = 'block';
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('module', this.selectedModule);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.currentTaskId = result.task_id;
                this.showToast(result.message, 'success');
                
                // 更新步骤状态
                this.updateStepStatus(2, 'completed');
                this.updateStepStatus(3, 'active');
                
                // 显示参数配置
                this.showParametersSection();
            } else {
                this.showToast(result.error, 'error');
            }
        } catch (error) {
            this.showToast('文件上传失败', 'error');
        } finally {
            document.querySelector('.progress-container').style.display = 'none';
        }
    }
    
    showParametersSection() {
        const module = this.modules[this.selectedModule];
        const container = document.getElementById('parameters-container');
        
        container.innerHTML = '';
        
        Object.entries(module.parameters).forEach(([paramName, paramConfig]) => {
            const paramDiv = document.createElement('div');
            paramDiv.className = 'parameter-group';
            
            let inputHtml = '';
            
            switch (paramConfig.type) {
                case 'int':
                case 'float':
                    inputHtml = `
                        <input type="number" 
                               class="form-control" 
                               id="param-${paramName}" 
                               value="${paramConfig.default || ''}" 
                               ${paramConfig.min !== undefined ? `min="${paramConfig.min}"` : ''}
                               ${paramConfig.max !== undefined ? `max="${paramConfig.max}"` : ''}
                               ${paramConfig.type === 'float' ? 'step="0.1"' : ''}
                               ${paramConfig.required ? 'required' : ''}>
                    `;
                    break;
                    
                case 'str':
                    if (paramConfig.options) {
                        inputHtml = `
                            <select class="form-select" id="param-${paramName}" ${paramConfig.required ? 'required' : ''}>
                                ${paramConfig.options.map(option => 
                                    `<option value="${option}" ${option === paramConfig.default ? 'selected' : ''}>${option}</option>`
                                ).join('')}
                            </select>
                        `;
                    } else {
                        inputHtml = `
                            <input type="text" 
                                   class="form-control" 
                                   id="param-${paramName}" 
                                   value="${paramConfig.default || ''}" 
                                   ${paramConfig.required ? 'required' : ''}>
                        `;
                    }
                    break;
                    
                case 'bool':
                    inputHtml = `
                        <div class="form-check">
                            <input type="checkbox" 
                                   class="form-check-input" 
                                   id="param-${paramName}" 
                                   ${paramConfig.default ? 'checked' : ''}>
                            <label class="form-check-label" for="param-${paramName}">
                                启用
                            </label>
                        </div>
                    `;
                    break;
            }
            
            paramDiv.innerHTML = `
                <label class="form-label fw-bold">
                    ${paramName.replace('_', ' ')}
                    ${paramConfig.required ? '<span class="text-danger">*</span>' : ''}
                </label>
                ${inputHtml}
                <small class="text-muted">${paramConfig.description}</small>
            `;
            
            container.appendChild(paramDiv);
        });
        
        document.getElementById('parameters-section').style.display = 'block';
    }
    
    async startAnalysis() {
        if (!this.currentTaskId) {
            this.showToast('请先上传数据文件', 'warning');
            return;
        }
        
        // 收集参数
        const parameters = {};
        const module = this.modules[this.selectedModule];
        
        Object.keys(module.parameters).forEach(paramName => {
            const element = document.getElementById(`param-${paramName}`);
            if (element) {
                if (element.type === 'checkbox') {
                    parameters[paramName] = element.checked;
                } else if (element.type === 'number') {
                    parameters[paramName] = element.value ? parseFloat(element.value) : null;
                } else {
                    parameters[paramName] = element.value || null;
                }
            }
        });
        
        // 上传额外文件
        await this.uploadAdditionalFiles();
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    task_id: this.currentTaskId,
                    parameters: parameters
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showToast('分析已开始', 'success');
                
                // 更新步骤状态
                this.updateStepStatus(3, 'completed');
                this.updateStepStatus(4, 'active');
                
                // 显示分析状态
                this.showAnalysisStatus();
                
                // 开始状态检查
                this.startStatusCheck();
            } else {
                this.showToast(result.error, 'error');
            }
        } catch (error) {
            this.showToast('启动分析失败', 'error');
        }
    }
    
    async uploadAdditionalFiles() {
        const additionalInputs = document.querySelectorAll('[data-file-type]');
        
        for (const input of additionalInputs) {
            if (input.files.length > 0) {
                const formData = new FormData();
                formData.append('file', input.files[0]);
                formData.append('task_id', this.currentTaskId);
                formData.append('file_type', input.dataset.fileType);
                
                try {
                    await fetch('/api/upload_additional', {
                        method: 'POST',
                        body: formData
                    });
                } catch (error) {
                    console.error('上传额外文件失败:', error);
                }
            }
        }
    }
    
    showAnalysisStatus() {
        document.getElementById('parameters-section').style.display = 'none';
        document.getElementById('analysis-status').style.display = 'block';
    }
    
    startStatusCheck() {
        this.statusCheckInterval = setInterval(async () => {
            await this.checkStatus();
        }, 3000); // 每3秒检查一次
    }
    
    async checkStatus() {
        try {
            const response = await fetch(`/api/status/${this.currentTaskId}`);
            const status = await response.json();
            
            document.getElementById('status-message').textContent = 
                status.status === 'running' ? '正在分析数据...' : 
                status.status === 'completed' ? '分析完成！' : 
                status.status === 'error' ? '分析出错' : '未知状态';
            
            if (status.status === 'completed') {
                clearInterval(this.statusCheckInterval);
                this.showResults();
            } else if (status.status === 'error') {
                clearInterval(this.statusCheckInterval);
                this.showToast(status.error || '分析过程中出现错误', 'error');
            }
        } catch (error) {
            console.error('检查状态失败:', error);
        }
    }
    
    async showResults() {
        try {
            const response = await fetch(`/api/results/${this.currentTaskId}`);
            const results = await response.json();
            
            // 更新步骤状态
            this.updateStepStatus(4, 'completed');
            this.updateStepStatus(5, 'active');
            
            // 隐藏分析状态，显示结果
            document.getElementById('analysis-status').style.display = 'none';
            document.getElementById('results-section').style.display = 'block';
            
            // 渲染结果
            this.renderResults(results.results);
            
        } catch (error) {
            this.showToast('获取结果失败', 'error');
        }
    }
    
    renderResults(results) {
        const container = document.getElementById('results-container');
        container.innerHTML = '';
        
        if (results.length === 0) {
            container.innerHTML = '<p class="text-muted">没有生成结果文件</p>';
            return;
        }
        
        results.forEach(result => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'result-item';
            
            const icon = result.type === 'image' ? 'fas fa-image' : 'fas fa-file-alt';
            const sizeText = this.formatFileSize(result.size);
            
            resultDiv.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <i class="${icon} me-2"></i>
                        <strong>${result.filename}</strong>
                        <small class="text-muted ms-2">(${sizeText})</small>
                    </div>
                    <div>
                        ${result.type === 'image' ? 
                            `<button class="btn btn-sm btn-outline-primary me-2" onclick="viewImage('${result.download_url}')">
                                <i class="fas fa-eye"></i> 预览
                            </button>` : ''}
                        <a href="${result.download_url}" class="btn btn-sm btn-primary">
                            <i class="fas fa-download"></i> 下载
                        </a>
                    </div>
                </div>
            `;
            
            container.appendChild(resultDiv);
        });
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    updateStepStatus(stepNumber, status) {
        const step = document.getElementById(`step${stepNumber}`);
        step.className = `step ${status}`;
    }
    
    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        const toastBody = document.getElementById('toast-body');
        
        // 设置图标和颜色
        const icon = document.querySelector('#toast .toast-header i');
        icon.className = `fas me-2 ${
            type === 'success' ? 'fa-check-circle text-success' :
            type === 'error' ? 'fa-exclamation-circle text-danger' :
            type === 'warning' ? 'fa-exclamation-triangle text-warning' :
            'fa-info-circle text-primary'
        }`;
        
        toastBody.textContent = message;
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }
}

// 全局函数
function viewImage(url) {
    window.open(url, '_blank');
}

function startNewAnalysis() {
    location.reload();
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new NeuronAnalysisTool();
});