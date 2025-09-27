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
        在此页面，您可以选择不同的热力图分析类型：<br>
        1. <strong>行为序列热力图</strong>: 分析特定行为前后时间窗口的神经元活动模式<br>
        2. <strong>EM排序热力图</strong>: 基于峰值或钙波时间对神经元进行排序的热力图<br>
        3. <strong>多天数据组合热力图</strong>: 对比分析多天实验数据的神经元活动变化
      </template>
    </el-alert>

    <!-- 分析类型选择卡 -->
    <el-tabs v-model="activeTab" type="card" class="analysis-tabs">
      <!-- 行为序列热力图 -->
      <el-tab-pane label="行为序列热力图" name="behavior">
        <el-row :gutter="20">
          <!-- 左侧参数面板 -->
          <el-col :xs="24" :sm="24" :md="8" :lg="6">
            <div class="params-panel card">
              <h3 class="section-title">
                <el-icon><Setting /></el-icon>
                行为分析参数
              </h3>
              

              
              <el-alert
                v-if="!hasBehaviorFile"
                :key="'alert-' + forceUpdateKey"
                title="请先上传数据文件"
                type="info"
                :closable="false"
                show-icon
                style="margin-bottom: 15px;"
              >
                上传包含行为标签的数据文件后，系统将自动检测并提供可用的行为选项。
              </el-alert>
              
              <el-form :model="behaviorParams" label-width="120px" size="small">
                <el-form-item label="起始行为">
                  <el-select
                    v-model="behaviorParams.start_behavior"
                    placeholder="选择起始行为"
                    style="width: 100%"
                    :loading="behaviorLabelsLoading"
                    :disabled="behaviorLabelsLoading"
                  >
                    <el-option
                      v-for="behavior in behaviorOptions"
                      :key="behavior.value"
                      :label="behavior.label"
                      :value="behavior.value"
                    />
                  </el-select>
                  <div class="param-help">
                    <span v-if="behaviorLabelsLoading">正在从上传的数据中提取行为标签...</span>
                    <span v-else>分析从此行为开始</span>
                  </div>
                </el-form-item>
                
                <el-form-item label="结束行为">
                  <el-select
                    v-model="behaviorParams.end_behavior"
                    placeholder="选择结束行为"
                    style="width: 100%"
                    :loading="behaviorLabelsLoading"
                    :disabled="behaviorLabelsLoading"
                  >
                    <el-option
                      v-for="behavior in behaviorOptions"
                      :key="behavior.value"
                      :label="behavior.label"
                      :value="behavior.value"
                    />
                  </el-select>
                  <div class="param-help">
                    <span v-if="behaviorLabelsLoading">正在从上传的数据中提取行为标签...</span>
                    <span v-else>分析到此行为结束</span>
                  </div>
                </el-form-item>
                
                <el-form-item label="行为前时间">
                  <el-input-number
                    v-model="behaviorParams.pre_behavior_time"
                    :min="1"
                    :max="60"
                    :step="1"
                    :precision="1"
                    style="width: 100%"
                  />
                  <div class="param-help">单位：秒</div>
                </el-form-item>
                
                <el-form-item label="行为后时间">
                  <el-input-number
                    v-model="behaviorParams.post_behavior_time"
                    :min="1"
                    :max="120"
                    :step="1"
                    :precision="1"
                    style="width: 100%"
                  />
                  <div class="param-help">单位：秒</div>
                </el-form-item>
                
                <el-form-item label="采样频率">
                  <el-input-number
                    v-model="behaviorParams.sampling_rate"
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
                    v-model="behaviorParams.min_behavior_duration"
                    :min="0.1"
                    :max="10"
                    :step="0.1"
                    :precision="1"
                    style="width: 100%"
                  />
                  <div class="param-help">单位：秒</div>
                </el-form-item>
              </el-form>
            </div>
          </el-col>
          
          <!-- 右侧内容区域 -->
          <el-col :xs="24" :sm="24" :md="16" :lg="18">
            <!-- 文件上传区域 -->
            <div class="upload-section card">
              <h3 class="section-title">
                <el-icon><Upload /></el-icon>
                数据文件上传
              </h3>
              
              <el-upload
                ref="behaviorUploadRef"
                :file-list="behaviorFileList"
                :on-change="handleBehaviorFileChange"
                :on-remove="handleBehaviorFileRemove"
                :before-upload="() => false"
                :auto-upload="false"
                accept=".xlsx,.xls"
                drag
                :limit="1"
                list-type="text"
              >
                <el-icon class="el-icon--upload"><upload-filled /></el-icon>
                <div class="el-upload__text">
                  将文件拖到此处，或<em>点击上传</em>
                </div>
                <template #tip>
                  <div class="el-upload__tip">
                    支持 Excel 文件格式，文件应包含钙信号数据和行为标签
                  </div>
                </template>
              </el-upload>
              
              <!-- 文件状态显示 -->
              <div v-if="hasBehaviorFile" class="file-status-display">
                <el-divider content-position="left">
                  <el-icon><Document /></el-icon>
                  已上传文件
                </el-divider>
                
                <div v-for="(file, index) in behaviorFileList" :key="index" class="file-status-item">
                  <div class="file-info">
                    <div class="file-name">
                      <el-icon><DocumentAdd /></el-icon>
                      {{ file.name }}
                    </div>
                    <div class="file-size">
                      {{ (file.size / 1024 / 1024).toFixed(2) }} MB
                    </div>
                  </div>
                  
                  <div class="behavior-detection-status">
                    <div v-if="behaviorLabelsLoading" class="loading-status">
                      <el-icon class="is-loading"><Loading /></el-icon>
                      正在检测行为标签...
                    </div>
                    <div v-else-if="behaviorOptions.length > 0" class="success-status">
                      <div class="detection-result">
                        <el-icon><SuccessFilled /></el-icon>
                        检测到 {{ behaviorOptions.length }} 种行为
                      </div>
                      <div v-if="behaviorParams.start_behavior && behaviorParams.end_behavior" class="selected-behaviors">
                        <div class="selected-behavior-item">
                          <span class="behavior-label">起始:</span>
                          <span class="behavior-value">{{ behaviorParams.start_behavior }}</span>
                        </div>
                        <div class="selected-behavior-item">
                          <span class="behavior-label">结束:</span>
                          <span class="behavior-value">{{ behaviorParams.end_behavior }}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <!-- 分析控制区域 -->
            <div class="analysis-section card">
              <h3 class="section-title">
                <el-icon><DataAnalysis /></el-icon>
                开始分析
              </h3>
              
              <div class="analysis-controls">
                <el-button
                  type="primary"
                  :loading="behaviorAnalysisLoading"
                  :disabled="!hasBehaviorFile || !behaviorParams.start_behavior || !behaviorParams.end_behavior"
                  @click="startBehaviorAnalysis"
                >
                  <el-icon><VideoPlay /></el-icon>
                  开始行为序列分析
                </el-button>
              </div>
            </div>
          </el-col>
        </el-row>
      </el-tab-pane>

      <!-- EM排序热力图 -->
      <el-tab-pane label="EM排序热力图" name="em-sort">
        <el-row :gutter="20">
          <!-- 左侧参数面板 -->
          <el-col :xs="24" :sm="24" :md="8" :lg="6">
            <div class="params-panel card">
              <h3 class="section-title">
                <el-icon><Setting /></el-icon>
                EM排序参数
              </h3>
              

              
              <el-form :model="emSortParams" label-width="120px" size="small">
                <el-form-item label="时间范围开始">
                  <el-input-number
                    v-model="emSortParams.stamp_min"
                    :min="0"
                    :step="1"
                    :precision="2"
                    style="width: 100%"
                    placeholder="留空表示从数据开始"
                  />
                </el-form-item>
                
                <el-form-item label="时间范围结束">
                  <el-input-number
                    v-model="emSortParams.stamp_max"
                    :min="0"
                    :step="1"
                    :precision="2"
                    style="width: 100%"
                    placeholder="留空表示到数据结尾"
                  />
                </el-form-item>
                
                <el-form-item label="排序方式">
                  <el-select
                    v-model="emSortParams.sort_method"
                    style="width: 100%"
                  >
                    <el-option label="按峰值时间排序" value="peak" />
                    <el-option label="按钙波时间排序" value="calcium_wave" />
                    <el-option label="自定义排序" value="custom" />
                  </el-select>
                  <div class="param-help">神经元排序算法</div>
                </el-form-item>
                
                <el-form-item 
                  v-if="emSortParams.sort_method === 'custom'" 
                  label="自定义顺序"
                >
                  <el-input
                    v-model="emSortParams.custom_neuron_order"
                    type="textarea"
                    :rows="3"
                    placeholder="输入神经元ID，用逗号分隔，如：n53,n40,n29"
                  />
                  <div class="param-help">神经元ID列表</div>
                </el-form-item>
                
                <el-form-item label="采样频率">
                  <el-input-number
                    v-model="emSortParams.sampling_rate"
                    :min="0.1"
                    :max="100"
                    :step="0.1"
                    :precision="1"
                    style="width: 100%"
                  />
                  <div class="param-help">单位：Hz</div>
                </el-form-item>
                
                <el-form-item label="钙波阈值">
                  <el-input-number
                    v-model="emSortParams.calcium_wave_threshold"
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
          
          <!-- 右侧内容区域 -->
          <el-col :xs="24" :sm="24" :md="16" :lg="18">
            <!-- 文件上传区域 -->
            <div class="upload-section card">
              <h3 class="section-title">
                <el-icon><Upload /></el-icon>
                数据文件上传
              </h3>
              
              <el-upload
                ref="emSortUploadRef"
                :file-list="emSortFileList"
                :on-change="handleEmSortFileChange"
                :on-remove="handleEmSortFileRemove"
                :before-upload="() => false"
                :auto-upload="false"
                accept=".xlsx,.xls"
                drag
                :limit="1"
                list-type="text"
              >
                <el-icon class="el-icon--upload"><upload-filled /></el-icon>
                <div class="el-upload__text">
                  将文件拖到此处，或<em>点击上传</em>
                </div>
                <template #tip>
                  <div class="el-upload__tip">
                    支持 Excel 文件格式，包含神经元钙信号数据
                  </div>
                </template>
              </el-upload>
            </div>
            
            <!-- 分析控制区域 -->
            <div class="analysis-section card">
              <h3 class="section-title">
                <el-icon><DataAnalysis /></el-icon>
                开始分析
              </h3>
              
              <div class="analysis-controls">
                <el-button
                  type="primary"
                  :loading="emSortAnalysisLoading"
                  :disabled="!emSortFileList.length"
                  @click="startEmSortAnalysis"
                >
                  <el-icon><TrendCharts /></el-icon>
                  开始EM排序分析
                </el-button>
              </div>
            </div>
          </el-col>
        </el-row>
      </el-tab-pane>

      <!-- 多天数据组合热力图 -->
      <el-tab-pane label="多天数据组合热力图" name="multi-day">
        <el-row :gutter="20">
          <!-- 左侧参数面板 -->
          <el-col :xs="24" :sm="24" :md="8" :lg="6">
            <div class="params-panel card">
              <h3 class="section-title">
                <el-icon><Setting /></el-icon>
                多天分析参数
              </h3>
              
              <el-form :model="multiDayParams" label-width="120px" size="small">
                <el-form-item label="排序方式">
                  <el-select
                    v-model="multiDayParams.sort_method"
                    style="width: 100%"
                  >
                    <el-option label="按峰值时间排序" value="peak" />
                    <el-option label="按钙波时间排序" value="calcium_wave" />
                  </el-select>
                  <div class="param-help">神经元排序算法</div>
                </el-form-item>
                
                <el-form-item label="钙波阈值">
                  <el-input-number
                    v-model="multiDayParams.calcium_wave_threshold"
                    :min="0.1"
                    :max="5.0"
                    :step="0.1"
                    :precision="1"
                    style="width: 100%"
                  />
                  <div class="param-help">标准差的倍数</div>
                </el-form-item>
                
                <el-form-item label="生成组合图">
                  <el-switch
                    v-model="multiDayParams.create_combination"
                    active-text="是"
                    inactive-text="否"
                  />
                  <div class="param-help">生成多天对比热力图</div>
                </el-form-item>
                
                <el-form-item label="生成单独图">
                  <el-switch
                    v-model="multiDayParams.create_individual"
                    active-text="是"
                    inactive-text="否"
                  />
                  <div class="param-help">生成每天单独热力图</div>
                </el-form-item>
              </el-form>
            </div>
          </el-col>
          
          <!-- 右侧内容区域 -->
          <el-col :xs="24" :sm="24" :md="16" :lg="18">
            <!-- 文件上传区域 -->
            <div class="upload-section card">
              <h3 class="section-title">
                <el-icon><Upload /></el-icon>
                多天数据文件上传
              </h3>
              
              <el-upload
                ref="multiDayUploadRef"
                :file-list="multiDayFileList"
                :on-change="handleMultiDayFileChange"
                :on-remove="handleMultiDayFileRemove"
                :before-upload="() => false"
                :auto-upload="false"
                accept=".xlsx,.xls"
                drag
                multiple
                :limit="10"
                list-type="text"
              >
                <el-icon class="el-icon--upload"><upload-filled /></el-icon>
                <div class="el-upload__text">
                  将多个文件拖到此处，或<em>点击上传</em>
                </div>
                <template #tip>
                  <div class="el-upload__tip">
                    支持同时上传多个Excel文件，每个文件代表一天的数据
                  </div>
                </template>
              </el-upload>
              
              <!-- 文件标签配置 -->
              <div v-if="multiDayFileList.length > 0" class="file-labels">
                <h4>文件标签配置</h4>
                <el-row :gutter="10">
                  <el-col 
                    v-for="(file, index) in multiDayFileList" 
                    :key="index"
                    :xs="24" :sm="12" :md="8"
                  >
                    <div class="file-label-item">
                      <div class="file-name">{{ file.name }}</div>
                      <el-input
                        v-model="multiDayLabels[index]"
                        placeholder="输入天数标签 (如: day0)"
                        size="small"
                      />
                    </div>
                  </el-col>
                </el-row>
              </div>
            </div>
            
            <!-- 分析控制区域 -->
            <div class="analysis-section card">
              <h3 class="section-title">
                <el-icon><DataAnalysis /></el-icon>
                开始分析
              </h3>
              
              <div class="analysis-controls">
                <el-button
                  type="primary"
                  :loading="multiDayAnalysisLoading"
                  :disabled="!multiDayFileList.length || multiDayLabels.some(label => !label)"
                  @click="startMultiDayAnalysis"
                >
                  <el-icon><Calendar /></el-icon>
                  开始多天对比分析
                </el-button>
              </div>
            </div>
          </el-col>
        </el-row>
      </el-tab-pane>
    </el-tabs>

    <!-- 结果展示区域 -->
    <div v-if="currentResult" class="result-section card">
      <h3 class="section-title">
        <el-icon><PictureRounded /></el-icon>
        分析结果
      </h3>
      
      <!-- 行为序列热力图结果 -->
      <div v-if="activeTab === 'behavior' && behaviorAnalysisResult" class="behavior-result">
        <div class="result-summary">
          <el-descriptions :column="3" border>
            <el-descriptions-item label="分析文件">{{ behaviorAnalysisResult.filename }}</el-descriptions-item>
            <el-descriptions-item label="行为配对数">{{ behaviorAnalysisResult.behavior_pairs_count }}</el-descriptions-item>
            <el-descriptions-item label="神经元数量">{{ behaviorAnalysisResult.neuron_count }}</el-descriptions-item>
            <el-descriptions-item label="起始行为">{{ behaviorAnalysisResult.start_behavior }}</el-descriptions-item>
            <el-descriptions-item label="结束行为">{{ behaviorAnalysisResult.end_behavior }}</el-descriptions-item>
            <el-descriptions-item label="分析状态">{{ behaviorAnalysisResult.status }}</el-descriptions-item>
          </el-descriptions>
        </div>
        
        <div v-if="behaviorAnalysisResult.heatmap_images" class="heatmap-gallery">
          <h4>生成的热力图</h4>
          <el-row :gutter="10">
            <el-col
              v-for="(image, index) in behaviorAnalysisResult.heatmap_images"
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
              </div>
            </el-col>
          </el-row>
        </div>
      </div>
      
      <!-- EM排序热力图结果 -->
      <div v-if="activeTab === 'em-sort' && emSortAnalysisResult" class="em-sort-result">
        <div class="result-summary">
          <el-descriptions :column="3" border>
            <el-descriptions-item label="分析文件">{{ emSortAnalysisResult.filename }}</el-descriptions-item>
            <el-descriptions-item label="排序方式">{{ emSortAnalysisResult.analysis_info.sort_method }}</el-descriptions-item>
            <el-descriptions-item label="神经元数量">{{ emSortAnalysisResult.analysis_info.total_neurons }}</el-descriptions-item>
            <el-descriptions-item label="时间范围">
              {{ emSortAnalysisResult.analysis_info.time_range.start_seconds.toFixed(2) }}s - 
              {{ emSortAnalysisResult.analysis_info.time_range.end_seconds.toFixed(2) }}s
            </el-descriptions-item>
            <el-descriptions-item label="持续时间">{{ emSortAnalysisResult.analysis_info.time_range.duration_seconds.toFixed(2) }}秒</el-descriptions-item>
            <el-descriptions-item label="行为类型数">{{ emSortAnalysisResult.analysis_info.behavior_types.length }}</el-descriptions-item>
          </el-descriptions>
        </div>
        
        <div class="single-heatmap">
          <h4>EM排序热力图</h4>
          <div class="heatmap-container">
            <img 
              :src="emSortAnalysisResult.heatmap_image" 
              alt="EM排序热力图" 
              class="single-heatmap-image" 
              @click="openSingleHeatmapModal(emSortAnalysisResult.heatmap_image, 'EM排序热力图')"
            />
          </div>
        </div>
      </div>
      
      <!-- 多天数据组合热力图结果 -->
      <div v-if="activeTab === 'multi-day' && multiDayAnalysisResult" class="multi-day-result">
        <div class="result-summary">
          <el-descriptions :column="3" border>
            <el-descriptions-item label="处理天数">{{ multiDayAnalysisResult.analysis_info.total_days }}</el-descriptions-item>
            <el-descriptions-item label="排序方式">{{ multiDayAnalysisResult.analysis_info.sort_method }}</el-descriptions-item>
            <el-descriptions-item label="组合图生成">{{ multiDayAnalysisResult.analysis_info.combination_created ? '是' : '否' }}</el-descriptions-item>
            <el-descriptions-item label="单独图生成">{{ multiDayAnalysisResult.analysis_info.individual_created ? '是' : '否' }}</el-descriptions-item>
            <el-descriptions-item label="天数标签">{{ multiDayAnalysisResult.day_labels.join(', ') }}</el-descriptions-item>
            <el-descriptions-item label="分析状态">成功</el-descriptions-item>
          </el-descriptions>
        </div>
        
        <!-- 组合热力图 -->
        <div v-if="multiDayAnalysisResult.combination_heatmap" class="combination-heatmap">
          <div class="section-header-combo">
            <el-icon><DataBoard /></el-icon>
            <h4>多天组合热力图 (多子图并排显示)</h4>
          </div>
          <div class="heatmap-container-combo">
            <img 
              :src="multiDayAnalysisResult.combination_heatmap.image" 
              alt="多天组合热力图" 
              class="combination-heatmap-image" 
              @click="openSingleHeatmapModal(multiDayAnalysisResult.combination_heatmap.image, '多天组合热力图')"
            />
            <div class="combo-description">
              <p><strong>组合热力图特点：</strong></p>
              <ul>
                <li>📊 多天数据并排对比</li>
                <li>🔄 统一的神经元排序</li>
                <li>🎯 便于跨天比较</li>
              </ul>
            </div>
          </div>
        </div>
        
        <!-- 单独热力图 -->
        <div v-if="multiDayAnalysisResult.individual_heatmaps && multiDayAnalysisResult.individual_heatmaps.length > 0" class="individual-heatmaps">
          <div class="section-header-individual">
            <el-icon><PictureRounded /></el-icon>
            <h4>单独热力图 (每天独立显示)</h4>
          </div>
          <div class="individual-description">
            <p><strong>单独热力图特点：</strong></p>
            <ul>
              <li>🔍 每天独立详细展示</li>
              <li>📈 各自数据的独立排序</li>
              <li>🎨 清晰的单日模式</li>
              <li>🔄 根据自身特点排序</li>
            </ul>
          </div>
          <el-row :gutter="15">
            <el-col
              v-for="(heatmap, index) in multiDayAnalysisResult.individual_heatmaps"
              :key="index"
              :xs="24" :sm="12" :md="8"
            >
              <div class="individual-heatmap-item">
                <div class="item-header">
                  <el-icon><Calendar /></el-icon>
                  <span class="day-label">{{ heatmap.day.toUpperCase() }}</span>
                </div>
                <img 
                  :src="heatmap.image" 
                  :alt="heatmap.day + '热力图'" 
                  class="individual-heatmap-image" 
                  @click="openSingleHeatmapModal(heatmap.image, heatmap.day + '热力图')"
                />
                <div class="item-info">
                  <div class="info-item">
                    <span>神经元: {{ heatmap.info.total_neurons }}</span>
                  </div>
                  <div class="info-item">
                    <span>CD1事件: {{ heatmap.info.cd1_events_count }}</span>
                  </div>
                  <div class="info-item">
                    <span>排序: {{ heatmap.info.sort_method }}</span>
                  </div>
                </div>
              </div>
            </el-col>
          </el-row>
        </div>
      </div>
    </div>

    <!-- 热力图放大模态框 -->
    <el-dialog
      v-model="heatmapModalVisible"
      :title="selectedHeatmap?.title || '热力图详情'"
      width="80%"
      class="heatmap-modal"
    >
      <div class="modal-heatmap-container">
        <img 
          v-if="selectedHeatmap?.url"
          :src="selectedHeatmap.url" 
          :alt="selectedHeatmap.title"
          class="modal-heatmap-image"
        />
        <img 
          v-else-if="selectedHeatmapUrl"
          :src="selectedHeatmapUrl" 
          :alt="selectedHeatmapTitle"
          class="modal-heatmap-image"
        />
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, reactive, nextTick, triggerRef, toRefs } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  TrendCharts,
  Setting,
  Upload,
  UploadFilled,
  DataAnalysis,
  VideoPlay,
  PictureRounded,
  Calendar,
  Document,
  DocumentAdd,
  Loading,
  SuccessFilled,
  DataBoard
} from '@element-plus/icons-vue'

// 响应式数据
const activeTab = ref('behavior')

// 默认行为选项（作为备用）
const defaultBehaviorOptions = [
  { label: '破开种子壳', value: 'Crack-seeds-shells' },
  { label: '吃饲料', value: 'Eat-feed' },
  { label: '吃种子仁', value: 'Eat-seed-kernels' },
  { label: '探索', value: 'Explore' },
  { label: '搜索种子', value: 'Explore-search-seeds' },
  { label: '发现种子', value: 'Find-seeds' },
  { label: '获取饲料', value: 'Get-feed' },
  { label: '获取种子', value: 'Get-seeds' },
  { label: '抓取种子', value: 'Grab-seeds' },
  { label: '整理', value: 'Groom' },
  { label: '嗅饲料', value: 'Smell-feed' },
  { label: '嗅种子', value: 'Smell-Get-seeds' },
  { label: '储存种子', value: 'Store-seeds' },
  { label: '饮水', value: 'Water' }
]

// 动态行为选项
const behaviorOptions = ref([...defaultBehaviorOptions])

// 行为序列热力图参数
const behaviorParams = reactive({
  start_behavior: 'Explore',
  end_behavior: 'Water',
  pre_behavior_time: 15.0,
  post_behavior_time: 45.0,
  sampling_rate: 4.8,
  min_behavior_duration: 1.0
})

// EM排序热力图参数
const emSortParams = reactive({
  stamp_min: null,  // 不填时使用整个数据范围
  stamp_max: null,  // 不填时使用整个数据范围
  sort_method: 'peak',
  custom_neuron_order: '',
  sampling_rate: 4.8,
  calcium_wave_threshold: 1.5,
  min_prominence: 1.0,
  min_rise_rate: 0.1,
  max_fall_rate: 0.05
})

// 多天数据组合热力图参数
const multiDayParams = reactive({
  sort_method: 'peak',
  calcium_wave_threshold: 1.5,
  min_prominence: 1.0,
  min_rise_rate: 0.1,
  max_fall_rate: 0.05,
  create_combination: true,
  create_individual: true
})

// 文件列表
const behaviorFileList = ref([])
const emSortFileList = ref([])
const forceUpdateKey = ref(0)
const multiDayFileList = ref([])
const multiDayLabels = ref([])

// 计算属性确保响应式更新
const behaviorFileCount = computed(() => {
  return behaviorFileList.value?.length || 0
})

const hasBehaviorFile = computed(() => {
  return behaviorFileCount.value > 0
})

// EM排序热力图计算属性
const emSortFileCount = computed(() => {
  return emSortFileList.value?.length || 0
})

const hasEmSortFile = computed(() => {
  return emSortFileCount.value > 0
})

// 加载状态
const behaviorAnalysisLoading = ref(false)
const emSortAnalysisLoading = ref(false)
const multiDayAnalysisLoading = ref(false)
const behaviorLabelsLoading = ref(false)

// 分析结果
const behaviorAnalysisResult = ref(null)
const emSortAnalysisResult = ref(null)
const multiDayAnalysisResult = ref(null)

// 模态框相关
const heatmapModalVisible = ref(false)
const selectedHeatmap = ref(null)
const selectedHeatmapUrl = ref('')
const selectedHeatmapTitle = ref('')

// 计算属性
const currentResult = computed(() => {
  switch (activeTab.value) {
    case 'behavior':
      return behaviorAnalysisResult.value
    case 'em-sort':
      return emSortAnalysisResult.value
    case 'multi-day':
      return multiDayAnalysisResult.value
    default:
      return null
  }
})

// 获取行为标签函数
const fetchBehaviorLabels = async (file) => {
  if (!file) return
  
  behaviorLabelsLoading.value = true
  
  try {
    const formData = new FormData()
    formData.append('file', file.raw || file)
    
    const response = await fetch('http://localhost:8000/api/heatmap/behaviors', {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || '获取行为标签失败')
    }
    
    const result = await response.json()
    
    if (result.success && result.behaviors && result.behaviors.length > 0) {
      // 更新行为选项为从数据中提取的标签
      behaviorOptions.value = result.behaviors.map(behavior => ({
        label: behavior,
        value: behavior
      }))
      
      // 自动选择第一个行为作为默认值，让用户可以立即开始分析
      behaviorParams.start_behavior = result.behaviors[0]
      behaviorParams.end_behavior = result.behaviors[0]
      
      ElMessage.success(`成功检测到 ${result.behaviors.length} 种行为标签，已自动选择"${result.behaviors[0]}"作为默认分析行为`)
    } else {
      // 如果没有找到行为标签，使用默认选项
      behaviorOptions.value = [...defaultBehaviorOptions]
      behaviorParams.start_behavior = 'Eat-seed-kernels'
      behaviorParams.end_behavior = 'Eat-seed-kernels'
      ElMessage.warning('未在数据中找到有效的行为标签，使用默认选项')
    }
    
  } catch (error) {
    console.error('获取行为标签失败:', error)
    
    // 出错时使用默认选项，并设置默认行为让用户可以继续操作
    behaviorOptions.value = [...defaultBehaviorOptions]
    behaviorParams.start_behavior = 'Eat-seed-kernels'
    behaviorParams.end_behavior = 'Eat-seed-kernels'
    
    ElMessage.error(`获取行为标签失败: ${error.message}，已使用默认行为选项`)
  } finally {
    behaviorLabelsLoading.value = false
  }
}

// 文件处理函数
const handleBehaviorFileChange = async (file, fileList) => {
  behaviorFileList.value = fileList
  
  // 强制触发响应式更新
  triggerRef(behaviorFileList)
  forceUpdateKey.value++
  
  // 确保DOM更新
  await nextTick()
  
  if (fileList.length > 0) {
    await fetchBehaviorLabels(fileList[0])
  } else {
    // 如果没有文件，恢复默认选项
    behaviorOptions.value = [...defaultBehaviorOptions]
    behaviorParams.start_behavior = 'Eat-seed-kernels'
    behaviorParams.end_behavior = 'Eat-seed-kernels'
  }
}

const handleBehaviorFileRemove = (file, fileList) => {
  behaviorFileList.value = fileList
  
  // 强制触发响应式更新
  triggerRef(behaviorFileList)
  forceUpdateKey.value++
  
  // 如果没有文件，恢复默认选项
  if (fileList.length === 0) {
    behaviorOptions.value = [...defaultBehaviorOptions]
    behaviorParams.start_behavior = 'Eat-seed-kernels'
    behaviorParams.end_behavior = 'Eat-seed-kernels'
  }
}

const handleEmSortFileChange = async (file, fileList) => {
  emSortFileList.value = fileList
  
  if (fileList.length > 0) {
    // 文件上传成功，可以开始分析
    ElMessage.success('文件上传成功，可以开始EM排序分析')
  }
}

const handleEmSortFileRemove = (file, fileList) => {
  emSortFileList.value = fileList
}

const handleMultiDayFileChange = (file, fileList) => {
  multiDayFileList.value = fileList
  // 自动生成标签
  multiDayLabels.value = fileList.map((f, index) => {
    // 尝试从文件名中提取天数信息
    const dayMatch = f.name.match(/day\s*(\d+)/i)
    if (dayMatch) {
      return `day${dayMatch[1]}`
    }
    return `day${index}`
  })
}

const handleMultiDayFileRemove = (file, fileList) => {
  multiDayFileList.value = fileList
  multiDayLabels.value = multiDayLabels.value.slice(0, fileList.length)
}

// 分析函数
const startBehaviorAnalysis = async () => {
  if (behaviorFileList.value.length === 0) {
    ElMessage.warning('请先上传数据文件')
    return
  }
  
  if (!behaviorParams.start_behavior || !behaviorParams.end_behavior) {
    ElMessage.warning('请选择起始和结束行为')
    return
  }
  
  behaviorAnalysisLoading.value = true
  
  try {
    const formData = new FormData()
    formData.append('file', behaviorFileList.value[0].raw)
    formData.append('start_behavior', behaviorParams.start_behavior)
    formData.append('end_behavior', behaviorParams.end_behavior)
    formData.append('pre_behavior_time', behaviorParams.pre_behavior_time)
    formData.append('post_behavior_time', behaviorParams.post_behavior_time)
    formData.append('min_duration', behaviorParams.min_behavior_duration)
    formData.append('sampling_rate', behaviorParams.sampling_rate)
    
    const response = await fetch('http://localhost:8000/api/heatmap/analyze', {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || '分析失败')
    }
    
    const result = await response.json()
    behaviorAnalysisResult.value = result
    
    if (result.heatmap_images && result.heatmap_images.length > 0) {
      ElMessage.success(`行为序列热力图分析完成！生成了 ${result.heatmap_images.length} 张热力图`)
    } else {
      ElMessage.warning('分析完成，但没有生成图像')
    }
    
  } catch (error) {
    console.error('行为序列分析失败:', error)
    ElMessage.error('行为序列分析失败: ' + (error.message || '未知错误'))
  } finally {
    behaviorAnalysisLoading.value = false
  }
}

const startEmSortAnalysis = async () => {
  if (emSortFileList.value.length === 0) {
    ElMessage.warning('请先上传数据文件')
    return
  }
  
  emSortAnalysisLoading.value = true
  
  try {
    const formData = new FormData()
    formData.append('file', emSortFileList.value[0].raw)
    
    // 只有当值不为null时才添加到FormData中
    if (emSortParams.stamp_min !== null) {
      formData.append('stamp_min', emSortParams.stamp_min)
    }
    if (emSortParams.stamp_max !== null) {
      formData.append('stamp_max', emSortParams.stamp_max)
    }
    
    formData.append('sort_method', emSortParams.sort_method)
    if (emSortParams.custom_neuron_order) {
      formData.append('custom_neuron_order', emSortParams.custom_neuron_order)
    }
    formData.append('sampling_rate', emSortParams.sampling_rate.toString())
    formData.append('calcium_wave_threshold', emSortParams.calcium_wave_threshold.toString())
    formData.append('min_prominence', emSortParams.min_prominence.toString())
    formData.append('min_rise_rate', emSortParams.min_rise_rate.toString())
    formData.append('max_fall_rate', emSortParams.max_fall_rate.toString())
    
    const response = await fetch('http://localhost:8000/api/heatmap/em-sort', {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || 'EM排序分析失败')
    }
    
    const result = await response.json()
    emSortAnalysisResult.value = result
    
    ElMessage.success('EM排序热力图分析完成！')
    
  } catch (error) {
    console.error('EM排序分析失败:', error)
    ElMessage.error('EM排序分析失败: ' + (error.message || '未知错误'))
  } finally {
    emSortAnalysisLoading.value = false
  }
}

const startMultiDayAnalysis = async () => {
  if (multiDayFileList.value.length === 0) {
    ElMessage.warning('请先上传多天数据文件')
    return
  }
  
  if (multiDayLabels.value.some(label => !label)) {
    ElMessage.warning('请为所有文件设置天数标签')
    return
  }
  
  multiDayAnalysisLoading.value = true
  
  try {
    const formData = new FormData()
    
    // 添加文件
    multiDayFileList.value.forEach(file => {
      formData.append('files', file.raw)
    })
    
    // 添加参数
    formData.append('day_labels', multiDayLabels.value.join(','))
    formData.append('sort_method', multiDayParams.sort_method)
    formData.append('calcium_wave_threshold', multiDayParams.calcium_wave_threshold)
    formData.append('min_prominence', multiDayParams.min_prominence)
    formData.append('min_rise_rate', multiDayParams.min_rise_rate)
    formData.append('max_fall_rate', multiDayParams.max_fall_rate)
    formData.append('create_combination', multiDayParams.create_combination)
    formData.append('create_individual', multiDayParams.create_individual)
    
    const response = await fetch('http://localhost:8000/api/heatmap/multi-day', {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || '多天分析失败')
    }
    
    const result = await response.json()
    multiDayAnalysisResult.value = result
    
    ElMessage.success(`多天数据热力图分析完成！处理了 ${result.day_labels.length} 天的数据`)
    
  } catch (error) {
    console.error('多天分析失败:', error)
    ElMessage.error('多天分析失败: ' + (error.message || '未知错误'))
  } finally {
    multiDayAnalysisLoading.value = false
  }
}

// 模态框函数
const openHeatmapModal = (heatmap, index) => {
  selectedHeatmap.value = heatmap
  selectedHeatmapUrl.value = ''
  selectedHeatmapTitle.value = ''
  heatmapModalVisible.value = true
}

const openSingleHeatmapModal = (imageUrl, title) => {
  selectedHeatmap.value = null
  selectedHeatmapUrl.value = imageUrl
  selectedHeatmapTitle.value = title
  heatmapModalVisible.value = true
}
</script>

<style scoped>
.heatmap {
  padding: 20px;
  background-color: #f5f7fa;
  min-height: 100vh;
}

.page-title {
  color: #303133;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.info-alert {
  margin-bottom: 20px;
}

.analysis-tabs {
  margin-bottom: 20px;
}

.card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
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

.analysis-controls {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.file-labels {
  margin-top: 20px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 6px;
}

.file-label-item {
  margin-bottom: 10px;
}

.file-name {
  font-size: 12px;
  color: #606266;
  margin-bottom: 5px;
  word-break: break-all;
}

/* 文件状态显示样式 */
.file-status-display {
  margin-top: 20px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 6px;
  border: 1px solid #e4e7ed;
}

.file-status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  background-color: white;
  border-radius: 6px;
  margin-bottom: 10px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.file-status-item:last-child {
  margin-bottom: 0;
}

.file-info {
  flex: 1;
}

.file-info .file-name {
  font-size: 14px;
  font-weight: 500;
  color: #303133;
  margin-bottom: 3px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.file-info .file-size {
  font-size: 12px;
  color: #909399;
}

.behavior-detection-status {
  flex-shrink: 0;
  text-align: right;
}

.loading-status {
  color: #409eff;
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.success-status {
  color: #67c23a;
  font-size: 12px;
}

.detection-result {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 6px;
}

.selected-behaviors {
  margin-top: 6px;
  font-size: 11px;
}

.selected-behavior-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 2px;
}

.behavior-label {
  color: #909399;
  font-weight: normal;
}

.behavior-value {
  color: #303133;
  font-weight: 500;
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.is-loading {
  animation: rotating 2s linear infinite;
}

@keyframes rotating {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.heatmap-gallery {
  margin-top: 20px;
}

.heatmap-item {
  margin-bottom: 15px;
  text-align: center;
}

.heatmap-image {
  width: 100%;
  height: auto;
  border-radius: 6px;
  cursor: pointer;
  transition: transform 0.3s ease;
}

.heatmap-image:hover {
  transform: scale(1.05);
}

.heatmap-title {
  margin-top: 8px;
  font-size: 14px;
  color: #606266;
  font-weight: 500;
}

.single-heatmap,
.combination-heatmap {
  margin-top: 20px;
}

.heatmap-container {
  text-align: center;
  padding: 20px;
  background-color: #f8f9fa;
  border-radius: 6px;
}

.single-heatmap-image,
.combination-heatmap-image {
  max-width: 100%;
  height: auto;
  border-radius: 6px;
  cursor: pointer;
  transition: transform 0.3s ease;
}

.single-heatmap-image:hover,
.combination-heatmap-image:hover {
  transform: scale(1.02);
}

/* 组合热力图样式增强 */
.section-header-combo {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
  padding: 15px;
  background: linear-gradient(135deg, #e3f2fd, #bbdefb);
  border-left: 5px solid #2196f3;
  border-radius: 8px;
}

.section-header-combo h4 {
  margin: 0;
  color: #1565c0;
  font-weight: 600;
}

.heatmap-container-combo {
  background: linear-gradient(135deg, #f8f9fa, #e9ecef);
  border: 2px solid #2196f3;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
}

.combo-description {
  margin-top: 15px;
  padding: 15px;
  background: #fff;
  border-radius: 8px;
  border-left: 4px solid #2196f3;
}

.combo-description p {
  margin: 0 0 10px 0;
  color: #1565c0;
  font-size: 16px;
}

.combo-description ul {
  margin: 0;
  padding-left: 20px;
}

.combo-description li {
  margin-bottom: 5px;
  color: #424242;
  font-size: 14px;
}

/* 单独热力图样式增强 */
.section-header-individual {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
  padding: 15px;
  background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
  border-left: 5px solid #4caf50;
  border-radius: 8px;
}

.section-header-individual h4 {
  margin: 0;
  color: #2e7d32;
  font-weight: 600;
}

.individual-description {
  margin-bottom: 20px;
  padding: 15px;
  background: #fff;
  border-radius: 8px;
  border-left: 4px solid #4caf50;
}

.individual-description p {
  margin: 0 0 10px 0;
  color: #2e7d32;
  font-size: 16px;
}

.individual-description ul {
  margin: 0;
  padding-left: 20px;
}

.individual-description li {
  margin-bottom: 5px;
  color: #424242;
  font-size: 14px;
}

.individual-heatmaps {
  margin-top: 20px;
}

.individual-heatmap-item {
  margin-bottom: 20px;
  background: #fff;
  border: 2px solid #4caf50;
  border-radius: 12px;
  overflow: hidden;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(76, 175, 80, 0.1);
}

.individual-heatmap-item:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(76, 175, 80, 0.2);
}

.item-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 15px;
  background: linear-gradient(135deg, #4caf50, #66bb6a);
  color: white;
  font-weight: 600;
}

.day-label {
  font-size: 16px;
  letter-spacing: 1px;
}

.individual-heatmap-image {
  width: 100%;
  height: auto;
  display: block;
  transition: transform 0.3s ease;
}

.individual-heatmap-image:hover {
  transform: scale(1.02);
}

.item-info {
  padding: 12px 15px;
  background: #f8f9fa;
  display: flex;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap;
}

.info-item {
  display: flex;
  align-items: center;
  font-size: 12px;
  color: #666;
  font-weight: 500;
  flex: 1;
  min-width: 80px;
  justify-content: center;
  padding: 4px 8px;
  background: #fff;
  border-radius: 4px;
  border: 1px solid #e0e0e0;
}

.modal-heatmap-container {
  text-align: center;
}

.modal-heatmap-image {
  max-width: 100%;
  height: auto;
  border-radius: 6px;
}

.result-summary {
  margin-bottom: 20px;
}

@media (max-width: 768px) {
  .heatmap {
    padding: 10px;
  }
  
  .analysis-controls {
    flex-direction: column;
  }
  
  .analysis-controls .el-button {
    width: 100%;
  }
}
</style>