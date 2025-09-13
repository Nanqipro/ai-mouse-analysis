# 热力图分析模块集成总结

## 📋 已完成的工作

### 1. ✅ 集成 heatmap_sort_EM.py 功能 - EM排序热力图
**文件**: `/backend/src/heatmap_em_sort.py`

**功能特点**:
- 支持多种排序方式：峰值时间、钙波时间、自定义顺序
- 时间区间筛选功能
- 行为标记和区间显示
- 高质量热力图生成
- 完整的中文注释和参数配置

**主要类和函数**:
- `EMSortHeatmapConfig`: 配置类
- `analyze_em_sort_heatmap()`: 主分析函数
- `detect_first_calcium_wave()`: 钙波检测
- `sort_neurons_by_custom_order()`: 自定义排序

### 2. ✅ 集成 heatmap_comb-sort.py 功能 - 多天数据组合热力图
**文件**: `/backend/src/heatmap_multi_day.py`

**功能特点**:
- 多天神经元数据的对齐和比较分析
- 基于神经元对应表的数据映射
- 支持峰值时间和钙波时间排序
- CD1行为事件标记
- 生成组合热力图和单独热力图

**主要类和函数**:
- `MultiDayHeatmapConfig`: 配置类
- `analyze_multiday_heatmap()`: 主分析函数
- `create_multiday_combination_heatmap()`: 组合热力图创建
- `create_single_day_heatmap()`: 单日热力图创建

### 3. ✅ 优化现有 heatmap_behavior.py 功能 - 行为序列热力图
**文件**: `/backend/src/heatmap_behavior.py` (已存在，保持完善)

**功能特点**:
- 特定行为前后时间窗口分析
- 行为连续性检测
- 多种神经元排序方式
- 个体序列和平均序列热力图

### 4. ✅ 更新后端API接口
**文件**: `/backend/main.py`

**新增路由**:
1. `POST /api/heatmap/em-sort` - EM排序热力图分析
2. `POST /api/heatmap/multi-day` - 多天数据组合热力图分析
3. `POST /api/heatmap/analyze` - 行为序列热力图分析 (已存在)

**API特点**:
- 完整的参数验证
- 错误处理和日志记录
- Base64图像编码返回
- 详细的分析信息返回

### 5. ✅ 更新前端界面
**文件**: `/frontend/src/views/Heatmap.vue`

**界面特点**:
- 选项卡式界面设计，支持三种分析类型
- 每种分析类型独立的参数配置
- 文件上传和标签管理
- 结果展示和图像放大功能
- 响应式设计，支持移动端

## 🔧 三种热力图功能对比

| 功能特点 | 行为序列热力图 | EM排序热力图 | 多天数据组合热力图 |
|---------|-------------|-------------|-----------------|
| **主要用途** | 分析特定行为的神经活动模式 | 基于神经元特征排序的整体分析 | 多天实验数据的对比分析 |
| **输入数据** | 单个文件(含行为标签) | 单个文件(神经元数据) | 多个文件(多天数据) |
| **排序方式** | 全局/局部/首图/自定义 | 峰值/钙波/自定义 | 峰值/钙波 |
| **时间筛选** | 行为相关时间窗口 | 可自定义时间区间 | 全时间范围 |
| **行为标记** | 行为开始/结束标记 | 所有行为区间标记 | CD1行为标记 |
| **输出图像** | 个体序列+平均图 | 单个排序热力图 | 组合图+单独图 |

## 🚀 如何启动和测试

### 1. 安装依赖
```bash
# 后端依赖
cd backend
pip install -r requirements.txt

# 前端依赖
cd frontend
npm install
```

### 2. 启动服务
```bash
# 启动后端服务 (端口8000)
cd backend
python main.py

# 启动前端服务 (端口5173)
cd frontend
npm run dev
```

### 3. 测试步骤

#### 测试行为序列热力图:
1. 选择"行为序列热力图"选项卡
2. 上传包含行为标签的Excel文件
3. 选择起始和结束行为
4. 设置分析参数
5. 点击"开始行为序列分析"

#### 测试EM排序热力图:
1. 选择"EM排序热力图"选项卡
2. 上传神经元数据文件
3. 设置时间范围和排序方式
4. 如果选择自定义排序，输入神经元ID列表
5. 点击"开始EM排序分析"

#### 测试多天数据组合热力图:
1. 选择"多天数据组合热力图"选项卡
2. 上传多个天数的数据文件
3. 为每个文件设置天数标签 (如day0, day3等)
4. 设置分析参数
5. 点击"开始多天对比分析"

## 📁 项目结构

```
ai-mouse-analysis/
├── backend/
│   ├── src/
│   │   ├── heatmap_behavior.py      # 行为序列热力图
│   │   ├── heatmap_em_sort.py       # EM排序热力图
│   │   ├── heatmap_multi_day.py     # 多天组合热力图
│   │   └── overall_heatmap.py       # 整体热力图 (原有)
│   └── main.py                      # FastAPI主应用
├── frontend/
│   └── src/
│       └── views/
│           └── Heatmap.vue          # 热力图分析界面
└── algorithm/
    └── heatmap/                     # 原始算法文件
        ├── heatmap_sort_EM.py       # 已集成
        ├── heatmap_comb-sort.py     # 已集成
        └── heatmap_behavior.py      # 已集成
```

## 🎯 API 接口文档

### 1. 行为序列热力图分析
**POST** `/api/heatmap/analyze`

**参数**:
- `file`: 数据文件
- `start_behavior`: 起始行为
- `end_behavior`: 结束行为
- `pre_behavior_time`: 行为前时间(秒)
- `min_duration`: 最小持续时间(秒)
- `sampling_rate`: 采样频率(Hz)

### 2. EM排序热力图分析
**POST** `/api/heatmap/em-sort`

**参数**:
- `file`: 数据文件
- `stamp_min`: 最小时间戳 (可选)
- `stamp_max`: 最大时间戳 (可选)
- `sort_method`: 排序方式 (peak/calcium_wave/custom)
- `custom_neuron_order`: 自定义神经元顺序 (可选)
- `sampling_rate`: 采样频率(Hz)
- `calcium_wave_threshold`: 钙波阈值

### 3. 多天数据组合热力图分析
**POST** `/api/heatmap/multi-day`

**参数**:
- `files`: 多个数据文件
- `day_labels`: 天数标签 (逗号分隔)
- `sort_method`: 排序方式 (peak/calcium_wave)
- `calcium_wave_threshold`: 钙波阈值
- `create_combination`: 是否生成组合图
- `create_individual`: 是否生成单独图

## 📊 返回数据格式

所有API都返回包含以下字段的JSON:
- `success`: 是否成功
- `filename/filenames`: 文件名
- `heatmap_image/heatmap_images`: Base64编码的图像
- `analysis_info`: 分析信息
- `config`: 使用的配置参数
- `message`: 状态消息

## 🔍 故障排除

### 常见问题:
1. **导入错误**: 确保所有依赖包已安装
2. **文件上传失败**: 检查文件格式和大小限制
3. **分析失败**: 检查数据格式和参数设置
4. **图像显示异常**: 检查Base64编码和网络连接

### 调试建议:
1. 查看浏览器控制台错误信息
2. 检查后端服务日志
3. 验证数据文件格式
4. 测试API接口响应

## 🎉 总结

成功集成了三种不同的热力图分析功能：
1. **行为序列热力图** - 专注于特定行为的时序分析
2. **EM排序热力图** - 提供灵活的神经元排序和时间筛选
3. **多天数据组合热力图** - 支持多天实验数据的对比分析

每种功能都有独立的界面、参数配置和分析逻辑，用户可以根据研究需求选择合适的分析方法。所有功能都遵循统一的代码规范，包含完整的中文注释和错误处理。
