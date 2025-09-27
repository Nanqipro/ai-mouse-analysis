# Principal Neuron 项目使用文档

## 项目概述

Principal Neuron 是一个专门用于分析小鼠脑神经元活动与行为关联的综合性可视化分析工具。该项目主要专注于识别和分析在特定行为状态下起关键作用的神经元，通过效应量计算、空间位置可视化和神经元关系分析来揭示神经活动模式。

## 🎯 核心功能

### 1. **效应量计算 (Effect Size Calculation)**
- **Cohen's d 效应量计算**：量化神经元对特定行为的判别能力
- **从原始数据到效应量**：完整的数据预处理和效应量计算流程
- **关键神经元识别**：基于效应量阈值自动识别关键神经元
- **阈值优化**：自动推荐最佳效应量阈值

### 2. **空间位置可视化**
- **二维空间映射**：在神经元的真实相对空间位置上可视化神经元活动
- **背景神经元显示**：显示所有神经元作为背景，突出关键神经元
- **颜色编码系统**：不同行为使用不同颜色标识

### 3. **神经元关系分析**
- **单一行为分析**：每种行为（Close、Middle、Open）的关键神经元分布
- **共享神经元分析**：两两行为间共享的关键神经元
- **特有神经元分析**：仅在特定行为中活跃的特有神经元

## 📁 项目结构

```
principal_neuron/
├── src/                           # 源代码目录
│   ├── effect_size_calculator.py    # 🆕 效应量计算器（从analysis_results.py集成）
│   ├── main_emtrace01_analysis.py   # 主分析脚本
│   ├── data_loader.py              # 数据加载模块
│   ├── config.py                   # 配置文件
│   ├── plotting_utils.py           # 绘图工具
│   └── ...
├── data/                          # 数据目录
│   ├── EMtrace01_raw_data.xlsx     # 原始神经元数据（可选）
│   ├── EMtrace01-3标签版.csv       # 预计算的效应量数据
│   └── EMtrace01_Max_position.csv  # 神经元位置数据
├── output_plots/                  # 输出图表目录
├── app.py                         # Web应用入口
└── README_USAGE.md               # 使用文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装必要的Python包
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. 数据准备

支持两种数据输入方式：

#### 方式一：原始神经元数据
- 文件格式：Excel (.xlsx) 或 CSV (.csv)
- 数据结构：
  ```
  神经元1 | 神经元2 | ... | 神经元N | 行为标签
  0.234   | 0.567   | ... | 0.123   | Close
  0.345   | 0.678   | ... | 0.234   | Middle
  ...
  ```

#### 方式二：预计算的效应量数据
- 文件格式：CSV
- 数据结构：
  ```
  Behavior | Neuron_1 | Neuron_2 | ... | Neuron_N
  Close    | 0.234    | 0.567    | ... | 0.123
  Middle   | 0.345    | 0.678    | ... | 0.234
  Open     | 0.456    | 0.789    | ... | 0.345
  ```

#### 神经元位置数据
- 文件格式：CSV
- 数据结构：
  ```
  NeuronID | X | Y
  1        | 100 | 200
  2        | 150 | 250
  ...
  ```

### 3. 运行分析

#### 基本运行
```bash
cd principal_neuron/src
python main_emtrace01_analysis.py
```

#### 自定义分析流程

```python
from effect_size_calculator import EffectSizeCalculator, load_and_calculate_effect_sizes
import pandas as pd
import numpy as np

# 方法1：从原始数据计算效应量
results = load_and_calculate_effect_sizes(
    neuron_data_path="data/raw_neuron_data.xlsx",
    behavior_col="behavior",  # 行为标签列名，None则使用最后一列
    output_dir="output"
)

# 方法2：使用效应量计算器类
calculator = EffectSizeCalculator()

# 假设已有神经元数据和行为标签
neuron_data = np.random.randn(1000, 50)  # 1000个样本，50个神经元
behavior_data = np.random.choice(['Close', 'Middle', 'Open'], 1000)

# 计算效应量
effect_sizes, X_scaled, y_encoded = calculator.calculate_effect_sizes_from_raw_data(
    neuron_data, behavior_data
)

# 识别关键神经元
key_neurons = calculator.identify_key_neurons(effect_sizes, threshold=0.4)

# 获取top神经元
top_neurons = calculator.get_top_neurons_per_behavior(effect_sizes, top_n=10)

# 导出结果
calculator.export_effect_sizes_to_csv(effect_sizes, "output/effect_sizes.csv")
```

## ⚙️ 配置选项

在 `config.py` 中可以调整以下参数：

```python
# 效应量阈值
EFFECT_SIZE_THRESHOLD = 0.4

# 行为颜色配置
BEHAVIOR_COLORS = {
    'Close': 'red',
    'Middle': 'blue', 
    'Open': 'green'
}

# 可视化参数
SHOW_BACKGROUND_NEURONS = True
BACKGROUND_NEURON_COLOR = 'lightgray'
BACKGROUND_NEURON_SIZE = 20
```

## 📊 输出结果

### 1. 效应量数据文件
- `effect_sizes.csv`: 完整的效应量数据表
- `neuron_ranking_by_effect_size.csv`: 按效应量排序的神经元列表

### 2. 可视化图表
- `effect_size_histogram.png`: 效应量分布直方图
- `effect_size_boxplot.png`: 效应量箱线图
- `plot_all_behaviors_3x3_grid.png`: 3x3网格综合分析图
- 各种行为分析图表（单一行为、共享神经元、特有神经元等）

### 3. 分析报告
程序会在控制台输出详细的分析结果，包括：
- 每种行为的关键神经元列表
- 效应量统计信息
- 阈值推荐建议

## 🔬 算法原理

### Cohen's d 效应量计算

对于每种行为 $B_i$ 和每个神经元 $N_j$，计算效应量：

$$d = \frac{|\mu_{B_i} - \mu_{other}|}{\sigma_{pooled}}$$

其中：
- $\mu_{B_i}$：行为 $B_i$ 下神经元 $N_j$ 的平均活动
- $\mu_{other}$：其他行为下神经元 $N_j$ 的平均活动  
- $\sigma_{pooled}$：合并标准差

$$\sigma_{pooled} = \sqrt{\frac{\sigma_{B_i}^2 + \sigma_{other}^2}{2}}$$

### 关键神经元识别

神经元被识别为关键神经元的条件：
$$d_{ij} \geq \theta$$

其中 $\theta$ 是效应量阈值（默认0.4）。

## 📈 高级功能

### 1. 自动阈值推荐

程序可以根据期望的神经元数量范围自动推荐最佳阈值：

```python
from main_emtrace01_analysis import suggest_threshold_for_neuron_count

# 推荐阈值，使每种行为有5-10个关键神经元
threshold = suggest_threshold_for_neuron_count(
    df_effects, 
    min_neurons=5, 
    max_neurons=10
)
```

### 2. 效应量分布分析

```python
from main_emtrace01_analysis import analyze_effect_sizes

# 分析效应量分布并生成可视化
analyze_effect_sizes(df_effect_sizes_long)
```

### 3. Web应用界面

启动交互式Web应用：

```bash
python app.py
```

然后在浏览器中访问 `http://localhost:5000`

## 🛠️ 故障排除

### 常见问题

1. **效应量计算失败**
   - 检查原始数据格式是否正确
   - 确保行为标签列存在且格式正确
   - 验证神经元数据是否为数值型

2. **可视化图表为空**
   - 检查神经元位置数据是否加载正确
   - 验证效应量阈值是否合适
   - 确认选定的神经元在位置数据中存在

3. **内存不足错误**
   - 减少数据集大小或分批处理
   - 降低图表分辨率设置

### 调试模式

在代码中添加详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 现在所有函数都会输出详细的调试信息
```

## 📚 扩展开发

### 添加新的行为类别

1. 在数据中添加新的行为标签
2. 在 `config.py` 中添加对应的颜色配置
3. 无需修改其他代码，系统会自动适应

### 自定义可视化

参考 `plotting_utils.py` 中的函数，可以：
- 修改颜色方案
- 调整图表布局
- 添加新的可视化类型

### 集成到其他项目

```python
# 将principal_neuron作为模块导入
from principal_neuron.src.effect_size_calculator import EffectSizeCalculator
from principal_neuron.src.main_emtrace01_analysis import analyze_effect_sizes

# 在你的项目中使用
calculator = EffectSizeCalculator()
# ... 你的分析代码
```

## 📞 支持与贡献

如有问题或建议，请：
1. 检查本文档的故障排除部分
2. 查看代码中的注释和文档字符串
3. 提交issue或pull request

## 🔄 更新日志

### v2.0 (最新)
- ✅ 集成了完整的效应量计算功能
- ✅ 支持从原始数据直接计算效应量
- ✅ 增加了自动阈值推荐功能
- ✅ 完善了数据预处理流程
- ✅ 改进了错误处理和用户体验

### v1.0
- 基础的可视化功能
- 预计算效应量数据支持
- 空间位置映射 