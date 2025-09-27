# 小鼠脑神经元行为关联可视化工具 (v2.0)

本项目旨在分析和可视化小鼠在特定行为（如"Close"，"Middle"，"Open"）中关键神经元的活动数据。
它通过**完整的效应量计算流程**从原始神经元数据到效应量分析，识别关键神经元，并在神经元的相对空间位置上绘制这些神经元，以及它们之间的共享和特有关系。

## 🆕 新增功能 (v2.0)

- ✅ **完整效应量计算**：集成了从 `analysis_results.py` 的 Cohen's d 效应量计算功能
- ✅ **原始数据支持**：支持直接从原始神经元活动数据计算效应量
- ✅ **智能工作流**：自动判断使用原始数据计算还是加载预计算效应量
- ✅ **阈值优化**：自动推荐最佳效应量阈值
- ✅ **完整文档**：详细的使用文档和示例代码

## 项目结构

```
principal_neuron/
├── data/                     # 存放数据文件
│   ├── EMtrace01_raw_data.xlsx      # 🆕 原始神经元数据 (可选)
│   ├── EMtrace01-3标签版.csv        # 预计算的效应量数据
│   └── EMtrace01_Max_position.csv   # 神经元相对位置数据
├── output_plots/             # 生成的图表将保存于此
├── src/                      # 源代码目录
│   ├── effect_size_calculator.py    # 🆕 效应量计算器 (集成自analysis_results.py)
│   ├── main_emtrace01_analysis.py   # 主分析脚本 (已更新工作流)
│   ├── data_loader.py              # 数据加载和初步处理模块
│   ├── config.py                   # 配置文件 (阈值、颜色等)
│   ├── plotting_utils.py           # 绘图工具函数模块
│   └── __init__.py                 # (可选, 使src成为一个包)
├── example_usage.py          # 🆕 使用示例和教程
├── README_USAGE.md           # 🆕 详细使用文档
├── app.py                    # Web应用入口
└── README.md                 # 本说明文件
```

## 🚀 快速开始

### 方式1：使用原始神经元数据 (推荐)

```bash
# 1. 环境准备
pip install pandas numpy matplotlib seaborn scikit-learn

# 2. 准备原始数据
# 将您的原始神经元数据文件 (Excel/CSV格式) 放入 data/ 目录
# 数据格式: 神经元1 | 神经元2 | ... | 行为标签

# 3. 运行分析
python src/main_emtrace01_analysis.py
```

### 方式2：使用预计算效应量数据

```bash
# 1. 准备预计算效应量文件
# 将效应量数据文件命名为 EMtrace01-3标签版.csv 放入 data/ 目录

# 2. 运行分析
python src/main_emtrace01_analysis.py
```

### 方式3：运行示例代码

```bash
# 运行完整的使用示例
python example_usage.py
```

## 📚 详细文档

**完整使用说明请参阅 [README_USAGE.md](README_USAGE.md)**，包含：
- 详细的数据格式说明
- 完整的 API 使用指南
- 效应量计算原理
- 高级功能使用
- 故障排除指南

## 如何运行 (详细说明)

1.  **环境准备**:
    *   确保已安装 Python 和必要的库 (pandas, numpy, matplotlib, seaborn, scikit-learn)。
    *   可以通过 `pip install pandas numpy matplotlib seaborn scikit-learn` 进行安装。

2.  **数据准备**:
    
    **选项A: 原始神经元数据 (🆕 推荐)**
    *   准备包含神经元活动数据和行为标签的Excel或CSV文件
    *   数据格式：每行一个样本，列为神经元活动值，最后一列为行为标签
    *   将文件放入 `data/` 文件夹，程序会自动检测并计算效应量
    
    **选项B: 预计算效应量数据**
    *   将您的神经元效应大小数据文件命名为 `EMtrace01-3标签版.csv` 并放入 `data` 文件夹。
        *   该文件应包含行为名称，以及对应不同神经元 (`Neuron_X`) 的效应大小值。
    
    **神经元位置数据 (必需)**
    *   将您的神经元位置数据文件命名为 `EMtrace01_Max_position.csv` 并放入 `data` 文件夹。
        *   该文件应包含神经元编号 (`number`) 及其相对X (`relative_x`) 和Y (`relative_y`) 坐标。

3.  **运行脚本**:
    在项目根目录下执行以下命令：
    ```bash
    python src/main_emtrace01_analysis.py
    ```
    生成的图表将保存在 `output_plots` 文件夹中。

## 配置说明 (`src/config.py`)

*   `EFFECT_SIZE_THRESHOLD`: 用于筛选关键神经元的效应大小阈值。默认根据数据分析建议设置为 `0.4407`。
*   `BEHAVIOR_COLORS`: 定义了不同行为在图表中的基础颜色。
    *   示例: `{'Close': 'red', 'Middle': 'green', 'Open': 'blue'}`
*   `MIXED_BEHAVIOR_COLORS`: 定义了行为对共享神经元在Scheme B图中的混合颜色。
    *   键为按字母顺序排序的行为名称元组。
    *   示例: ` {'Close', 'Middle'): 'yellow', ('Close', 'Open'): 'magenta', ...}`

*   `STANDARD_KEY_NEURON_ALPHA` (浮点数, 默认 `0.7`): 关键神经元（在不进行特殊淡化处理时）的标准透明度。

*   `SHOW_BACKGROUND_NEURONS` (布尔值, 默认 `True`): 控制是否在图中绘制所有神经元作为背景点。设置为 `False` 则只绘制关键/共享/特有神经元。
*   `BACKGROUND_NEURON_COLOR` (字符串, 默认 `'lightgray'`): 背景神经元的颜色。
*   `BACKGROUND_NEURON_SIZE` (整数, 默认 `75`): 背景神经元的标记点大小。(之前是20，已根据用户反馈调整)
*   `BACKGROUND_NEURON_ALPHA` (浮点数, 默认 `0.3`): 背景神经元的透明度。

*   `USE_STANDARD_ALPHA_FOR_UNSHARED_IN_SCHEME_B` (布尔值, 默认 `True`): 在"两行为间共享的关键神经元图 (Scheme B)"中，控制非共享的关键神经元是否使用 `STANDARD_KEY_NEURON_ALPHA` (即不淡化)。设置为 `False` 则会使用一个较低的透明度值 (由 `plotting_utils.py` 中 `plot_shared_neurons_map` 函数的 `alpha_non_shared` 参数定义，默认为0.3) 来淡化显示这些非共享神经元。

## 绘图功能及选项 (`src/plotting_utils.py`)

脚本会生成以下主要的图表（部分图表现在会组合显示）：

1.  **图1-3: 单一行为的关键神经元图** (独立文件仍会生成)
    *   文件名示例: `plot_close_key_neurons.png`
    *   显示特定行为中，效应大小超过阈值的关键神经元在其相对位置的分布。
    *   选项 (`plot_single_behavior_activity_map` 函数):
        *   `show_title` (布尔值, 默认 `True`): 控制是否显示图表标题。
        *   `show_background_neurons` (布尔值, 通过 `config.py` 中的 `SHOW_BACKGROUND_NEURONS` 控制): 是否显示所有神经元作为背景。

2.  **图4-6: 两行为间共享的关键神经元图** (独立文件仍会生成)
    *   文件名示例: `plot_shared_close_middle_schemeB.png`
    *   显示两两行为之间共享的关键神经元。
    *   选项 (`plot_shared_neurons_map` 函数):
        *   `scheme` (字符串, 默认 `'B'`):
            *   `'A'`: 仅显示共享的神经元。
            *   `'B'`: 显示两个行为的所有关键神经元，并高亮显示共享的神经元。默认情况下（通过 `USE_STANDARD_ALPHA_FOR_UNSHARED_IN_SCHEME_B` 配置为 `True`），非共享的关键神经元将以其标准透明度显示。如果该配置设为 `False`，则非共享部分会以较低透明度进行淡化处理。
        *   `show_title` (布尔值, 默认 `True`): 控制是否显示图表标题。
        *   `show_background_neurons` (布尔值, 通过 `config.py` 中的 `SHOW_BACKGROUND_NEURONS` 控制): 是否显示所有神经元作为背景。
        *   可在 `main_emtrace01_analysis.py` 中修改 `scheme_to_use` 变量来切换方案，或为不同方案生成不同的图。

3.  **图7-9: 单一行为的特有关键神经元图** (独立文件仍会生成)
    *   文件名示例: `plot_unique_close_neurons.png`
    *   显示仅在特定行为中效应大小超过阈值，而不存在于其他行为关键神经元列表中的特有神经元。
    *   选项 (`plot_unique_neurons_map` 函数):
        *   `show_title` (布尔值, 默认 `True`): 控制是否显示图表标题。
        *   `show_background_neurons` (布尔值, 通过 `config.py` 中的 `SHOW_BACKGROUND_NEURONS` 控制): 是否显示所有神经元作为背景。

4.  **新增组合图: 3x3 网格汇总图**
    *   文件名: `plot_all_behaviors_3x3_grid.png`
    *   将上述9种核心分析图表（3个单一行为，3个两两共享，3个单一特有）排列在一个3x3的网格中显示。
        *   **第一行**: 展示三个主要行为各自的关键神经元分布（例如：Close Key, Middle Key, Open Key）。
        *   **第二行**: 展示两两行为间的共享关键神经元（例如：Close-Middle Shared, Close-Open Shared, Middle-Open Shared），默认使用Scheme B。
        *   **第三行**: 展示三个主要行为各自的特有关键神经元（例如：Close Unique, Middle Unique, Open Unique）。
    *   每个子图都包含其原始的标题和图例（字体会略微缩小以适应组合布局）。
    *   组合图具有一个主标题，概括显示内容。
    *   此图提供了一个全面的、并列比较所有关键神经元分析结果的视图。
    *   *注意*: 独立生成的9个图表文件仍然会按原样保存。

## 未来可能的扩展

*   支持更多行为的分析。
*   实现三行为共享神经元的可视化。
 