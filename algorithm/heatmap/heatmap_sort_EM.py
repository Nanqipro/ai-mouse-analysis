# 有放入CD1的数据进行热图绘制
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
from scipy.signal import find_peaks
from scipy import stats

# 自定义参数配置
# 可以根据需要修改默认值
class Config:
    # 输入文件路径
    INPUT_FILE = '../../datasets/EMtrace01.xlsx'
    # 输出文件名前缀
    OUTPUT_PREFIX = '../../graph/heatmap_sort-'
    # 时间戳区间默认值（None表示不限制）
    STAMP_MIN = 1  # 最小时间戳
    STAMP_MAX = 2999  # 最大时间戳
    # 排序方式：'peak'（默认，按峰值时间排序）、'calcium_wave'（按第一次真实钙波发生时间排序）或'custom'（按自定义顺序排序）
    SORT_METHOD = 'peak'
    # 自定义神经元排序顺序（仅在SORT_METHOD='custom'时使用）
    CUSTOM_NEURON_ORDER = ['n53', 'n40', 'n29', 'n34', 'n4', 'n32', 'n25', 'n27', 'n22', 'n55', 'n21', 'n5', 'n19']
    # 采样率 (Hz)
    SAMPLING_RATE = 4.8  # 采样频率，用于将时间戳转换为秒
    # 钙波检测参数
    CALCIUM_WAVE_THRESHOLD = 1.5  # 钙波阈值（标准差的倍数）
    MIN_PROMINENCE = 1.0  # 最小峰值突出度
    MIN_RISE_RATE = 0.1  # 最小上升速率
    MAX_FALL_RATE = 0.05  # 最大下降速率（下降应当比上升慢）

# 解析命令行参数（如果需要从命令行指定参数）
def parse_args():
    parser = argparse.ArgumentParser(description='神经元活动热图生成工具，支持自定义时间区间和排序方式')
    parser.add_argument('--input', type=str, help='输入数据文件路径')
    parser.add_argument('--output-prefix', type=str, help='输出文件名前缀')
    parser.add_argument('--stamp-min', type=float, help='最小时间戳值')
    parser.add_argument('--stamp-max', type=float, help='最大时间戳值')
    parser.add_argument('--sort-method', type=str, choices=['peak', 'calcium_wave', 'custom'], 
                        help='排序方式：peak（按峰值时间排序）、calcium_wave（按第一次真实钙波时间排序）或custom（按自定义顺序排序）')
    parser.add_argument('--ca-threshold', type=float, help='钙波检测阈值（标准差的倍数）')
    parser.add_argument('--min-prominence', type=float, help='最小峰值突出度')
    return parser.parse_args()

# 解析命令行参数并更新配置
args = parse_args()
if args.input:
    Config.INPUT_FILE = args.input
if args.output_prefix:
    Config.OUTPUT_PREFIX = args.output_prefix
if args.stamp_min is not None:
    Config.STAMP_MIN = args.stamp_min
if args.stamp_max is not None:
    Config.STAMP_MAX = args.stamp_max
if args.sort_method:
    Config.SORT_METHOD = args.sort_method
if args.ca_threshold is not None:
    Config.CALCIUM_WAVE_THRESHOLD = args.ca_threshold
if args.min_prominence is not None:
    Config.MIN_PROMINENCE = args.min_prominence

# 加载数据
print(f"正在从 {Config.INPUT_FILE} 加载数据...")
day6_data_full = pd.read_excel(Config.INPUT_FILE)

# 将 'stamp' 列设置为索引
stamp_column = day6_data_full['stamp'].copy()  # 保存原始时间戳
# 创建秒为单位的时间索引
seconds_index = stamp_column / Config.SAMPLING_RATE
day6_data_full = day6_data_full.set_index('stamp')

# 检查是否存在 'behavior' 列
has_behavior = 'behavior' in day6_data_full.columns

# 分离 'behavior' 列（如果存在）
if has_behavior:
    behavior_data_full = day6_data_full['behavior']
    neural_data_full = day6_data_full.drop(columns=['behavior'])
else:
    neural_data_full = day6_data_full.copy()

# 基于全局数据进行神经元排序（重要：确保不同时间区间的神经元排序一致）
print("正在基于全局数据计算神经元排序...")
neural_data_full_standardized = (neural_data_full - neural_data_full.mean()) / neural_data_full.std()

# 函数：按自定义神经元顺序排序
def sort_neurons_by_custom_order(data_columns, custom_order):
    """
    按自定义神经元顺序排序
    
    指定神经元按给定顺序排在前面，剩余神经元按字符串排序排在后面
    
    参数:
    data_columns -- 数据中的神经元列名
    custom_order -- 自定义的神经元顺序列表
    
    返回:
    sorted_neurons -- 按自定义顺序排列的神经元列表
    """
    available_neurons = set(data_columns)
    
    # 首先按照自定义顺序排列存在的神经元
    ordered_neurons = []
    for neuron in custom_order:
        if neuron in available_neurons:
            ordered_neurons.append(neuron)
    
    # 找出剩余的神经元，按字符串大小顺序排列
    remaining_neurons = sorted(list(available_neurons - set(ordered_neurons)))
    
    # 合并两部分：自定义顺序 + 剩余神经元（按大小排序）
    final_order = ordered_neurons + remaining_neurons
    
    return final_order

# 函数：检测神经元第一次真实钙波发生的时间点
def detect_first_calcium_wave(neuron_data):
    """
    检测神经元第一次真实钙波发生的时间点
    
    参数:
    neuron_data -- 包含神经元活动的时间序列数据（标准化后）
    
    返回:
    first_wave_time -- 第一次真实钙波发生的时间点，如果没有检测到则返回数据最后一个时间点
    """
    # 计算阈值（基于数据的标准差）
    threshold = Config.CALCIUM_WAVE_THRESHOLD
    
    # 使用find_peaks函数检测峰值
    peaks, properties = find_peaks(neuron_data, 
                                 height=threshold, 
                                 prominence=Config.MIN_PROMINENCE,
                                 distance=5)  # 要求峰值之间至少间隔5个时间点
    
    if len(peaks) == 0:
        # 如果没有检测到峰值，返回时间序列的最后一个点
        return neuron_data.index[-1]
    
    # 对每个峰值进行验证，确认是否为真实钙波（上升快，下降慢）
    for peak_idx in peaks:
        # 确保峰值不在时间序列的开始或结束处
        if peak_idx <= 1 or peak_idx >= len(neuron_data) - 2:
            continue
            
        # 计算峰值前的上升速率（取峰值前5个点或更少）
        pre_peak_idx = max(0, peak_idx - 5)
        rise_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[pre_peak_idx]) / (peak_idx - pre_peak_idx)
        
        # 计算峰值后的下降速率（取峰值后10个点或更少）
        post_peak_idx = min(len(neuron_data) - 1, peak_idx + 10)
        if post_peak_idx <= peak_idx:
            continue
        
        fall_rate = (neuron_data.iloc[peak_idx] - neuron_data.iloc[post_peak_idx]) / (post_peak_idx - peak_idx)
        
        # 确认是否符合钙波特征：上升快，下降慢
        if rise_rate > Config.MIN_RISE_RATE and 0 < fall_rate < Config.MAX_FALL_RATE:
            # 找到第一个真实钙波，返回时间点
            return neuron_data.index[peak_idx]
    
    # 如果没有满足条件的钙波，返回时间序列的最后一个点
    return neuron_data.index[-1]

# 根据排序方式选择相应的排序算法（基于全局数据）
if Config.SORT_METHOD == 'peak':
    # 原始方法：按峰值时间排序
    # 对于每个神经元，找到其信号达到最大值的时间戳（基于全局数据）
    peak_times = neural_data_full_standardized.idxmax()
    
    # 将神经元按照峰值时间从早到晚排序
    sorted_neurons = peak_times.sort_values().index
    
    sort_method_str = "Sorted by peak time (global)"
elif Config.SORT_METHOD == 'custom':
    # 自定义方法：按指定的神经元顺序排序
    sorted_neurons = sort_neurons_by_custom_order(neural_data_full_standardized.columns, Config.CUSTOM_NEURON_ORDER)
    
    sort_method_str = "Sorted by custom order"
    print(f"使用自定义神经元排序")
    print(f"指定顺序: {Config.CUSTOM_NEURON_ORDER}")
    print("剩余神经元将按字符串大小顺序排列在指定神经元下方")
else:  # 'calcium_wave'
    # 新方法：按第一次真实钙波发生时间排序（基于全局数据）
    first_wave_times = {}
    
    # 对每个神经元进行钙波检测
    for neuron in neural_data_full_standardized.columns:
        neuron_data = neural_data_full_standardized[neuron]
        first_wave_times[neuron] = detect_first_calcium_wave(neuron_data)
    
    # 转换为Series以便排序
    first_wave_times_series = pd.Series(first_wave_times)
    
    # 按第一次钙波时间排序
    sorted_neurons = first_wave_times_series.sort_values().index
    
    sort_method_str = "Sorted by first calcium wave time (global)"

print(f"神经元排序方式: {sort_method_str}")
print(f"神经元排序基于全局数据，确保不同时间区间的热图具有一致的纵坐标排序")

# 现在根据配置的时间戳区间筛选数据
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    # 确定实际的最小值和最大值
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else neural_data_full.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else neural_data_full.index.max()
    
    # 筛选神经元数据，保留指定区间内的数据
    neural_data_interval = neural_data_full.loc[min_stamp:max_stamp]
    
    # 如果有行为数据，也进行筛选
    if has_behavior:
        behavior_data_interval = behavior_data_full.loc[min_stamp:max_stamp]
        frame_lost = behavior_data_interval
    
    print(f"已筛选时间戳区间: {min_stamp} 到 {max_stamp}")
    # 对应的秒数区间
    min_seconds = min_stamp / Config.SAMPLING_RATE
    max_seconds = max_stamp / Config.SAMPLING_RATE
    print(f"对应的时间区间: {min_seconds:.2f}s 到 {max_seconds:.2f}s")
else:
    # 如果没有指定时间区间，使用全部数据
    neural_data_interval = neural_data_full
    if has_behavior:
        frame_lost = behavior_data_full

# 对筛选后的区间数据进行标准化
neural_data_interval_standardized = (neural_data_interval - neural_data_interval.mean()) / neural_data_interval.std()

# 根据全局排序后的神经元顺序重新排列筛选后数据的列
sorted_day6_data = neural_data_interval_standardized[sorted_neurons]

# **步骤4：找到所有行为标签的区间**

# 初始化行为区间变量
behavior_intervals = {}
unique_behaviors = []
global_unique_behaviors = []  # 全局行为类型

# 只有当behavior列存在时才处理行为标签
if has_behavior:
    # 获取全局数据中所有不同的行为标签（用于创建一致的图例）
    global_unique_behaviors = behavior_data_full.dropna().unique()
    print(f"全局行为类型: {list(global_unique_behaviors)}")
    
    # 获取当前时间区间内的行为标签
    unique_behaviors = frame_lost.dropna().unique()
    print(f"当前时间区间内的行为类型: {list(unique_behaviors)}")
    
    # 初始化当前时间区间内行为的区间字典
    for behavior in unique_behaviors:
        behavior_intervals[behavior] = []
    
    # 对frame_lost进行处理，找出每种行为的连续区间
    current_behavior = None
    start_time = None
    
    # 为了确保最后一个区间也被记录，将索引列表扩展一个元素
    extended_index = list(frame_lost.index) + [None]
    extended_values = list(frame_lost.values) + [None]
    
    for i, (timestamp, behavior) in enumerate(zip(extended_index, extended_values)):
        # 最后一个元素特殊处理
        if i == len(frame_lost):
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((start_time, extended_index[i-1]))
            break
        
        # 跳过空值
        if pd.isna(behavior):
            # 如果之前有行为，则结束当前区间
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((start_time, timestamp))
                start_time = None
                current_behavior = None
            continue
        
        # 如果是新的行为类型或第一个行为
        if behavior != current_behavior:
            # 如果之前有行为，先结束当前区间
            if start_time is not None and current_behavior is not None:
                behavior_intervals[current_behavior].append((start_time, timestamp))
            
            # 开始新的行为区间
            start_time = timestamp
            current_behavior = behavior

# **步骤5：绘制热图并标注所有行为区间**

# 设置绘图颜色范围
vmin, vmax = -2, 2  # 控制颜色对比度

# 创建图形和轴，使用更高的高度比例和精确调整来确保对齐
fig = plt.figure(figsize=(60, 25))

# 使用更精确的GridSpec布局系统
# 修改为2行2列布局：左侧为行为线条和热图，右侧为图例
from matplotlib.gridspec import GridSpec
grid = GridSpec(2, 2, height_ratios=[0.5, 6], width_ratios=[6, 0.5], hspace=0.05, wspace=0.02, figure=fig)

# 先创建热图子图（左下角）
ax_heatmap = fig.add_subplot(grid[1, 0])

# 为了确保对齐，我们首先定义X轴范围
# 创建一个X轴数据点数组，这个数组将用于两个图表
data_x_points = np.arange(len(sorted_day6_data.index))

# 绘制热图
heatmap = sns.heatmap(sorted_day6_data.T, cmap='viridis', cbar=False, vmin=vmin, vmax=vmax, ax=ax_heatmap)

# 置空刻度位置，稍后设置
# 设置热图的x轴范围为精确数据点范围
ax_heatmap.set_xlim(-0.5, len(sorted_day6_data.index) - 0.5)

# 创建行为标记子图（左上角）
ax_behavior = fig.add_subplot(grid[0, 0])

# 手动设置行为子图的X轴范围与热图完全相匹配
ax_behavior.set_xlim(-0.5, len(sorted_day6_data.index) - 0.5)

# 创建图例子图（右下角，与热图对齐）
ax_legend = fig.add_subplot(grid[1, 1])

# 只有当behavior列存在时才添加行为标记
if has_behavior and len(global_unique_behaviors) > 0:
    # 创建固定的行为颜色映射，确保相同行为始终使用相同颜色
    # 预定义所有可能的行为及其固定颜色，使用更加鲜明和对比度更高的颜色
    fixed_color_map = {
        # === 原有配色（保持不变）===
        'Crack-seeds-shells': '#FF9500',    # 明亮橙色
        'Eat-feed': '#0066CC',              # 深蓝色
        'Eat-seed-kernels': '#00CC00',      # 亮绿色
        'Explore': '#FF0000',               # 鲜红色
        'Explore-search-seeds': '#9900FF',  # 亮紫色
        'Find-seeds': '#994C00',            # 深棕色
        'Get-feed': '#FF00CC',              # 亮粉色
        'Get-seeds': '#000000',             # 黑色
        'Grab-seeds': '#AACC00',            # 亮黄绿色
        'Groom': '#00CCFF',                 # 亮蓝绿色 - 原有的整理行为
        'Smell-feed': '#66B3FF',            # 亮蓝色
        'Smell-Get-seeds': '#33FF33',       # 鲜绿色
        'Store-seeds': '#FF6666',           # 亮红色
        'Water': '#CC99FF',                 # 亮紫色
        
        
        # === 高架十字迷宫行为标签 ===
        'Close-arm': '#8B0000',             # 深红色 - 封闭臂
        'Close-armed-Exp': '#CD5C5C',       # 印度红 - 封闭臂探索
        'Closed-arm-freezing': '#2F2F2F',  # 深灰色 - 封闭臂僵直
        'Middle-zone': '#FFD700',           # 金色 - 中央区域
        'Middle-zone-freezing': '#B8860B',  # 深金黄色 - 中央区域僵直
        'Open-arm': '#32CD32',              # 酸橙绿 - 开放臂
        'Open-arm-exp': '#00FF7F',          # 春绿色 - 开放臂探索
        'open-arm-freezing': '#006400',     # 深绿色 - 开放臂僵直
        'Open-arm-head dipping': '#7CFC00', # 草绿色 - 开放臂头部探测
        
        # === 新增家庭笼行为标签 ===
        'Active': '#FF4500',                # 橙红色 - 活跃状态
        'Drink': '#1E90FF',                 # 道奇蓝 - 饮水
        'Move': '#32CD32',                  # 酸橙绿 - 移动
        'Scratch': '#CD853F',               # 秘鲁色 - 抓挠
        'Scratch + Groom': '#DDA0DD',       # 李子色 - 抓挠+整理组合行为
        'Sleep': '#2F4F4F',                 # 深灰绿色 - 睡眠
        'Wake': '#FFD700',                  # 金色 - 清醒
        'zone-out': '#708090'               # 石板灰 - 发呆状态
    }
    
    # 为全局行为类型创建颜色映射（确保图例一致性）
    global_color_map = {}
    for i, behavior in enumerate(global_unique_behaviors):
        if behavior in fixed_color_map:
            # 使用预定义的颜色
            global_color_map[behavior] = fixed_color_map[behavior]
        else:
            # 对于未预定义的行为，使用更鲜明的调色板
            # 从tab10调色板获取颜色，它提供更高的对比度
            colors = plt.cm.tab10(np.linspace(0, 1, len(global_unique_behaviors)))
            global_color_map[behavior] = colors[i % 10]
    
    # 为当前时间区间内的行为创建颜色映射（用于绘制）
    color_map = {}
    for behavior in unique_behaviors:
        if behavior in global_color_map:
            color_map[behavior] = global_color_map[behavior]
    
    # 创建全局图例补丁列表（基于全局行为类型，确保图例一致性）
    legend_patches = []
    for behavior in sorted(global_unique_behaviors):  # 按字母顺序排序确保一致性
        behavior_color = global_color_map[behavior]
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=behavior_color, alpha=0.9, label=behavior))
    
    # 将所有行为绘制在同一条水平线上（只绘制当前时间区间内存在的行为）
    y_position = 0.5  # 固定的Y轴位置，居中
    line_height = 0.8  # 线条的高度
    
    # 为当前时间区间内每种行为绘制区间，都在同一水平线上
    for behavior, intervals in behavior_intervals.items():
        behavior_color = color_map[behavior]
        
        for start_time, end_time in intervals:
            # 检查时间点是否在排序后的数据索引中
            if start_time in sorted_day6_data.index and end_time in sorted_day6_data.index:
                # 获取对应的绘图位置
                start_idx = sorted_day6_data.index.get_loc(start_time)  # 获取索引位置
                end_idx = sorted_day6_data.index.get_loc(end_time)
                
                # 在使用前转换为实际坐标位置 (对应网格)
                start_pos = start_idx # 网格的中心是整数位置
                end_pos = end_idx
                
                # 如果区间有宽度
                if end_pos - start_pos > 0:  
                    # 在行为标记子图中绘制区间，使用精确的数据点对齐方式
                    # 提高alpha值从0.7到0.9，使颜色更加明显
                    rect = plt.Rectangle((start_pos - 0.5, y_position - line_height/2), 
                                        end_pos - start_pos, line_height, 
                                        color=behavior_color, alpha=0.9, 
                                        ec='black', linewidth=0.5)  # 添加黑色边框以增强可见度
                    ax_behavior.add_patch(rect)
                    
                    # 在热图中添加区间边界垂直线
                    # 使用与热图网格线一致的位置，确保对齐
                    ax_heatmap.axvline(x=start_pos - 0.5, color='white', linestyle='--', linewidth=2, alpha=0.5)
                    ax_heatmap.axvline(x=end_pos - 0.5, color='white', linestyle='--', linewidth=2, alpha=0.5)
    
    # 再次确认两个坐标轴的对齐情况
    # 唯一正确的做法是设置完全相同的范围
    ax_heatmap.set_xlim(-0.5, len(sorted_day6_data.index) - 0.5)
    ax_behavior.set_xlim(-0.5, len(sorted_day6_data.index) - 0.5)
    
    # 设置行为子图的Y轴范围，只显示一条线
    ax_behavior.set_ylim(0, 1)
    
    # 移除Y轴刻度和标签
    ax_behavior.set_yticks([])
    ax_behavior.set_yticklabels([])
    
    # 特别重要：移除X轴刻度，让它只在热图上显示
    ax_behavior.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax_behavior.set_title('Behavior Timeline', fontsize=40, pad=10)
    ax_behavior.set_xlabel('')  # Remove x-axis label, shared with the heatmap below

    # 更强制地隐藏X轴刻度和标签
    ax_behavior.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # 确保行为图和热图水平对齐
    ax_behavior.set_anchor('SW')
    # 去除行为子图边框
    ax_behavior.spines['top'].set_visible(False)
    ax_behavior.spines['right'].set_visible(False)
    ax_behavior.spines['bottom'].set_visible(False)
    ax_behavior.spines['left'].set_visible(False)
    
    # 在单独的图例子图中添加图例（右侧）
    ax_legend.axis('off')  # 隐藏图例子图的坐标轴
    
    # 计算图例的行数，垂直排列所有行为类型
    num_behaviors = len(legend_patches)
    
    # 计算合适的字体大小，使图例高度与热图一致
    # 根据行为数量动态调整字体大小
    # if num_behaviors <= 5:
    #     legend_fontsize = 20
    #     title_fontsize = 22
    # elif num_behaviors <= 8:
    #     legend_fontsize = 16
    #     title_fontsize = 18
    # elif num_behaviors <= 12:
    #     legend_fontsize = 14
    #     title_fontsize = 16
    # else:
    #     legend_fontsize = 12
    #     title_fontsize = 14

    legend_fontsize = 40  # 设置您想要的字体大小
    title_fontsize = 40   # 设置标题字体大小
    
    legend = ax_legend.legend(handles=legend_patches, loc='center left', fontsize=legend_fontsize, 
                           title='Behavior Types', title_fontsize=title_fontsize, ncol=1,
                           frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0, 0.5))

# 在第426帧处添加白色虚线
# 检查数据中是否有足够的时间戳
if len(sorted_day6_data.index) > 426:
    # 绘制垂直线，白色虚线
    ax_heatmap.axvline(x=426 - 0.5, color='white', linestyle='--', linewidth=4)

# 移除热图标题，直接设置轴标签
ax_heatmap.set_xlabel('Time (s)', fontsize=40)      # 增大X轴标签字体
ax_heatmap.set_ylabel('neuron', fontsize=40)        # 增大Y轴标签字体

# 修改Y轴标签（神经元标签）的字体大小和粗细，设置为水平方向
ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), fontsize=23, fontweight='bold', rotation=0)

# 修改X轴标签（时间戳）的字体大小和粗细
ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), fontsize=30, fontweight='bold')

# 设置X轴刻度，以10秒为间隔，仿照show_trace.py的方法
# 使用配置中的采样率
time_per_frame = 1.0 / Config.SAMPLING_RATE  # 每帧的时间间隔（秒）

# 获取当前筛选后数据的时间范围
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    # 如果指定了时间区间，使用筛选后的范围
    min_stamp = sorted_day6_data.index.min()
    max_stamp = sorted_day6_data.index.max()
    min_seconds = min_stamp / Config.SAMPLING_RATE
    max_seconds = max_stamp / Config.SAMPLING_RATE
    # 确保起始时间不小于0
    min_seconds = max(0, min_seconds)
else:
    # 如果没有指定时间范围，从0开始显示
    min_stamp = sorted_day6_data.index.min()
    max_stamp = sorted_day6_data.index.max()
    min_seconds = max(0, min_stamp / Config.SAMPLING_RATE)
    max_seconds = max_stamp / Config.SAMPLING_RATE

# 生成以10秒为间隔的时间刻度（秒）
import matplotlib.ticker as ticker
# 计算合适的刻度起始点（向下取整到最近的10秒）
start_tick = int(min_seconds // 10) * 10
time_ticks_seconds = np.arange(start_tick, max_seconds + 10, 10)

# 将秒转换为对应的数据点位置（相对于筛选后的数据）
xtick_positions = []
xtick_labels = []

for t in time_ticks_seconds:
    # 将秒转换为时间戳
    timestamp = t * Config.SAMPLING_RATE
    # 检查这个时间戳是否在筛选后的数据范围内
    if min_stamp <= timestamp <= max_stamp and timestamp in sorted_day6_data.index:
        # 获取在筛选后数据中的索引位置
        pos = sorted_day6_data.index.get_loc(timestamp)
        xtick_positions.append(pos)
        xtick_labels.append(f'{int(t)}')

# 如果没有找到合适的刻度位置，使用默认方法
if len(xtick_positions) == 0:
    numpoints = sorted_day6_data.shape[0]
    # 生成均匀分布的刻度
    num_ticks = min(10, numpoints // 10 + 1)  # 最多10个刻度
    xtick_positions = np.linspace(0, numpoints - 1, num_ticks, dtype=int)
    # 计算对应的时间标签
    xtick_labels = []
    for pos in xtick_positions:
        timestamp = sorted_day6_data.index[pos]
        seconds = timestamp / Config.SAMPLING_RATE
        xtick_labels.append(f'{int(seconds)}')

# 设置X轴刻度位置和标签
ax_heatmap.set_xticks(xtick_positions)
ax_heatmap.set_xticklabels(xtick_labels, fontsize=30, fontweight='bold', rotation=45)

# 应用紧凑布局
# 不使用tight_layout()，因为它与GridSpec布局不兼容
# 而是使用之前设置的subplots_adjust()已经足够调整布局

# 从输入文件路径中提取文件名（不包括路径和扩展名），仿照show_trace.py的方法
import os
input_filename = os.path.basename(Config.INPUT_FILE)
input_filename = os.path.splitext(input_filename)[0]  # 去除扩展名

# 构建输出文件名：目录 + 前缀 + 排序方式 + 输入文件名 + 时间戳信息
stamp_info = ''
if Config.STAMP_MIN is not None or Config.STAMP_MAX is not None:
    min_stamp = Config.STAMP_MIN if Config.STAMP_MIN is not None else sorted_day6_data.index.min()
    max_stamp = Config.STAMP_MAX if Config.STAMP_MAX is not None else sorted_day6_data.index.max()
    min_seconds = min_stamp / Config.SAMPLING_RATE
    max_seconds = max_stamp / Config.SAMPLING_RATE
    stamp_info = f'_{min_seconds:.2f}s_{max_seconds:.2f}s'

output_filename = f"{Config.OUTPUT_PREFIX}{input_filename}_{Config.SORT_METHOD}{stamp_info}.png"

# 在保存前使用多重方法确保对齐

# 1. 强制更新布局
fig.canvas.draw()

# 2. 再次确认轴范围一致
ax_heatmap.set_xlim(-0.5, len(sorted_day6_data.index) - 0.5)
ax_behavior.set_xlim(-0.5, len(sorted_day6_data.index) - 0.5)

# 3. 达到更精确的对齐
# 获取热图的实际边界位置（像素单位）
heatmap_bbox = ax_heatmap.get_position()
behavior_bbox = ax_behavior.get_position()

# 不能直接修改Bbox对象，需要创建新的
# 使用Bbox的坐标创建新的位置，保持高度不变，但使用热图的宽度和水平位置
from matplotlib.transforms import Bbox
new_behavior_pos = Bbox([[heatmap_bbox.x0, behavior_bbox.y0], 
                        [heatmap_bbox.x0 + heatmap_bbox.width, behavior_bbox.y0 + behavior_bbox.height]])

# 设置新的位置
ax_behavior.set_position(new_behavior_pos)

# 4. 保存图像
print(f"正在保存图像到 {output_filename}")
fig.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, dpi=100)
plt.close()

# 输出保存信息
print(f"热图已保存至: {output_filename}")
print("程序执行完成")