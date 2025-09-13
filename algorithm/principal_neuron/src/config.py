# EMtrace01 分析的配置文件

# 基于效应大小识别关键神经元的阈值
EFFECT_SIZE_THRESHOLD = 0.5

# 不同行为的颜色配置
BEHAVIOR_COLORS = {
    'Close-Arm': 'red',    # "靠近臂"行为
    'Middle-Zone': 'green', # "中间区域"行为
    'Open-Arm': 'blue',     # "开放臂"行为
}

# 行为对共享神经元的混合颜色配置 (键: 按字母顺序排序的行为名称元组)
MIXED_BEHAVIOR_COLORS = {
    ('Close-Arm', 'Middle-Zone'): 'yellow',  # Close-Arm & Middle-Zone 的共享神经元颜色
    ('Close-Arm', 'Open-Arm'): 'magenta', # Close-Arm & Open-Arm 的共享神经元颜色
    ('Middle-Zone', 'Open-Arm'): 'cyan',    # Middle-Zone & Open-Arm 的共享神经元颜色
    # ('Close-Arm', 'Middle-Zone', 'Open-Arm'): 'lightgray' # 未来若需分析三者共享，可启用此颜色
}

# 目标神经元数量 (供参考，在阈值建议功能中使用过)
TARGET_MIN_NEURONS = 5
TARGET_MAX_NEURONS = 10 

# --- Key Neuron Plotting Options ---
STANDARD_KEY_NEURON_ALPHA = 0.7 # Standard alpha for key neurons when not specifically faded

# --- Background Neuron Plotting Options ---
# Determines if all neurons (not just key ones) should be plotted in the background
SHOW_BACKGROUND_NEURONS = True  # Set to False to disable plotting all neurons in background
BACKGROUND_NEURON_COLOR = 'lightgray'  # Color for background neurons
BACKGROUND_NEURON_SIZE = 75            # Marker size for background neurons
BACKGROUND_NEURON_ALPHA = 0.3          # Transparency for background neurons 

# --- Shared Neuron Plot (Scheme B) Options ---
# Determines if non-shared key neurons in Scheme B plots use standard alpha (not faded)
USE_STANDARD_ALPHA_FOR_UNSHARED_IN_SCHEME_B = True # Set to False to use a more faded alpha (alpha_non_shared parameter in plotting_utils)