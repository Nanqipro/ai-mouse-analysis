import pandas as pd
import numpy as np

# 创建测试数据
np.random.seed(42)
time_points = 1000
neurons = 5

# 生成模拟钙信号数据
data = {}
data['Time'] = np.arange(time_points)

for i in range(1, neurons + 1):
    # 基线信号
    baseline = np.random.normal(100, 5, time_points)
    
    # 添加一些钙瞬变
    signal = baseline.copy()
    for _ in range(np.random.randint(3, 8)):
        start = np.random.randint(0, time_points - 100)
        duration = np.random.randint(20, 80)
        amplitude = np.random.uniform(20, 50)
        
        # 创建钙瞬变形状
        x = np.arange(duration)
        transient = amplitude * np.exp(-x/20) * (1 - np.exp(-x/5))
        
        end = min(start + duration, time_points)
        signal[start:end] += transient[:end-start]
    
    data[f'Neuron_{i}'] = signal

# 创建DataFrame并保存
df = pd.DataFrame(data)

# 使用ExcelWriter保存到dF工作表
with pd.ExcelWriter('test_data.xlsx') as writer:
    df.to_excel(writer, sheet_name='dF', index=False)

print('测试数据文件已创建')