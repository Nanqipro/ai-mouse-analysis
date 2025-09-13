import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, peak_widths
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy import signal
import os
import json
import base64
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional

def detect_calcium_transients(data, fs=4.8, params=None):
    """
    Detects calcium transients in calcium imaging data.
    All parameters are passed via the 'params' dictionary.
    This version uses derivative for start point and exponential fit for end point.
    """
    default_params = {
        'min_snr': 3.5, 'min_duration': 12, 'smooth_window': 31, 'peak_distance': 24,
        'baseline_percentile': 8, 'max_duration': 800, 'detect_subpeaks': False,
        'subpeak_prominence': 0.15, 'subpeak_width': 5, 'subpeak_distance': 8,
        'min_morphology_score': 0.20, 'min_exp_decay_score': 0.12, 'filter_strength': 1.0,
        'start_deriv_threshold_sd': 4.0, 'end_exp_fit_factor_m': 3.5
    }
    if params is None:
        params = {}
    
    run_params = default_params.copy()
    run_params.update(params)

    # Use local variables for easier access
    p = run_params
    min_snr, min_duration, smooth_window, peak_distance = p['min_snr'], p['min_duration'], p['smooth_window'], p['peak_distance']
    baseline_percentile, max_duration, filter_strength = p['baseline_percentile'], p['max_duration'], p['filter_strength']
    start_deriv_threshold_sd, end_exp_fit_factor_m = p['start_deriv_threshold_sd'], p['end_exp_fit_factor_m']

    if smooth_window > 1 and smooth_window % 2 == 0:
        smooth_window += 1
    smoothed_data = signal.savgol_filter(data, smooth_window, 3) if smooth_window > 1 else data.copy()
    
    baseline = np.percentile(smoothed_data, baseline_percentile)
    noise_level = np.std(smoothed_data[smoothed_data < np.percentile(smoothed_data, 50)])
    if noise_level == 0:
        noise_level = 1e-9

    # --- Start point detection setup ---
    dF_dt = np.gradient(smoothed_data)
    baseline_deriv_mask = smoothed_data < np.percentile(smoothed_data, 50)
    baseline_derivative = dF_dt[baseline_deriv_mask]
    deriv_mean = np.mean(baseline_derivative)
    deriv_std = np.std(baseline_derivative)
    deriv_threshold = deriv_mean + start_deriv_threshold_sd * deriv_std

    # --- Peak detection ---
    threshold = baseline + min_snr * noise_level
    prominence_threshold = noise_level * 1.2 * filter_strength
    min_width_frames = min_duration // 6
    peaks, peak_props = find_peaks(smoothed_data, height=threshold, prominence=prominence_threshold, width=min_width_frames, distance=peak_distance)
    
    if len(peaks) == 0:
        return [], smoothed_data

    # --- End point detection setup ---
    def exp_decay(t, A, tau, C):
        return A * np.exp(-t / tau) + C
        
    transients = []
    for i, peak_idx in enumerate(peaks):
        # --- New start point detection ---
        start_idx = peak_idx
        left_limit = 0 if i == 0 else peaks[i-1]
        while start_idx > left_limit:
            if dF_dt[start_idx] < deriv_threshold:
                break
            start_idx -= 1
        
        # --- New end point detection ---
        prelim_end_idx = peak_idx
        right_limit = len(smoothed_data) - 1 if i == len(peaks) - 1 else peaks[i+1]
        while prelim_end_idx < right_limit and smoothed_data[prelim_end_idx] > baseline:
            prelim_end_idx += 1
        
        end_idx = prelim_end_idx
        decay_data = smoothed_data[peak_idx:prelim_end_idx]
        if len(decay_data) > 3:
            t_decay = np.arange(len(decay_data))
            try:
                initial_A = smoothed_data[peak_idx] - baseline
                initial_tau = max(1.0, len(decay_data) / 2)
                initial_C = baseline
                popt, _ = curve_fit(
                    exp_decay, t_decay, decay_data, 
                    p0=(initial_A, initial_tau, initial_C),
                    maxfev=5000,
                    bounds=([0, 1e-9, -np.inf], [np.inf, np.inf, np.inf])
                )
                A_fit, tau_fit, C_fit = popt
                if 0 < tau_fit < len(data):
                    calculated_end_idx = peak_idx + int(end_exp_fit_factor_m * tau_fit)
                    end_idx = min(calculated_end_idx, right_limit, len(smoothed_data) - 1)
            except (RuntimeError, ValueError):
                pass
        
        # Final checks on boundaries
        if i < len(peaks) - 1 and end_idx >= peaks[i+1]:
            end_idx = peak_idx + np.argmin(smoothed_data[peak_idx:peaks[i+1]])
        if i > 0 and start_idx <= peaks[i-1]:
            start_idx = peaks[i-1] + np.argmin(smoothed_data[peaks[i-1]:peak_idx])

        duration_frames = end_idx - start_idx
        if not (min_duration <= duration_frames <= max_duration):
            continue
            
        # Feature calculation
        amplitude = smoothed_data[peak_idx] - baseline
        widths_info = peak_widths(smoothed_data, [peak_idx], rel_height=0.5)

        transients.append({
            'start': int(start_idx), 
            'peak': int(peak_idx), 
            'end': int(end_idx), 
            'amplitude': float(amplitude),
            'duration': float(duration_frames / fs),
            'fwhm': float(widths_info[0][0] / fs) if len(widths_info[0]) > 0 else np.nan,
            'rise_time': float((peak_idx - start_idx) / fs),
            'decay_time': float((end_idx - peak_idx) / fs),
            'auc': float(trapezoid(smoothed_data[start_idx:end_idx] - baseline, dx=1/fs)),
            'snr': float(amplitude / noise_level)
        })
        
    return transients, smoothed_data

def extract_calcium_features(neuron_data, fs=4.8, visualize=False, params=None):
    if params is None:
        params = {}
    transients, smoothed_data = detect_calcium_transients(neuron_data, fs=fs, params=params)
    if not transients:
        return pd.DataFrame(), None, smoothed_data
    feature_table = pd.DataFrame(transients)
    if visualize:
        fig = visualize_calcium_transients(neuron_data, smoothed_data, transients, fs=fs)
        return feature_table, fig, smoothed_data
    return feature_table, None, smoothed_data

def visualize_calcium_transients(raw_data, smoothed_data, transients, fs=4.8):
    fig, ax = plt.subplots(figsize=(20, 6))
    time_axis = np.arange(len(raw_data)) / fs
    ax.plot(time_axis, raw_data, color='grey', alpha=0.6, label='Raw Signal')
    ax.plot(time_axis, smoothed_data, color='blue', label='Smoothed Signal')
    for transient in transients:
        start_time, peak_time, end_time = transient['start']/fs, transient['peak']/fs, transient['end']/fs
        ax.axvspan(start_time, end_time, color='yellow', alpha=0.3)
        ax.plot(peak_time, smoothed_data[transient['peak']], 'rv')
    ax.set_title("Detected Calcium Transients")
    ax.set_xlabel(f"Time (seconds, fs={fs}Hz)")
    ax.set_ylabel("Fluorescence (a.u.)")
    ax.legend()
    return fig

def analyze_all_neurons_transients(data_df, neuron_columns, fs=4.8, start_id=1, file_info=None, params=None):
    all_transients = []
    for neuron in neuron_columns:
        feature_table, _, _ = extract_calcium_features(data_df[neuron].values, fs=fs, params=params)
        if not feature_table.empty:
            feature_table['neuron_id'] = neuron
            feature_table['event_id'] = range(start_id, start_id + len(feature_table))
            start_id += len(feature_table)
            all_transients.append(feature_table)
    if not all_transients:
        return pd.DataFrame(), start_id
    final_table = pd.concat(all_transients, ignore_index=True)
    if file_info:
        for key, value in file_info.items():
            final_table[key] = value
    return final_table, start_id

def get_interactive_data(file_path: str, neuron_id: str) -> Dict[str, Any]:
    """
    获取用于交互式图表的原始数据
    
    Args:
        file_path: 文件路径
        neuron_id: 神经元ID
    
    Returns:
        包含时间和数据的字典
    """
    try:
        # 读取数据
        data = pd.read_excel(file_path)
        
        if neuron_id not in data.columns:
            raise ValueError(f"神经元 {neuron_id} 不存在")
        
        # 获取神经元数据
        neuron_data = data[neuron_id].values
        
        # 生成时间轴（假设采样频率为4.8Hz）
        time_points = np.arange(len(neuron_data)) / 4.8
        
        return {
            'time': time_points.tolist(),
            'data': neuron_data.tolist(),
            'neuron_id': neuron_id,
            'total_points': len(neuron_data)
        }
        
    except Exception as e:
        raise Exception(f"获取交互式数据失败: {str(e)}")

def extract_manual_range(file_path: str, neuron_id: str, start_time: float, end_time: float, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    基于用户选择的时间范围进行钙事件提取
    
    Args:
        file_path: 文件路径
        neuron_id: 神经元ID
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        params: 参数字典
    
    Returns:
        提取结果
    """
    try:
        # 读取数据
        data = pd.read_excel(file_path)
        
        if neuron_id not in data.columns:
            raise ValueError(f"神经元 {neuron_id} 不存在")
        
        # 获取神经元数据
        neuron_data = data[neuron_id].values
        fs = params.get('fs', 4.8)
        
        # 转换时间到索引
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        
        # 确保索引在有效范围内
        start_idx = max(0, start_idx)
        end_idx = min(len(neuron_data), end_idx)
        
        if start_idx >= end_idx:
            raise ValueError("无效的时间范围")
        
        # 提取选定范围的数据
        selected_data = neuron_data[start_idx:end_idx]
        
        # 检测钙瞬变
        transients, smoothed_data = detect_calcium_transients(selected_data, fs, params)
        
        # 调整瞬变的时间索引（相对于原始数据）
        adjusted_transients = []
        for transient in transients:
            adjusted_transient = {}
            # 确保所有值都是Python原生类型，避免JSON序列化错误
            adjusted_transient['start'] = int(transient['start'] + start_idx)
            adjusted_transient['peak'] = int(transient['peak'] + start_idx)
            adjusted_transient['end'] = int(transient['end'] + start_idx)
            adjusted_transient['amplitude'] = float(transient['amplitude'])
            adjusted_transient['duration'] = float(transient['duration'])
            adjusted_transient['fwhm'] = float(transient['fwhm']) if not np.isnan(transient['fwhm']) else None
            adjusted_transient['rise_time'] = float(transient['rise_time'])
            adjusted_transient['decay_time'] = float(transient['decay_time'])
            adjusted_transient['auc'] = float(transient['auc'])
            adjusted_transient['snr'] = float(transient['snr'])
            adjusted_transient['start_time'] = float(adjusted_transient['start'] / fs)
            adjusted_transient['peak_time'] = float(adjusted_transient['peak'] / fs)
            adjusted_transient['end_time'] = float(adjusted_transient['end'] / fs)
            adjusted_transients.append(adjusted_transient)
        
        # 生成可视化
        fig = visualize_calcium_transients(
            selected_data, smoothed_data, transients, fs
        )
        
        # 转换图表为base64
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            'success': True,
            'neuron_id': neuron_id,
            'time_range': {'start': start_time, 'end': end_time},
            'transients_count': len(adjusted_transients),
            'transients': adjusted_transients,
            'features': adjusted_transients,  # 添加features字段以兼容前端显示
            'plot': plot_base64
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def run_batch_extraction(file_paths, output_dir, fs=4.8, **kwargs):
    all_results = []
    current_event_id = 1
    for file_path in file_paths:
        try:
            data_df = pd.read_excel(file_path, sheet_name='dF', header=0)
            neuron_columns = data_df.columns[1:]
            file_info = {'source_file': os.path.basename(file_path)}
            result_df, next_start_id = analyze_all_neurons_transients(
                data_df=data_df, neuron_columns=neuron_columns, fs=fs,
                start_id=current_event_id, file_info=file_info, params=kwargs
            )
            if not result_df.empty:
                all_results.append(result_df)
                current_event_id = next_start_id
        except Exception as e:
            print(f"Error processing file {os.path.basename(file_path)}: {e}")
            continue
    if not all_results:
        return None
    final_df = pd.concat(all_results, ignore_index=True)
    output_filename = f"batch_run_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}_features.xlsx"
    output_path = os.path.join(output_dir, output_filename)
    final_df.to_excel(output_path, index=False)
    return output_path