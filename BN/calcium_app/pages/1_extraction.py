import streamlit as st
import pandas as pd
import os
import datetime
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.extraction_logic import run_batch_extraction, extract_calcium_features

st.set_page_config(page_title="事件提取", layout="wide")
st.title("🧪 钙事件提取")
st.markdown("在此页面，您可以：\n1. **参数调试**: 上传一个文件，通过可视化预览找到最佳参数。\n2. **批量提取**: 使用找到的参数处理所有上传的文件。")

with st.sidebar:
    st.header("🔬 分析参数")
    fs = st.number_input("采样频率 (Hz)", 0.1, 100.0, 4.8, 0.1, help="默认为 4.8Hz。")
    min_duration_frames = st.slider("最小持续时间 (帧)", 1, 100, 12, 1)
    max_duration_frames = st.slider("最大持续时间 (帧)", 50, 2000, 800, 10)
    min_snr = st.slider("最小信噪比", 1.0, 10.0, 3.5, 0.1)
    smooth_window = st.slider("平滑窗口 (帧, 奇数)", 3, 101, 31, 2)
    peak_distance_frames = st.slider("峰值最小距离 (帧)", 1, 100, 24, 1)
    filter_strength = st.slider("过滤强度", 0.5, 2.0, 1.0, 0.1)

params = {
    'min_duration': min_duration_frames, 'max_duration': max_duration_frames,
    'min_snr': min_snr, 'smooth_window': smooth_window,
    'peak_distance': peak_distance_frames, 'filter_strength': filter_strength
}

uploaded_files = st.file_uploader("上传 Excel 文件 (需含 'dF' 工作表)", type=['xlsx', 'xls'], accept_multiple_files=True)

if uploaded_files:
    st.markdown("---")
    st.header("🔬 参数调试与单神经元可视化")
    try:
        first_file = uploaded_files[0]
        @st.cache_data
        def load_data(file):
            return pd.read_excel(file, sheet_name='dF', header=0)
        df_visualize = load_data(first_file)
        
        neuron_cols = df_visualize.columns[1:]
        selected_neuron = st.selectbox("选择一个神经元进行预览:", options=neuron_cols)

        if st.button("📊 生成预览图"):
            if selected_neuron:
                with st.spinner(f"正在分析神经元 {selected_neuron}..."):
                    feature_table, fig, _ = extract_calcium_features(
                        df_visualize[selected_neuron].values, fs=fs, visualize=True, params=params
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                    if not feature_table.empty:
                        st.dataframe(feature_table)
                    else:
                        st.warning("未检测到有效事件。")
    except Exception as e:
        st.error(f"处理预览时出错: {e}")

    st.markdown("---")
    st.header("🚀 批量提取事件")
    if st.button("开始批量处理所有上传的文件"):
        with st.spinner('正在进行批量分析...'):
            # Setup directories
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            upload_dir = os.path.join("calcium_app", "uploads", timestamp)
            results_dir = os.path.join("calcium_app", "results")
            os.makedirs(upload_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            
            # Save files and run extraction
            saved_file_paths = [os.path.join(upload_dir, f.name) for f in uploaded_files]
            for i, file in enumerate(uploaded_files):
                with open(saved_file_paths[i], "wb") as f:
                    f.write(file.getbuffer())
            
            result_path = run_batch_extraction(saved_file_paths, results_dir, fs=fs, **params)
            
            if result_path and os.path.exists(result_path):
                st.success("🎉 分析完成！")
                st.session_state['extraction_result_path'] = result_path
                with open(result_path, "rb") as f:
                    st.download_button("📥 下载结果文件", f, file_name=os.path.basename(result_path))
            else:
                st.error("批量分析未生成任何结果。")

if 'extraction_result_path' in st.session_state:
    st.success("结果已生成，可前往下一步进行聚类分析。")