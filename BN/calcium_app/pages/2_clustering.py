import streamlit as st
import os
import glob
import pandas as pd
from datetime import datetime

# 模块导入路径设置
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clustering_logic import (
    load_data,
    enhance_preprocess_data,
    cluster_kmeans,
    visualize_clusters_2d,
    visualize_feature_distribution,
    analyze_clusters,
    add_cluster_to_excel
)

st.set_page_config(page_title="聚类分析", page_icon="🧩", layout="wide")

st.title("🧩 聚类分析")
st.markdown("---")
st.markdown("""
在此页面，您可以对提取出的钙事件特征进行聚类分析。
1.  系统会自动从`results`文件夹加载特征文件 (`*_features.xlsx`)。
2.  请选择希望将事件划分成的簇数（K值）。
3.  点击"开始聚类分析"按钮，查看结果。
""")

# --- 状态初始化 ---
if 'cluster_results' not in st.session_state:
    st.session_state.cluster_results = None
if 'selected_feature_file' not in st.session_state:
    st.session_state.selected_feature_file = None

# --- 1. 文件选择 ---
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

feature_files = glob.glob(os.path.join(results_dir, '*_features.xlsx'))

if not feature_files:
    st.warning('⚠️ 在 "results" 文件夹中未找到特征文件 (*_features.xlsx)。请先在"事件提取"页面生成特征文件。')
    st.stop()

# 解析文件名以获取更友好的显示名称
file_options = {}
for f in feature_files:
    basename = os.path.basename(f)
    try:
        # 尝试从文件名解析时间戳
        timestamp_str = basename.split('_features.xlsx')[0].split('_')[-1]
        dt_obj = datetime.strptime(timestamp_str, '%Y%m%d-%H%M%S')
        friendly_name = f"{basename} (创建于: {dt_obj.strftime('%Y-%m-%d %H:%M:%S')})"
        file_options[friendly_name] = f
    except (ValueError, IndexError):
        # 如果格式不匹配，则直接使用文件名
        file_options[basename] = f

selected_friendly_name = st.selectbox(
    "请选择要进行聚类分析的特征文件:",
    options=list(file_options.keys()),
    index=0
)
st.session_state.selected_feature_file = file_options[selected_friendly_name]
st.info(f"已选择文件: `{os.path.basename(st.session_state.selected_feature_file)}`")


# --- 2. 参数设置 ---
st.markdown("聚类参数设置")
col1, col2 = st.columns(2)
with col1:
    k_value = st.slider("选择聚类数 (K)", min_value=2, max_value=15, value=3, step=1)
with col2:
    dim_reduction_method = st.selectbox("选择降维可视化方法", options=['PCA', 't-SNE'], index=0)


# --- 3. 执行与显示 ---
if st.button("🚀 开始聚类分析", type="primary"):
    if st.session_state.selected_feature_file:
        with st.spinner("正在执行聚类分析，请稍候..."):
            try:
                # 加载数据
                df = load_data(st.session_state.selected_feature_file)

                # 预处理
                features_scaled, feature_names, df_clean = enhance_preprocess_data(df)

                # 聚类
                labels = cluster_kmeans(features_scaled, k_value)
                df_clean['cluster'] = labels
                
                # 分析
                cluster_summary = analyze_clusters(df_clean.drop('cluster', axis=1), labels)

                # 可视化
                fig_2d = visualize_clusters_2d(features_scaled, labels, feature_names, method=dim_reduction_method.lower())
                fig_dist = visualize_feature_distribution(df_clean, labels)

                # 保存结果到 session state
                st.session_state.cluster_results = {
                    "df_with_labels": df_clean,
                    "summary": cluster_summary,
                    "fig_2d": fig_2d,
                    "fig_dist": fig_dist,
                    "k": k_value,
                    "method": dim_reduction_method
                }
                st.success("✅ 聚类分析完成！")

            except Exception as e:
                st.error(f"聚类分析过程中出现错误: {e}")
                st.exception(e) # 显示详细错误信息
                st.session_state.cluster_results = None

# --- 结果展示 ---
if st.session_state.cluster_results:
    st.markdown("---")
    st.markdown("### 📊 聚类结果")

    results = st.session_state.cluster_results
    
    st.markdown(f"#### K = {results['k']} 的 {results['method']} 降维可视化")
    st.pyplot(results['fig_2d'])

    st.markdown("#### 各聚类中特征的分布")
    st.pyplot(results['fig_dist'])

    st.markdown("#### 聚类统计摘要")
    st.dataframe(results['summary'])
    
    # --- 下载结果 ---
    st.markdown("#### 下载结果")
    output_basename = os.path.basename(st.session_state.selected_feature_file).replace('_features.xlsx', '')
    output_filename = f"{output_basename}_clustered_k{results['k']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.xlsx"
    output_path = os.path.join(results_dir, output_filename)

    # 需要一个函数来将dataframe转换为in-memory excel
    @st.cache_data
    def to_excel(df):
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        processed_data = output.getvalue()
        return processed_data

    excel_data = to_excel(results['df_with_labels'])

    st.download_button(
        label="📥 下载带有聚类标签的Excel文件",
        data=excel_data,
        file_name=output_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.info(f"提示：结果将保存为 `{output_filename}`")

else:
    st.info("点击上方按钮开始分析。")
