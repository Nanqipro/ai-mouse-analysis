from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import List, Optional
import os
import shutil
import tempfile
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

# 导入核心逻辑模块
from src.extraction_logic import run_batch_extraction, extract_calcium_features, get_interactive_data, extract_manual_range
from src.clustering_logic import (
    load_data,
    enhance_preprocess_data,
    cluster_kmeans,
    visualize_clusters_2d,
    visualize_feature_distribution,
    analyze_clusters
)
from src.heatmap_behavior import (
    BehaviorHeatmapConfig,
    load_and_validate_data,
    find_behavior_pairs,
    extract_behavior_sequence_data,
    standardize_neural_data,
    create_behavior_sequence_heatmap,
    create_average_sequence_heatmap,
    get_global_neuron_order
)
from src.overall_heatmap import (
    OverallHeatmapConfig,
    generate_overall_heatmap
)
import numpy as np
from src.utils import save_plot_as_base64
import base64
from io import BytesIO

app = FastAPI(title="钙信号分析平台 API", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],  # Vue开发服务器地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建必要的目录
UPLOADS_DIR = Path("uploads")
RESULTS_DIR = Path("results")
TEMP_DIR = Path("temp")

for dir_path in [UPLOADS_DIR, RESULTS_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True)

@app.get("/")
async def root():
    return {"message": "钙信号分析平台 API"}

@app.post("/api/extraction/preview")
async def preview_extraction(
    file: UploadFile = File(...),
    fs: float = Form(4.8),
    min_duration_frames: int = Form(12),
    max_duration_frames: int = Form(800),
    min_snr: float = Form(3.5),
    smooth_window: int = Form(31),
    peak_distance_frames: int = Form(24),
    filter_strength: float = Form(1.0),
    neuron_id: str = Form(...)
):
    """预览单个神经元的事件提取结果"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 读取数据
        df = pd.read_excel(temp_file, sheet_name='dF', header=0)
        
        # 如果neuron_id是'temp'，只返回神经元列表
        if neuron_id == 'temp':
            temp_file.unlink()
            return {
                "success": True,
                "neuron_columns": df.columns[1:].tolist(),
                "features": [],
                "plot": None
            }
        
        if neuron_id not in df.columns:
            raise HTTPException(status_code=400, detail=f"神经元 {neuron_id} 不存在")
        
        # 设置参数
        params = {
            'min_duration': min_duration_frames,
            'max_duration': max_duration_frames,
            'min_snr': min_snr,
            'smooth_window': smooth_window,
            'peak_distance': peak_distance_frames,
            'filter_strength': filter_strength
        }
        
        # 提取特征并生成可视化
        feature_table, fig, _ = extract_calcium_features(
            df[neuron_id].values, fs=fs, visualize=True, params=params
        )
        
        # 将图表转换为base64
        plot_base64 = save_plot_as_base64(fig)
        
        # 清理临时文件
        temp_file.unlink()
        
        return {
            "success": True,
            "features": feature_table.to_dict('records') if not feature_table.empty else [],
            "plot": plot_base64,
            "neuron_columns": df.columns[1:].tolist()
        }
        
    except Exception as e:
        # 清理临时文件
        if temp_file.exists():
            temp_file.unlink()
        print(f"Error in preview_extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extraction/interactive_data")
async def get_interactive_extraction_data(
    file: UploadFile = File(...),
    neuron_id: str = Form(...)
):
    """获取交互式图表数据"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 获取交互式数据
        interactive_data = get_interactive_data(str(temp_file), neuron_id)
        
        # 清理临时文件
        temp_file.unlink()
        
        return {
            "success": True,
            "data": interactive_data
        }
        
    except Exception as e:
        # 清理临时文件
        if temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extraction/manual_extract")
async def manual_extraction(
    file: UploadFile = File(...),
    neuron_id: str = Form(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
    fs: float = Form(4.8),
    min_duration_frames: int = Form(5),
    max_duration_frames: int = Form(100),
    min_snr: float = Form(2.0),
    smooth_window: int = Form(5),
    peak_distance_frames: int = Form(10),
    filter_strength: float = Form(0.1)
):
    """基于用户选择的时间范围进行手动提取"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 构建参数字典
        params = {
            'fs': fs,
            'min_duration_frames': min_duration_frames,
            'max_duration_frames': max_duration_frames,
            'min_snr': min_snr,
            'smooth_window': smooth_window,
            'peak_distance_frames': peak_distance_frames,
            'filter_strength': filter_strength
        }
        
        # 执行手动提取
        result = extract_manual_range(str(temp_file), neuron_id, start_time, end_time, params)
        
        # 清理临时文件
        temp_file.unlink()
        
        return result
        
    except Exception as e:
        # 清理临时文件
        if temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extraction/batch")
async def batch_extraction(
    files: List[UploadFile] = File(...),
    fs: float = Form(4.8),
    min_duration_frames: int = Form(12),
    max_duration_frames: int = Form(800),
    min_snr: float = Form(3.5),
    smooth_window: int = Form(31),
    peak_distance_frames: int = Form(24),
    filter_strength: float = Form(1.0)
):
    """批量处理文件进行事件提取"""
    try:
        # 创建时间戳目录
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        upload_dir = UPLOADS_DIR / timestamp
        upload_dir.mkdir(exist_ok=True)
        
        # 保存上传的文件
        saved_file_paths = []
        for file in files:
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_file_paths.append(str(file_path))
        
        # 设置参数
        params = {
            'min_duration': min_duration_frames,
            'max_duration': max_duration_frames,
            'min_snr': min_snr,
            'smooth_window': smooth_window,
            'peak_distance': peak_distance_frames,
            'filter_strength': filter_strength
        }
        
        # 执行批量提取
        result_path = run_batch_extraction(saved_file_paths, str(RESULTS_DIR), fs=fs, **params)
        
        if result_path and os.path.exists(result_path):
            return {
                "success": True,
                "result_file": os.path.basename(result_path),
                "message": "批量分析完成"
            }
        else:
            raise HTTPException(status_code=500, detail="批量分析未生成任何结果")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/files")
async def list_result_files():
    """获取结果文件列表"""
    try:
        feature_files = list(RESULTS_DIR.glob("*_features.xlsx"))
        files_info = []
        
        for file_path in feature_files:
            try:
                # 尝试从文件名解析时间戳
                basename = file_path.name
                timestamp_str = basename.split('_features.xlsx')[0].split('_')[-1]
                dt_obj = datetime.strptime(timestamp_str, '%Y%m%d-%H%M%S')
                friendly_name = f"{basename} (创建于: {dt_obj.strftime('%Y-%m-%d %H:%M:%S')})"
            except (ValueError, IndexError):
                friendly_name = basename
            
            files_info.append({
                "filename": basename,
                "friendly_name": friendly_name,
                "path": str(file_path)
            })
        
        return {"files": files_info}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clustering/analyze")
async def clustering_analysis(
    filename: str = Form(...),
    k_value: int = Form(3),
    dim_reduction_method: str = Form("pca")
):
    """执行聚类分析"""
    try:
        file_path = RESULTS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 加载数据
        df = load_data(str(file_path))
        
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
        
        # 转换图表为base64
        plot_2d_base64 = save_plot_as_base64(fig_2d)
        plot_dist_base64 = save_plot_as_base64(fig_dist)
        
        # 保存结果
        output_basename = filename.replace('_features.xlsx', '')
        output_filename = f"{output_basename}_clustered_k{k_value}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.xlsx"
        output_path = RESULTS_DIR / output_filename
        df_clean.to_excel(output_path, index=False)
        
        return {
            "success": True,
            "summary": cluster_summary.to_dict('records'),
            "plot_2d": plot_2d_base64,
            "plot_dist": plot_dist_base64,
            "result_file": output_filename,
            "k_value": k_value,
            "method": dim_reduction_method
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/behavior/detect")
async def detect_behavior_events(
    file: UploadFile = File(...)
):
    """检测行为事件配对"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"behavior_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"检测行为事件，文件: {temp_file}")
        
        # 加载数据
        data = load_and_validate_data(str(temp_file))
        print(f"数据加载成功，形状: {data.shape}")
        
        # 获取所有唯一的行为类型
        unique_behaviors = data['behavior'].unique().tolist()
        print(f"发现的行为类型: {unique_behaviors}")
        
        # 查找所有可能的行为配对（使用默认参数）
        behavior_events = []
        
        # 遍历所有可能的行为配对组合
        for start_behavior in unique_behaviors:
            for end_behavior in unique_behaviors:
                if start_behavior != end_behavior:
                    try:
                        pairs = find_behavior_pairs(
                            data, start_behavior, end_behavior, 
                            min_duration=1.0, sampling_rate=4.8
                        )
                        
                        for i, (start_begin, start_end, end_begin, end_end) in enumerate(pairs):
                            behavior_events.append({
                                'index': len(behavior_events) + 1,
                                'start_behavior': start_behavior,
                                'end_behavior': end_behavior,
                                'start_time': float(start_begin / 4.8),  # 转换为秒
                                'end_time': float(end_end / 4.8),  # 转换为秒
                                'duration': float((end_end - start_begin) / 4.8)  # 转换为秒
                            })
                    except Exception as e:
                        print(f"查找行为配对 {start_behavior} -> {end_behavior} 时出错: {e}")
                        continue
        
        print(f"检测到 {len(behavior_events)} 个行为事件配对")
        
        # 清理临时文件
        if temp_file.exists():
            temp_file.unlink()
        
        return {
            "success": True,
            "behavior_events": behavior_events,
            "available_behaviors": unique_behaviors,
            "message": f"检测到 {len(behavior_events)} 个行为事件配对"
        }
        
    except Exception as e:
        print(f"行为事件检测错误: {e}")
        # 清理临时文件
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=f"行为事件检测失败: {str(e)}")

@app.post("/api/heatmap/analyze")
async def heatmap_analysis(
    file: UploadFile = File(...),
    start_behavior: str = Form(...),
    end_behavior: str = Form(...),
    pre_behavior_time: float = Form(10.0),
    min_duration: float = Form(1.0),
    sampling_rate: float = Form(4.8)
):
    """热力图分析"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 创建配置对象
        config = BehaviorHeatmapConfig()
        config.INPUT_FILE = str(temp_file)
        config.START_BEHAVIOR = start_behavior
        config.END_BEHAVIOR = end_behavior
        config.PRE_BEHAVIOR_TIME = pre_behavior_time
        config.MIN_BEHAVIOR_DURATION = min_duration
        config.SAMPLING_RATE = sampling_rate
        config.OUTPUT_DIR = str(RESULTS_DIR / "heatmaps")
        config.SORTING_METHOD = 'first'
        
        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # 加载数据
        data = load_and_validate_data(str(temp_file))
        
        # 查找行为配对
        behavior_pairs = find_behavior_pairs(
            data, start_behavior, end_behavior, 
            min_duration, sampling_rate
        )
        
        if not behavior_pairs:
            raise HTTPException(status_code=400, detail="未找到符合条件的行为配对")
        
        # 提取所有行为序列数据
        all_sequence_data = []
        heatmap_images = []
        first_heatmap_order = None
        
        for i, (start_begin, start_end, end_begin, end_end) in enumerate(behavior_pairs):
            # 提取行为序列数据
            sequence_data = extract_behavior_sequence_data(
                data, start_begin, end_end, pre_behavior_time, sampling_rate
            )
            
            if sequence_data is not None:
                # 标准化数据
                standardized_data = standardize_neural_data(sequence_data)
                all_sequence_data.append(standardized_data)
                
                # 创建热力图
                fig, current_order = create_behavior_sequence_heatmap(
                    standardized_data, start_begin, end_end,
                    start_behavior, end_behavior, pre_behavior_time,
                    config, i, first_heatmap_order=first_heatmap_order
                )
                
                # 保存第一个热力图的排序顺序
                if i == 0 and current_order is not None:
                    first_heatmap_order = current_order
                
                # 将图表转换为base64
                plot_base64 = save_plot_as_base64(fig)
                heatmap_images.append({
                    "title": f"行为配对 {i+1} 热力图",
                    "url": f"data:image/png;base64,{plot_base64}"
                })
        
        # 创建平均热力图
        if len(all_sequence_data) > 1:
            avg_fig = create_average_sequence_heatmap(
                all_sequence_data, start_behavior, end_behavior,
                pre_behavior_time, config, first_heatmap_order=first_heatmap_order
            )
            avg_plot_base64 = save_plot_as_base64(avg_fig)
            heatmap_images.append({
                "title": "平均热力图",
                "url": f"data:image/png;base64,{avg_plot_base64}"
            })
        
        # 清理临时文件
        temp_file.unlink()
        
        # 获取神经元数量
        neuron_columns = [col for col in data.columns if col not in ['behavior']]
        
        return {
            "success": True,
            "filename": file.filename,
            "behavior_pairs_count": len(behavior_pairs),
            "neuron_count": len(neuron_columns),
            "start_behavior": start_behavior,
            "end_behavior": end_behavior,
            "status": "分析完成",
            "heatmap_images": heatmap_images
        }
        
    except Exception as e:
        # 清理临时文件
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/heatmap/overall")
async def overall_heatmap_analysis(
    file: UploadFile = File(...),
    stamp_min: Optional[float] = Form(None),
    stamp_max: Optional[float] = Form(None),
    sort_method: str = Form("peak"),
    calcium_wave_threshold: float = Form(1.5),
    min_prominence: float = Form(1.0),
    min_rise_rate: float = Form(0.1),
    max_fall_rate: float = Form(0.05)
):
    """整体热力图分析"""
    try:
        # 保存上传的文件
        temp_file = TEMP_DIR / f"overall_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"整体热力图分析，文件: {temp_file}")
        
        # 读取数据
        data = pd.read_excel(temp_file)
        print(f"数据加载成功，形状: {data.shape}")
        
        # 创建配置对象
        config = OverallHeatmapConfig()
        config.STAMP_MIN = stamp_min
        config.STAMP_MAX = stamp_max
        config.SORT_METHOD = sort_method
        config.CALCIUM_WAVE_THRESHOLD = calcium_wave_threshold
        config.MIN_PROMINENCE = min_prominence
        config.MIN_RISE_RATE = min_rise_rate
        config.MAX_FALL_RATE = max_fall_rate
        
        # 生成整体热力图
        fig, info = generate_overall_heatmap(data, config)
        
        # 将图表转换为base64
        plot_base64 = save_plot_as_base64(fig)
        
        # 清理临时文件
        temp_file.unlink()
        
        return {
            "success": True,
            "filename": file.filename,
            "heatmap_image": f"data:image/png;base64,{plot_base64}",
            "analysis_info": info,
            "config": {
                "stamp_min": stamp_min,
                "stamp_max": stamp_max,
                "sort_method": sort_method,
                "calcium_wave_threshold": calcium_wave_threshold,
                "min_prominence": min_prominence,
                "min_rise_rate": min_rise_rate,
                "max_fall_rate": max_fall_rate
            },
            "message": "整体热力图生成完成"
        }
        
    except Exception as e:
        print(f"整体热力图分析错误: {e}")
        # 清理临时文件
        if 'temp_file' in locals() and temp_file.exists():
            temp_file.unlink()
        raise HTTPException(status_code=500, detail=f"整体热力图分析失败: {str(e)}")

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """下载结果文件"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)