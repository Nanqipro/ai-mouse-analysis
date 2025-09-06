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
import numpy as np
from src.utils import save_plot_as_base64

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