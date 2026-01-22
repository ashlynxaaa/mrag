import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from data_loader import PartDataset

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties

# ================= 🔧 参数微调区 =================
CONFIG = {
    "MANUAL_STEP_PATH": r"D:\CODE\mraag\raw_step\1.STEP", 
    "SAMPLE_IDX": 0,          
    "DPI": 600,
    "FIG_SIZE": (18, 14),

    # 【紧凑度】: 0.7~0.8 比较合适。越小越放大。
    "ZOOM_FACTOR": 1, 

    # 【图例位置】: (x, y)。(0,0)是左下角。
    # 觉得挡住了就改这里
    "LEGEND_POS": (0.05, 0.15), 

    # --- 样式 ---
    "NODE_SIZE": 300,         # [加大] 以前是450，现在600，非常明显
    "NODE_COLOR": '#222222',  # 接近纯黑的深灰，更有质感
    "EDGE_WIDTH": 1.8,        
    "EDGE_ALPHA": 1.0,        
}
# ===============================================

RELATION_CONFIG = {
    "shared_edge":         {"color": "#000000", "label": "Shared Edge (Topology)", "priority": 0},
    "plane_parallel_area": {"color": "#1f77b4", "label": "Plane Parallel Area", "priority": 1},
    "axis_parallel_plane": {"color": "#2ca02c", "label": "Axis Parallel Plane", "priority": 2},
    "general_perp":        {"color": "#FF4500", "label": "General Perpendicularity", "priority": 3}
}

def get_real_centroids_from_step(step_path):
    if not os.path.exists(step_path): raise FileNotFoundError(f"❌ File not found: {step_path}")
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1: raise Exception("❌ STEP read failed.")
    reader.TransferRoots()
    shape = reader.Shape()
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    centroids = []
    while explorer.More():
        face = explorer.Current()
        props = GProp_GProps()
        brepgprop_SurfaceProperties(face, props)
        p = props.CentreOfMass()
        centroids.append([p.X(), p.Y(), p.Z()])
        explorer.Next()
    return np.array(centroids)

def main():
    print("Generating Clean Node Visualization...")
    try:
        dataset = PartDataset(processed_dir="data/processed")
        data = dataset[CONFIG['SAMPLE_IDX']]
        real_pos = get_real_centroids_from_step(CONFIG["MANUAL_STEP_PATH"])
        if len(real_pos) != data['face'].num_nodes:
            real_pos = real_pos[:min(len(real_pos), data['face'].num_nodes)]
    except Exception as e:
        print(f"❌ Error: {e}"); return

    fig = plt.figure(figsize=CONFIG["FIG_SIZE"]) 
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    
    # 1. 画边 (zorder=1, 在底层)
    # 先画线，这样线不会压在点上
    for edge_type, edge_index in data.edge_index_dict.items():
        _, rel_name, _ = edge_type
        style = RELATION_CONFIG.get(rel_name, {"color": "grey"})
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        mask = (src < len(real_pos)) & (dst < len(real_pos))
        src, dst = src[mask], dst[mask]
        if len(src) == 0: continue
        
        segments = np.stack((real_pos[src], real_pos[dst]), axis=1)
        lc = Line3DCollection(segments, colors=style["color"], 
                              linewidths=CONFIG["EDGE_WIDTH"], 
                              alpha=CONFIG["EDGE_ALPHA"], 
                              capstyle='round', 
                              zorder=1) # <--- 关键：边在底层
        ax.add_collection3d(lc)

    # 2. 画点 (zorder=100, 在顶层)
    # 没有文字了，只有纯粹的几何球体
    ax.scatter(real_pos[:, 0], real_pos[:, 1], real_pos[:, 2], 
               s=CONFIG["NODE_SIZE"], c=CONFIG["NODE_COLOR"], 
               depthshade=False, # 关闭光影，纯色填充
               edgecolors='none', 
               alpha=1.0, 
               zorder=100) # <--- 关键：点在最顶层，压住所有线

    # 3. 图例
    handles, labels = [], []
    sorted_rels = sorted(RELATION_CONFIG.keys(), key=lambda k: RELATION_CONFIG[k]["priority"])
    for k in sorted_rels:
        s = RELATION_CONFIG[k]
        handles.append(plt.Line2D([0], [0], color=s["color"], lw=4, solid_capstyle='round'))
        labels.append(s["label"])

    ax.legend(handles, labels, loc='lower left', 
              bbox_to_anchor=CONFIG["LEGEND_POS"], 
              fontsize=14, frameon=False, 
              title="MRAAG Relations", title_fontsize=16)

    # 4. 视角与裁剪
    x_lim = (real_pos[:,0].min(), real_pos[:,0].max())
    y_lim = (real_pos[:,1].min(), real_pos[:,1].max())
    z_lim = (real_pos[:,2].min(), real_pos[:,2].max())
    
    # 物理比例锁定 (扁的就是扁的)
    ax.set_box_aspect((x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]))
    
    # 手动缩放
    center = real_pos.mean(axis=0)
    max_range = np.array([x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]]).max() / 2.0
    
    limit = max_range * CONFIG["ZOOM_FACTOR"]
    
    ax.set_xlim(center[0] - limit, center[0] + limit)
    ax.set_ylim(center[1] - limit, center[1] + limit)
    ax.set_zlim(center[2] - limit, center[2] + limit)
    
    ax.set_axis_off() 
    ax.dist = 10 

    out_path = "results/mraag_clean_nodes.png"
    if not os.path.exists("results"): os.makedirs("results")
    
    plt.savefig(out_path, dpi=CONFIG["DPI"], bbox_inches='tight', pad_inches=0.0)
    print(f"\n✅ Image saved: {out_path}")

if __name__ == "__main__":
    main()