import torch
import os
import numpy as np
from harmo_gnn import HarmoGNN  # 导入你的模型定义
# 假设你有一个将 STEP 转为 Data 对象的函数，通常在你的 data_loader.py 里
# from data_processor import step_to_graph_data 

class InferenceBackend:
    def __init__(self, model_path, device='cuda'):
        """
        初始化后端引擎，加载模型权重
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Backend on {self.device}...")

        # 1. 定义模型配置 (必须与训练时一致)
        # 注意：这里的参数需要和你 best_model.pth 训练时的参数完全对应
        self.model_config = {
            'hidden_dim': 256,
            'edge_in_dim': 10,
            'num_sem_classes': 25,
            'num_inst_classes': 50,
            'num_bot_classes': 2,
            # 下面两个参数通常需要根据第一个样本动态获取，但在推理时可以硬编码或由预处理决定
            'face_in_dim': 64, # 假设值，请根据你的 .pt 文件确认
            'topo_dim': 4      # 假设值
        }

        # 2. 初始化模型架构
        # 注意：deg 参数在推理时如果不想重新计算，可以传入一个平均分布或者加载训练时的 deg
        # 这里为了简化，我们先设为 None，或者你需要加载训练好的 degree histogram
        dummy_deg = torch.ones(101).float() 
        
        self.model = HarmoGNN(
            face_in_dim=self.model_config['face_in_dim'],
            edge_in_dim=self.model_config['edge_in_dim'],
            hidden_dim=self.model_config['hidden_dim'],
            relation_types=['shared', 'parallel', 'perp', 'coaxial'], # 确保包含所有关系
            num_semantic_classes=self.model_config['num_sem_classes'],
            num_instance_classes=self.model_config['num_inst_classes'],
            num_bottom_classes=self.model_config['num_bot_classes'],
            topology_input_dim=self.model_config['topo_dim'],
            deg=dummy_deg
        ).to(self.device)

        # 3. 加载权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            # 如果保存的是 state_dict
            self.model.load_state_dict(checkpoint, strict=False)
            print("Model weights loaded successfully.")
        else:
            raise FileNotFoundError(f"Model path {model_path} does not exist!")

        self.model.eval() # 切换到评估模式

    def preprocess(self, step_file_path):
        """
        步骤 1: 将 STEP 文件转换为 PyTorch Geometric Data 对象
        """
        print(f"Processing {step_file_path}...")
        
        # --- 关键点 ---
        # 这里你需要调用你生成 .pt 文件的逻辑
        # 实际项目中，这里会调用 PythonOCC 读取 STEP，提取拓扑，生成 Tensor
        # 为了演示，假设我们直接加载已经生成好的 .pt 文件 (模拟)
        
        pt_file_path = step_file_path.replace(".step", ".pt")
        if not os.path.exists(pt_file_path):
            raise ValueError("对应 .pt 文件未找到，请确保已运行预处理脚本。")
            
        data = torch.load(pt_file_path)
        return data

    def predict(self, step_file_path):
        """
        步骤 2 & 3: 推理与解析
        """
        # 1. 获取图数据
        data = self.preprocess(step_file_path)
        data = data.to(self.device)

        # 2. 模型前向传播
        with torch.no_grad():
            # 注意：如果你的模型 forward 需要 apply_ablation_mask，这里也要加上
            # out_all, _ = self.model(data) 
            # 假设 model forward 返回 (out_sem, out_inst, out_bot), contrast_out
            (logits_sem, logits_inst, logits_bot), _ = self.model(data)

        # 3. 解析结果 (Argmax 获取类别索引)
        pred_sem = torch.argmax(logits_sem, dim=1).cpu().numpy()
        pred_inst = torch.argmax(logits_inst, dim=1).cpu().numpy()
        pred_bot = torch.argmax(logits_bot, dim=1).cpu().numpy()

        # 4. 格式化输出 (Post-processing)
        # 我们需要将结果转换成前端能看懂的结构：
        # List of Features: [{Type: 'Blind_Hole', Faces: [1, 2, 3]}, ...]
        
        results = self.format_results(pred_sem, pred_inst, pred_bot, step_file_path)
        return results

    def format_results(self, sem_labels, inst_labels, bot_labels, file_path):
        """
        将扁平的标签数组转换为结构化的特征列表
        """
        # 语义类别映射表 (根据你的数据集定义)
        SEMANTIC_NAMES = {
            0: "Through_Hole", 1: "Blind_Hole", 2: "Triangular_Pocket", 
            3: "Rectangular_Pocket", 24: "None", # 假设24是背景面
            # ... 补充你的映射 ...
        }

        # 简单的实例聚类逻辑：
        # 实际上你应该结合 Semantic 和 Instance Label 来分组
        
        unique_instances = np.unique(inst_labels)
        feature_report = []

        for inst_id in unique_instances:
            # 找到属于该实例的所有面索引
            face_indices = np.where(inst_labels == inst_id)[0]
            
            # 获取这些面的语义投票 (多数表决)
            sem_votes = sem_labels[face_indices]
            majority_sem = np.bincount(sem_votes).argmax()
            
            sem_name = SEMANTIC_NAMES.get(majority_sem, f"Unknown_{majority_sem}")

            # 过滤掉非特征面 (比如 Type 'None')
            if sem_name == "None":
                continue

            # 找到底面
            # bot_labels == 1 表示是底面
            bottom_faces = []
            for face_idx in face_indices:
                if bot_labels[face_idx] == 1:
                    bottom_faces.append(int(face_idx))

            feature_report.append({
                "instance_id": int(inst_id),
                "type": sem_name,
                "face_ids": [int(i) for i in face_indices], # 这里的索引对应 STEP 解析时的顺序
                "bottom_face_ids": bottom_faces,
                "confidence": 0.95 # 这里可以计算 softmax 概率作为置信度
            })

        return feature_report