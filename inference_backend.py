import os
import torch
import numpy as np
import networkx as nx 
from torch_geometric.data import HeteroData

# --- 引用你的现有模块 ---
from harmo_gnn import HarmoGNN
from step_parser import preprocess_step 

class IntelliMFRBackend:
    def __init__(self, model_weight_path, device='cpu'): 
        # 1. 强制检查 GPU 兼容性 (防止老显卡报错)
        if device == 'cuda' and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] < 3 or (cap[0] == 3 and cap[1] < 7):
                print(f"[Backend Warning] GPU capability {cap} is too old (<3.7). Switching to CPU.")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        print(f"[Backend] Initializing on {self.device}...")

        # 2. 定义模型配置 (必须与训练配置一致)
        self.config = {
            'hidden_dim': 256,
            'edge_in_dim': 10,
            'num_sem_classes': 25,
            'num_inst_classes': 50,
            'num_bot_classes': 2,
            'face_in_dim': 10, 
            'topo_dim': 4      
        }

        dummy_deg = torch.ones(101, dtype=torch.long)
        self.relations = ['shared_edge', 'plane_parallel_area', 'axis_parallel_plane', 'general_perp']

        # 3. 实例化模型
        self.model = HarmoGNN(
            face_in_dim=self.config['face_in_dim'],
            edge_in_dim=self.config['edge_in_dim'],
            hidden_dim=self.config['hidden_dim'],
            relation_types=self.relations,
            num_semantic_classes=self.config['num_sem_classes'],
            num_instance_classes=self.config['num_inst_classes'],
            num_bottom_classes=self.config['num_bot_classes'],
            topology_input_dim=self.config['topo_dim'],
            deg=dummy_deg
        ).to(self.device)

        # 4. 加载权重
        if not os.path.exists(model_weight_path):
            raise FileNotFoundError(f"Model weights not found at: {model_weight_path}")
        
        checkpoint = torch.load(model_weight_path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        print("[Backend] Model loaded successfully.")

        # 语义标签映射表
        self.SEMANTIC_MAP = {
            0: "Through_Hole", 1: "Blind_Hole", 2: "Triangular_Pocket", 
            3: "Rectangular_Pocket", 4: "Rectangular_Blind_Slot", 
            5: "Circular_Slot", 6: "Rectangular_Step", 7: "Circular_Step",
            # ... 根据您的数据集补充 ...
            24: "Stock_Face" 
        }

    def _ensure_tensor(self, data, dtype):
        """万能类型转换器"""
        if data is None: return None
        if isinstance(data, torch.Tensor): return data.to(dtype=dtype)
        if isinstance(data, np.ndarray): return torch.from_numpy(data).to(dtype=dtype)
        if isinstance(data, list): return torch.tensor(data, dtype=dtype)
        return data

    def _convert_dict_to_heterodata(self, data_dict):
        """将 step_parser 的输出转换为 PyG 数据对象"""
        data = HeteroData()
        
        # 1. 节点特征
        x = None
        if 'attrs' in data_dict: x = data_dict['attrs']
        elif 'face' in data_dict and isinstance(data_dict['face'], dict): x = data_dict['face'].get('x')
        elif 'x' in data_dict: x = data_dict['x']
        
        if x is None: raise KeyError(f"Missing node features 'x' or 'attrs'. Keys: {list(data_dict.keys())}")
        data['face'].x = self._ensure_tensor(x, torch.float)
        
        # 2. 拓扑特征
        topo = data_dict.get('concatenated_topology', None)
        if topo is not None:
             data['face'].concatenated_topology = self._ensure_tensor(topo, torch.float)
        else:
            # 补全
            data['face'].concatenated_topology = torch.zeros((data['face'].x.shape[0], 4)).float()

        # 3. 边关系
        relations_dict = data_dict.get('relations', data_dict)
        for rel in self.relations:
            if rel in relations_dict:
                edge_info = relations_dict[rel]
                edge_index = edge_info.get('edge_index') if isinstance(edge_info, dict) else edge_info
                edge_attr = edge_info.get('edge_attr') if isinstance(edge_info, dict) else None

                if edge_index is not None:
                    data['face', rel, 'face'].edge_index = self._ensure_tensor(edge_index, torch.long)
                if edge_attr is not None:
                    data['face', rel, 'face'].edge_attr = self._ensure_tensor(edge_attr, torch.float)
        return data

    def process_file(self, step_file_path):
        print(f"[Backend] Processing: {step_file_path}")
        
        # A. 解析
        raw_data_dict = preprocess_step(step_file_path, label_data=None)
        if raw_data_dict is None: raise ValueError("STEP parsing failed.")

        # B. 转换
        pyg_data = self._convert_dict_to_heterodata(raw_data_dict)
        pyg_data = pyg_data.to(self.device)

        # C. 推理
        with torch.no_grad():
            (logits_sem, logits_inst, logits_bot), _ = self.model(pyg_data)

        # D. 解析 (获得分类结果)
        pred_sem = torch.argmax(logits_sem, dim=1).cpu().numpy()
        pred_bot = torch.argmax(logits_bot, dim=1).cpu().numpy()
        
        # E. 后处理 (使用几何连通性)
        results = self._format_results_via_connectivity(pred_sem, pred_bot, pyg_data)
        return results

    def _format_results_via_connectivity(self, sem_labels, bot_labels, pyg_data):
        """
        核心逻辑：使用 NetworkX 查找连通分量，解决实例分割问题
        """
        features = []
        
        # 1. 构建物理连接图
        G = nx.Graph()
        num_nodes = len(sem_labels)
        G.add_nodes_from(range(num_nodes))
        
        # 只添加 shared_edge (物理邻接边)
        if ('face', 'shared_edge', 'face') in pyg_data.edge_index_dict:
            edge_index = pyg_data['face', 'shared_edge', 'face'].edge_index.cpu().numpy()
            edges = list(zip(edge_index[0], edge_index[1]))
            G.add_edges_from(edges)
        
        # 2. 按语义类别寻找连通分量
        unique_sem_classes = np.unique(sem_labels)
        global_instance_id_counter = 0

        for sem_cls in unique_sem_classes:
            if sem_cls == 24: continue # 跳过背景
            
            # 找出该类别的所有面
            face_indices_in_cls = np.where(sem_labels == sem_cls)[0]
            if len(face_indices_in_cls) == 0: continue

            # 提取子图并找连通块
            subgraph = G.subgraph(face_indices_in_cls)
            components = list(nx.connected_components(subgraph))
            
            sem_name = self.SEMANTIC_MAP.get(sem_cls, f"Type_{sem_cls}")

            for comp in components:
                comp_faces = list(comp)
                
                # 寻找底面
                bottom_faces = []
                for fid in comp_faces:
                    if bot_labels[fid] == 1:
                        bottom_faces.append(int(fid))

                feature_obj = {
                    "id": global_instance_id_counter,
                    "type": sem_name,
                    "face_indices": [int(i) for i in comp_faces],
                    "bottom_indices": bottom_faces
                }
                features.append(feature_obj)
                global_instance_id_counter += 1
            
        return features