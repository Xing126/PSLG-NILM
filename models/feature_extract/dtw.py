import os
import numpy as np
from tslearn.metrics import cdist_dtw
from tslearn.utils import to_time_series_dataset
from models.base_model import BaseModel

class DTWModel(BaseModel):
    """
    DTW 特征提取类
    
    该类通过计算样本与一组基准原型（Prototypes）之间的 DTW 距离来提取特征。
    继承自 BaseModel 以符合项目开发规范。
    """
    def __init__(self, name="DTW", config=None):
        super().__init__(name, config)
        self.latent_dim = self.config.get("latent_dim", 10)  # 原型数量
        self.prototypes = None

    def train(self, data):
        """
        “训练”过程：从数据中选取原型
        data: np.ndarray, 形状为 (n_samples, timesteps, n_features)
        """
        X = data
        if isinstance(data, dict):
            X = data.get('X')
            
        n_samples = X.shape[0]
        
        # 简单策略：选取前 latent_dim 个样本作为原型
        # 实际应用中可以考虑 K-Means 聚类中心
        num_protos = min(self.latent_dim, n_samples)
        self.prototypes = X[:num_protos]
        
        print(f"[{self.name}] 训练完成，选取了 {num_protos} 个原型。")
        
        # DTW 无需训练历史，返回空字典或基本信息
        training_history = {
            'loss': [0.0],
            'val_loss': [0.0],
            'epochs_trained': 1,
            'model_name': self.name
        }
        return training_history

    def extract_features(self, data):
        """计算到原型的 DTW 距离作为特征"""
        X = data
        if isinstance(data, dict):
            X = data.get('X')
            
        if self.prototypes is None:
            raise ValueError("Model must be trained or loaded before extracting features.")
            
        print(f"[{self.name}] 正在计算 DTW 距离特征...")
        # tslearn 的 cdist_dtw 接受 (n_samples, timesteps, n_features) 格式
        # 结果形状为 (n_samples, num_protos)
        feature_matrix = cdist_dtw(X, self.prototypes)
        
        return feature_matrix

    def save(self, path: str):
        """保存原型"""
        if not os.path.exists(path):
            os.makedirs(path)
        if self.prototypes is not None:
            np.save(os.path.join(path, f"{self.name}_prototypes.npy"), self.prototypes)

    def load(self, path: str):
        """加载原型"""
        proto_path = os.path.join(path, f"{self.name}_prototypes.npy")
        if os.path.exists(proto_path):
            self.prototypes = np.load(proto_path)

def dtw_feature_extract(data: np.ndarray, config: dict):
    """
    DTW 特征提取包装函数，适配 FeatureExtractStep
    """
    model = DTWModel(config=config)
    training_history = model.train(data)
    features = model.extract_features(data)
    return features, training_history
