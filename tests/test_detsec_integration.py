import numpy as np
import tensorflow as tf
from models.feature_extract.detsec_model import DETSECModel, detsec_ae

def test_detsec_model():
    print("Testing DETSECModel...")
    # Generate dummy data: (10 samples, 20 timesteps, 1 feature)
    n_samples = 10
    seq_len = 20
    n_features = 1
    X = np.random.rand(n_samples, seq_len, n_features).astype(np.float32)
    lengths = np.random.randint(10, 21, size=n_samples).astype(np.int32)
    
    config = {
        "latent_dim": 16,
        "epochs": 2,
        "pretrain_epochs": 1,
        "batch_size": 4,
        "n_clusters": 2,
        "lengths": lengths
    }
    
    # Test wrapper function
    print("Testing detsec_ae wrapper...")
    features, history = detsec_ae(X, config)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Training history keys: {history.keys()}")
    
    assert features.shape == (n_samples, 16)
    assert 'loss' in history
    assert history['epochs_trained'] == 2
    
    print("DETSECModel test passed!")

if __name__ == "__main__":
    try:
        test_detsec_model()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
