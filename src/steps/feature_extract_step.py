from src.framework.step import Step
from models.feature_extract.lstm_ae import lstm_ae
from models.feature_extract.bilstm_ae import bilstm_ae
from models.feature_extract.bilstm_ae_attantion import bilstm_ae_attention

class FeatureExtractStep(Step):
    """
    Step for extracting features from time series data using autoencoders.
    """
    def __init__(
        self,
        name="FeatureExtract",
        model_name="bilstm_ae",
        latent_dim=64,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        patience=5,
        attention_size=32
    ):
        super().__init__(name)
        self.model_name = model_name
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.attention_size = attention_size

    def run(self, context: dict) -> dict:
        """
        Extract features from time series data using the selected autoencoder model.
        
        Args:
            context (dict): Shared context containing data.
            
        Returns:
            dict: Updated context with extracted features.
        """
        log_dir = self.get_log_dir(context)
        
        # Get data from context
        data = context.get('data', {})
        
        if not data:
            print("[FeatureExtract] No data found in context")
            return context
        
        # Prepare config
        config = {
            "latent_dim": self.latent_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "patience": self.patience
        }
        
        if self.model_name in ['bilstm_ae_attention']:
            config["attention_size"] = self.attention_size
        
        # Extract features for each data item
        features = {}
        for file_name, data_item in data.items():
            print(f"[FeatureExtract] Processing {file_name}")
            
            # Convert DataFrame to numpy array if needed
            if hasattr(data_item, 'to_numpy'):
                # Assuming data is in shape (timesteps, features)
                # Reshape to (1, timesteps, features) for single sample
                np_data = data_item.to_numpy()
                if len(np_data.shape) == 2:
                    np_data = np_data.reshape(1, np_data.shape[0], np_data.shape[1])
                elif len(np_data.shape) == 1:
                    np_data = np_data.reshape(1, np_data.shape[0], 1)
            else:
                np_data = data_item
            
            # Validate data shape
            if len(np_data.shape) != 3:
                print(f"[FeatureExtract] Invalid data shape for {file_name}: {np_data.shape}")
                continue
            
            # Select model and extract features
            try:
                if self.model_name == "lstm_ae":
                    extracted_features, history = lstm_ae(np_data, config)
                elif self.model_name == "bilstm_ae":
                    extracted_features, history = bilstm_ae(np_data, config)
                elif self.model_name == "bilstm_ae_attention":
                    extracted_features, history = bilstm_ae_attention(np_data, config)
                else:
                    print(f"[FeatureExtract] Unknown model: {self.model_name}")
                    continue
                
                features[file_name] = extracted_features
                print(f"[FeatureExtract] Extracted features for {file_name}: {extracted_features.shape}")
                
            except Exception as e:
                print(f"[FeatureExtract] Error processing {file_name}: {e}")
                continue
        
        # Store features in context
        if features:
            context['features'] = features
            print(f"[FeatureExtract] Extracted features for {len(features)} files")
        else:
            print("[FeatureExtract] No features extracted")
        
        return context