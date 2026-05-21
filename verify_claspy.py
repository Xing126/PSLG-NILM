import sys
import os

# Add project root to path to simulate running from project root
sys.path.append(os.getcwd())

try:
    from models.time_segmentation.claspy.segmentation import BinaryClaSPSegmentation
    print("Successfully imported BinaryClaSPSegmentation from models.time_segmentation.claspy.segmentation")
    
    import numpy as np
    ts = np.random.rand(100)
    clasp = BinaryClaSPSegmentation(window_size=10)
    # Just check if it can be initialized and if internal imports work
    print("Successfully initialized BinaryClaSPSegmentation")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
