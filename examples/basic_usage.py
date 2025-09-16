# Basic Usage Example 
from core.sentinel import Sentinel 
import numpy as np 
 
# Initialize the security system 
sentinel = Sentinel() 
 
# Process input with protection 
sample_data = np.random.rand(28, 28) 
result = sentinel.process_input(sample_data) 
 
print(f"Input shape: {sample_data.shape}") 
print(f"Output shape: {result.shape}") 
print("Security protection active!") 
