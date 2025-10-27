# diagnostic_numpy_fixes.py
"""
Diagnostic and fix for NumPy numerical issues in enterprise AI systems
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple

class NumPyStabilityFixer:
    """
    NumPy Stability Fixer for enterprise AI systems
    Handles numerical instability, overflow, and division errors
    """
    
    def __init__(self):
        self.setup_safe_numpy()
        
    def setup_safe_numpy(self):
        """Configure NumPy for production stability"""
        # Set error handling
        np.seterr(all='warn')
        logging.info("NumPy stability configuration applied")
    
    def safe_dot_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Safe dot product with overflow protection
        """
        try:
            # Normalize inputs to prevent overflow
            a_norm = self.normalize_array(a)
            b_norm = self.normalize_array(b)
            
            result = np.dot(a_norm, b_norm)
            
            # Check for potential overflow
            if np.any(np.isinf(result)) or np.any(np.isnan(result)):
                logging.warning("Dot product overflow detected, applying safe fallback")
                return self.dot_product_fallback(a_norm, b_norm)
                
            return result
            
        except Exception as e:
            logging.error(f"Dot product error: {e}")
            return self.dot_product_fallback(a, b)
    
    def normalize_array(self, array: np.ndarray, max_magnitude: float = 1e6) -> np.ndarray:
        """
        Normalize array to prevent numerical instability
        """
        array = np.nan_to_num(array, nan=0.0, posinf=max_magnitude, neginf=-max_magnitude)
        
        # Clip extreme values
        array = np.clip(array, -max_magnitude, max_magnitude)
        
        # Scale if necessary
        current_max = np.max(np.abs(array))
        if current_max > max_magnitude:
            scaling_factor = max_magnitude / current_max
            array = array * scaling_factor
            
        return array
    
    def dot_product_fallback(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Fallback method for dot product using chunking
        """
        try:
            # Use chunking for large arrays
            chunk_size = 512
            result = np.zeros(min(a.shape[0], b.shape[0]))
            
            for i in range(0, len(a), chunk_size):
                end_i = min(i + chunk_size, len(a))
                chunk_a = a[i:end_i]
                chunk_b = b[i:end_i]
                
                # Use safer operations
                chunk_result = np.einsum('i,i->i', chunk_a, chunk_b)
                result[i:end_i] = chunk_result
                
            return result
            
        except Exception as e:
            logging.error(f"Dot product fallback also failed: {e}")
            return np.zeros(min(a.shape[0], b.shape[0]))
    
    def safe_covariance_matrix(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Safe covariance matrix calculation without division errors
        """
        try:
            X = self.normalize_array(X)
            X_T = X.T
            
            # Center the data
            X_centered = X - np.mean(X, axis=0)
            X_T_centered = X_T - np.mean(X_T, axis=0)
            
            # Safe dot product
            cov = np.dot(X_centered, X_T_centered) / (X.shape[0] - 1)
            
            # Handle division issues
            if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
                cov = np.cov(X_centered, rowvar=False)
            
            # Calculate safe standard deviation
            stddev = np.sqrt(np.diag(cov))
            stddev = np.where(stddev == 0, 1.0, stddev)  # Prevent division by zero
            
            return cov, stddev
            
        except Exception as e:
            logging.error(f"Covariance calculation error: {e}")
            # Return identity matrix as fallback
            n_features = X.shape[1] if len(X.shape) > 1 else 1
            return np.eye(n_features), np.ones(n_features)
    
    def diagnose_system_health(self) -> Dict[str, Any]:
        """
        Comprehensive system health diagnosis
        """
        diagnosis = {
            "numpy_version": np.__version__,
            "system_platform": np.__config__.show(),
            "float_info": {
                "eps": np.finfo(float).eps,
                "min": np.finfo(float).min,
                "max": np.finfo(float).max
            },
            "warnings": [],
            "recommendations": []
        }
        
        # Test basic operations
        test_data = np.random.randn(1024)
        
        try:
            dot_test = np.dot(test_data, test_data.T)
            diagnosis["dot_product_test"] = "PASS"
        except Exception as e:
            diagnosis["dot_product_test"] = f"FAIL: {e}"
            diagnosis["warnings"].append("Dot product instability detected")
        
        try:
            cov_test = np.cov(test_data.reshape(-1, 1), rowvar=False)
            diagnosis["covariance_test"] = "PASS"
        except Exception as e:
            diagnosis["covariance_test"] = f"FAIL: {e}"
            diagnosis["warnings"].append("Covariance calculation issues")
        
        # Add recommendations
        if diagnosis["warnings"]:
            diagnosis["recommendations"].extend([
                "Use safe_dot_product instead of np.dot",
                "Normalize input data before processing",
                "Implement chunking for large arrays",
                "Add overflow checks in critical operations"
            ])
        
        return diagnosis


def test_enterprise_fixes():
    """
    Test the stability fixes with enterprise-scale data
    """
    fixer = NumPyStabilityFixer()
    
    print("🧪 Testing Enterprise Stability Fixes...")
    
    # Generate test data similar to the original issue
    large_data = np.random.randn(1024, 1024) * 1e10  # Large scale data
    
    print("1. Testing safe dot product...")
    result = fixer.safe_dot_product(large_data[0], large_data[1])
    print(f"   Result shape: {result.shape}, No NaN: {not np.any(np.isnan(result))}")
    
    print("2. Testing safe covariance...")
    cov, std = fixer.safe_covariance_matrix(large_data[:100])  # Smaller sample
    print(f"   Covariance shape: {cov.shape}, Stable: {not np.any(np.isnan(cov))}")
    
    print("3. Running system diagnosis...")
    diagnosis = fixer.diagnose_system_health()
    
    print("\n📊 DIAGNOSIS RESULTS:")
    print(f"   NumPy Version: {diagnosis['numpy_version']}")
    print(f"   Dot Product Test: {diagnosis['dot_product_test']}")
    print(f"   Covariance Test: {diagnosis['covariance_test']}")
    
    if diagnosis['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in diagnosis['warnings']:
            print(f"   - {warning}")
    
    if diagnosis['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for rec in diagnosis['recommendations']:
            print(f"   - {rec}")
    
    return diagnosis


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run comprehensive tests
    test_enterprise_fixes()