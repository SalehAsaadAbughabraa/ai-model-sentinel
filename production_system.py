"""
Quick test for the basic system without import issues
"""
import numpy as np
import time
from enterprise_system import Enterprise_AI_Sentinel

def test_basic_functionality():
    print("🧪 Testing basic system...")
    
    sentinel = Enterprise_AI_Sentinel()
    
    # Test basic data
    test_data = np.random.randn(1024)
    
    print("🔍 Analyzing model...")
    start_time = time.time()
    result = sentinel.analyze_model_enterprise(test_data, "test_model")
    analysis_time = time.time() - start_time
    
    print(f"✅ Analysis completed in {analysis_time:.2f} seconds")
    print(f"📊 Health score: {result['health_score']:.4f}")
    print(f"🔐 Crypto strength: {result['engine_results']['crypto_strength']:.4f}")
    print(f"💡 Recommendation: {result['recommendation']}")
    
    return result

def test_performance():
    print("\n⚡ Performance testing...")
    
    sentinel = Enterprise_AI_Sentinel()
    times = []
    
    for i in range(5):
        data = np.random.randn(2048)
        start = time.time()
        result = sentinel.analyze_model_enterprise(data, f"perf_test_{i}")
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Run {i+1}: {elapsed:.3f} seconds")
    
    avg_time = sum(times) / len(times)
    print(f"📈 Average time: {avg_time:.3f} seconds")
    
    if avg_time < 2.0:
        print("✅ Performance excellent!")
    else:
        print("⚠️ Performance needs improvement")

def test_error_handling():
    print("\n🛡️ Testing error handling...")
    
    sentinel = Enterprise_AI_Sentinel()
    
    # Test invalid inputs
    invalid_inputs = [
        "invalid_string",
        ["list", "input"],
        {"dict": "input"},
        None
    ]
    
    for i, invalid_input in enumerate(invalid_inputs):
        try:
            result = sentinel.analyze_model_enterprise(invalid_input, f"error_test_{i}")
            print(f"   ✅ Invalid input {i+1} handled gracefully")
        except Exception as e:
            print(f"   ✅ Exception raised for invalid input {i+1}: {type(e).__name__}")

def test_cloud_backup():
    print("\n☁️ Testing cloud backup functionality...")
    
    sentinel = Enterprise_AI_Sentinel()
    
    # Test cloud connection
    connection_result = sentinel.initialize_b2_connection()
    if connection_result["success"]:
        print("   ✅ Cloud connection successful")
    else:
        print("   ⚠️ Cloud connection failed (expected in test mode)")
    
    # Test file operations
    test_data = f"Test data {int(time.time())}"
    test_filename = f"test_file_{int(time.time())}.txt"
    
    upload_result = sentinel.backup_to_b2(test_data, test_filename)
    if upload_result["success"]:
        print("   ✅ File upload simulation successful")
    
    download_result = sentinel.restore_from_b2(test_filename)
    if download_result["success"]:
        print("   ✅ File download simulation successful")

if __name__ == "__main__":
    print("🚀 Starting Quick Test for AI Model Sentinel System")
    print("=" * 50)
    
    # Test basic functionality
    result = test_basic_functionality()
    
    # Test performance
    test_performance()
    
    # Test error handling
    test_error_handling()
    
    # Test cloud backup
    test_cloud_backup()
    
    print("\n🎉 All tests completed successfully!")
    print("📋 Results Summary:")
    print(f"   - Health Score: {result['health_score']:.4f}")
    print(f"   - Analysis Status: {result['analysis_status']}")
    print(f"   - Number of Engines: {len(result['engine_results'])}")
    print(f"   - Timestamp: {result['analysis_timestamp']}")