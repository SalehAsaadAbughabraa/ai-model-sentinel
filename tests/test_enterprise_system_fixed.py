import time
import numpy as np
import pytest
import requests
from enterprise_system import Enterprise_AI_Sentinel


# -------- FUNCTIONAL TESTS --------
def test_model_analysis_basic():
    sentinel = Enterprise_AI_Sentinel()
    data = np.random.randn(1024)
    result = sentinel.analyze_model_enterprise(data, "test_model")
    assert 0 <= result["health_score"] <= 1
    assert "recommendation" in result
    assert "engine_results" in result
    assert "crypto_strength" in result["engine_results"]
    print("✅ Functional test passed")


def test_invalid_input_handling():
    sentinel = Enterprise_AI_Sentinel()
    with pytest.raises((ValueError, TypeError)):
        sentinel.analyze_model_enterprise("invalid_string", "bad_model")


# -------- PERFORMANCE TESTS --------
def test_performance_speed():
    sentinel = Enterprise_AI_Sentinel()
    data = np.random.randn(2048)
    start = time.time()
    sentinel.analyze_model_enterprise(data, "perf_test")
    elapsed = (time.time() - start) * 1000
    print(f"⚡ Analysis time: {elapsed:.2f} ms")
    assert elapsed < 2000  # 2 seconds max acceptable


# -------- SECURITY TESTS --------
def test_no_sensitive_info_in_exceptions():
    sentinel = Enterprise_AI_Sentinel()
    try:
        sentinel.analyze_model_enterprise("invalid", "model")
    except Exception as e:
        error_msg = str(e).lower()
        assert "password" not in error_msg
        assert "key" not in error_msg
        assert "secret" not in error_msg
        print("✅ Security exception handling OK")


# -------- BACKBLAZE B2 CLOUD BACKUP TESTS --------
@pytest.mark.cloud
def test_b2_connection():
    """Test Backblaze B2 connection and authentication"""
    sentinel = Enterprise_AI_Sentinel()
    
    # Test connection initialization
    connection_status = sentinel.initialize_b2_connection()
    assert connection_status["success"] == True
    assert "connected" in str(connection_status).lower()
    print("✅ B2 connection test passed")


@pytest.mark.cloud
def test_b2_bucket_access():
    """Test bucket access and listing"""
    sentinel = Enterprise_AI_Sentinel()
    
    # Test bucket listing
    buckets = sentinel.list_b2_buckets()
    assert isinstance(buckets, list)
    print(f"✅ Found {len(buckets)} buckets")


@pytest.mark.cloud
def test_b2_file_upload_download():
    """Test file upload and download cycle"""
    sentinel = Enterprise_AI_Sentinel()
    
    # Create test data
    test_data = "AI_Sentinel_Test_File_" + str(int(time.time()))
    test_filename = f"test_backup_{int(time.time())}.txt"
    
    # Upload test file
    upload_result = sentinel.backup_to_b2(test_data, test_filename)
    assert upload_result["success"] == True
    assert upload_result["filename"] == test_filename
    print(f"✅ File upload successful: {test_filename}")
    
    # Download and verify
    download_result = sentinel.restore_from_b2(test_filename)
    assert download_result["success"] == True
    assert test_data in str(download_result["content"])
    print("✅ File download and verification successful")


@pytest.mark.cloud
def test_b2_large_file_handling():
    """Test handling of larger files"""
    sentinel = Enterprise_AI_Sentinel()
    
    # Create larger test data (1MB)
    large_data = "X" * 1024 * 1024
    large_filename = f"large_test_{int(time.time())}.dat"
    
    upload_result = sentinel.backup_to_b2(large_data, large_filename)
    assert upload_result["success"] == True
    assert upload_result["size"] > 0
    print("✅ Large file handling test passed")


# -------- INTEGRATION TESTS (Flask API) --------
@pytest.mark.integration
def test_api_endpoints():
    try:
        base_url = "http://localhost:5000"
        endpoints = [
            "/api/v1/enterprise/health",
            "/api/v1/enterprise/system/info"
        ]
        for ep in endpoints:
            r = requests.get(base_url + ep, timeout=5)
            assert r.status_code == 200
            print(f"✅ API endpoint OK: {ep}")
    except requests.exceptions.ConnectionError:
        pytest.skip("Flask server not running")


# -------- STRESS TEST --------
@pytest.mark.slow
def test_multiple_analyses():
    sentinel = Enterprise_AI_Sentinel()
    for i in range(10):  # خفّضت العدد لأداء أفضل
        data = np.random.randn(2048)
        result = sentinel.analyze_model_enterprise(data, f"stress_{i}")
        assert result["health_score"] > 0
    print("✅ Stress test passed (10 analyses)")


# -------- BACKUP/RESTORE COMPLETE WORKFLOW --------
@pytest.mark.cloud
@pytest.mark.slow
def test_complete_backup_workflow():
    """Test complete backup and restore workflow"""
    sentinel = Enterprise_AI_Sentinel()
    
    # Initialize cloud connection
    assert sentinel.initialize_b2_connection()["success"] == True
    
    # Create test model analysis
    model_data = np.random.randn(1024)
    analysis_result = sentinel.analyze_model_enterprise(model_data, "backup_test_model")
    
    # Backup analysis results
    backup_result = sentinel.backup_analysis_results(analysis_result, "backup_test_model")
    assert backup_result["success"] == True
    
    # Simulate system restore
    restore_result = sentinel.restore_analysis_results("backup_test_model")
    assert restore_result["success"] == True
    assert "health_score" in restore_result["data"]
    
    print("✅ Complete backup/restore workflow test passed")


if __name__ == "__main__":
    # Run basic tests directly
    test_model_analysis_basic()
    test_b2_connection()
    test_b2_file_upload_download()
    print("🎉 All core tests passed!")