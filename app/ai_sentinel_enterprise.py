import redis.asyncio as redis
import json
import secrets
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Redis connection
redis_client = None

@app.on_event("startup")
async def startup_event():
    global redis_client
    try:
        redis_client = redis.from_url("redis://redis:6379/0", decode_responses=True)
        await redis_client.ping()
        print("✅ Redis connected successfully")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        redis_client = None

# نموذج طلب المسح
class ScanRequest(BaseModel):
    file_path: str

@app.get("/")
def read_root():
    return {"message": "AI Model Sentinel v2.0.0 is running!"}

@app.get("/health")
def health_check():
    redis_status = "connected" if redis_client else "disconnected"
    return {
        "status": "healthy", 
        "redis": redis_status,
        "version": "2.0.0"
    }

# إضافة endpoint جديد للكاش
@app.post("/cache/{key}")
async def cache_data(key: str, value: dict):
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not connected")
    
    await redis_client.setex(key, 3600, json.dumps(value))
    return {"status": "cached", "key": key}

@app.get("/cache/{key}")
async def get_cached_data(key: str):
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not connected")
    
    cached = await redis_client.get(key)
    if cached:
        return {"status": "hit", "data": json.loads(cached)}
    return {"status": "miss"}

# إضافة محرك المسح الأساسي
@app.post("/scan")
async def scan_file(request: ScanRequest):
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not connected")
    
    scan_id = secrets.token_hex(8)
    
    # محاكاة المسح
    scan_result = {
        "threat_level": "QUANTUM_HIGH",
        "threat_score": 0.75,
        "confidence": 0.85,
        "scan_id": scan_id,
        "file_path": request.file_path,
        "timestamp": secrets.token_hex(8)
    }
    
    # تخزين في الكاش باستخدام scan_id
    await redis_client.setex(
        f"scan:{scan_id}", 
        300,
        json.dumps(scan_result)
    )
    
    return scan_result

# إضافة endpoint لرؤية جميع عمليات المسح المخزنة
@app.get("/scans")
async def get_all_scans():
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not connected")
    
    try:
        keys = await redis_client.keys("scan:*")
        scans = {}
        for key in keys:
            scan_data = await redis_client.get(key)
            if scan_data:
                scan_json = json.loads(scan_data)
                scans[scan_json['scan_id']] = scan_json
        return {"scans": scans, "total": len(scans)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching scans: {str(e)}")

# إضافة endpoint لفحص مفتاح معين باستخدام scan_id - غير الاسم لتجنب التعارض
@app.get("/scan-result/{scan_id}")
async def get_scan_result(scan_id: str):
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not connected")
    
    scan_data = await redis_client.get(f"scan:{scan_id}")
    if scan_data:
        return {"status": "found", "data": json.loads(scan_data)}
    raise HTTPException(status_code=404, detail=f"Scan not found with ID: {scan_id}")

# إضافة endpoint لحذف مسح معين باستخدام scan_id
@app.delete("/scan-result/{scan_id}")
async def delete_scan_result(scan_id: str):
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not connected")
    
    deleted = await redis_client.delete(f"scan:{scan_id}")
    if deleted:
        return {"status": "deleted", "scan_id": scan_id}
    raise HTTPException(status_code=404, detail=f"Scan not found with ID: {scan_id}")

# endpoint للتصحيح - رؤية جميع المفاتيح
@app.get("/debug/keys")
async def debug_keys():
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not connected")
    
    try:
        all_keys = await redis_client.keys('*')
        keys_with_data = {}
        for key in all_keys:
            value = await redis_client.get(key)
            keys_with_data[key] = value
        return {"keys": keys_with_data, "total": len(keys_with_data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error debugging keys: {str(e)}")

# إضافة endpoint للبحث عن المسوحات بواسطة file_path
@app.get("/scans/search/{file_path}")
async def search_scans_by_path(file_path: str):
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not connected")
    
    try:
        keys = await redis_client.keys("scan:*")
        matching_scans = []
        
        for key in keys:
            scan_data = await redis_client.get(key)
            if scan_data:
                scan_json = json.loads(scan_data)
                if file_path in scan_json.get('file_path', ''):
                    matching_scans.append(scan_json)
        
        return {"matching_scans": matching_scans, "count": len(matching_scans)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# تنظيف جميع المسوحات (للتطوير فقط)
@app.delete("/scans/clear")
async def clear_all_scans():
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not connected")
    
    keys = await redis_client.keys("scan:*")
    if keys:
        await redis_client.delete(*keys)
        return {"status": "cleared", "deleted_count": len(keys)}
    return {"status": "no_scans_found"}

# إضافة endpoint للحصول على إحصائيات
@app.get("/stats")
async def get_stats():
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis not connected")
    
    keys = await redis_client.keys("scan:*")
    return {
        "total_scans": len(keys),
        "redis_status": "connected"
    }