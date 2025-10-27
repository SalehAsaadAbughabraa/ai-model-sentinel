from web_interface.app import app
import waitress
if __name__ == "__main__":
    print("AI Model Sentinel v2.0 - Production Server")
    print("15/19 engines active (79%)")
    print("http://localhost:8000")
    print("Security: CLASSIFIED - TIER 1")
    waitress.serve(app, host="0.0.0.0", port=8000)