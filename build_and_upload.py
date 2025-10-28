# build_and_upload.py
import subprocess
import sys
from pathlib import Path

class PackageBuilder:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
    
    def install_build_tools(self):
        print("Installing build tools...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "build", "twine", "setuptools", "wheel"], 
                         check=True)
            print("✅ Build tools installed")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install build tools: {e}")
            return False
    
    def build_package(self):
        print("Building package...")
        try:
            result = subprocess.run([sys.executable, "-m", "build"], 
                                  cwd=self.project_root, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Package built successfully")
                # عرض الملفات المبنية
                dist_dir = self.project_root / "dist"
                if dist_dir.exists():
                    files = list(dist_dir.glob("*"))
                    print(f"📦 Built files: {[f.name for f in files]}")
                return True
            else:
                print(f"❌ Build failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Build error: {e}")
            return False
    
    def check_package(self):
        print("Checking package...")
        try:
            result = subprocess.run([sys.executable, "-m", "twine", "check", "dist/*"], 
                                  cwd=self.project_root, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Package check passed")
                return True
            else:
                print(f"❌ Package check failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Check error: {e}")
            return False
    
    def upload_to_test_pypi(self):
        print("Uploading to Test PyPI...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "twine", "upload", "dist/*", 
                "--repository-url", "https://test.pypi.org/legacy/",
                "--verbose"
            ], cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("🎉 Upload to Test PyPI successful!")
                print("📦 Available at: https://test.pypi.org/project/ai-model-sentinel/")
                return True
            else:
                print(f"❌ Upload failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return False

if __name__ == "__main__":
    builder = PackageBuilder(".")
    
    print("🚀 Starting PyPI deployment process...")
    print("📁 Project:", builder.project_root)
    
    if builder.install_build_tools():
        if builder.build_package():
            if builder.check_package():
                print("\n" + "="*50)
                print("READY TO UPLOAD TO PyPI!")
                print("="*50)
                
                response = input("Upload to Test PyPI? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    builder.upload_to_test_pypi()
                else:
                    print("Package built successfully. You can upload manually later.")
            else:
                print("❌ Package check failed. Please fix issues above.")
        else:
            print("❌ Package build failed. Please check the setup files.")
    else:
        print("❌ Could not install build tools.")