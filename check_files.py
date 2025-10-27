import os

def check_mathematical_engine_files():
    base_path = "mathematical_engine"
    engines = {
        'prime_analysis': ['prime_neural_engine.py'],
        'fractal_analysis': ['fractal_analyzer.py'],
        'information_theory': ['information_theory_engine.py'],
        'golden_ratio': ['golden_ratio_analyzer.py'],
        'cryptographic_engine': ['cryptographic_engine.py']
    }
    
    print("🔍 CHECKING MATHEMATICAL ENGINE FILES...")
    print("=" * 50)
    
    for folder, files in engines.items():
        folder_path = os.path.join(base_path, folder)
        print(f"\n📁 {folder.upper()}:")
        if os.path.exists(folder_path):
            for file in files:
                file_path = os.path.join(folder_path, file)
                if os.path.exists(file_path):
                    print(f"   ✅ {file}")
                else:
                    print(f"   ❌ {file} - NOT FOUND")
                    # List what files are actually there
                    actual_files = [f for f in os.listdir(folder_path) if f.endswith('.py')]
                    if actual_files:
                        print(f"   📄 Actual files: {actual_files}")
        else:
            print(f"   ❌ Folder not found: {folder_path}")

if __name__ == "__main__":
    check_mathematical_engine_files()