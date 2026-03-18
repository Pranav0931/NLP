"""
Setup Validation Script
Verifies that all required packages are installed and accessible.
Usage: python verify_setup.py
"""

import sys
import importlib

# Required packages and minimum versions
REQUIRED_PACKAGES = {
    'pandas': '2.0.0',
    'numpy': '1.24.0',
    'matplotlib': '3.7.0',
    'seaborn': '0.12.0',
    'sklearn': '1.2.0',
    'nltk': '3.8.1',
    'transformers': '4.30.0',
    'torch': '2.0.0',
    'gradio': '3.32.0',
}

def check_package(package_name, min_version=None):
    """Check if a package is installed and return version."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def main():
    """Main validation function."""
    print_header("🔍 Sentiment Analysis Project - Setup Verification")
    
    packages_ok = True
    results = []
    
    print("\n📦 Checking Python packages...\n")
    
    for package, min_version in REQUIRED_PACKAGES.items():
        is_installed, version = check_package(package)
        status = "✅" if is_installed else "❌"
        
        if is_installed:
            results.append(f"{status} {package:<20} {version}")
        else:
            results.append(f"{status} {package:<20} NOT INSTALLED")
            packages_ok = False
    
    # Print results
    for result in results:
        print(result)
    
    # Python version check
    print_header("Python Environment")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    
    # Summary
    print_header("Validation Summary")
    
    if packages_ok:
        print("✅ All required packages are installed!")
        print("\n🚀 You're ready to run the notebook!")
        print("\nNext steps:")
        print("  1. Download IMDB Dataset.csv from Kaggle")
        print("  2. Place it in the same directory as the notebook")
        print("  3. Run: jupyter notebook Movie_Review_Sentiment_Analysis_Project.ipynb")
        return 0
    else:
        print("❌ Some packages are missing. Please install them using:")
        print("\n   pip install -r requirements.txt")
        print("\nOr install individually:")
        missing = [pkg for pkg, _ in REQUIRED_PACKAGES.items() 
                  if not check_package(pkg)[0]]
        for pkg in missing:
            print(f"   pip install {pkg}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
