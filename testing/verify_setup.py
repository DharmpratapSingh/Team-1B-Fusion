#!/usr/bin/env python3
"""
Setup Verification Script

Checks if all required files exist and dependencies are installed.

Usage:
    python verify_setup.py
"""

import sys
from pathlib import Path
import json

print("=" * 80)
print("Testing Infrastructure Setup Verification")
print("=" * 80)

all_ok = True

# Check required files
print("\n1. Checking required files...")
required_files = [
    "test_harness.py",
    "analyze_results.py",
    "test_config.json",
    "test_question_bank.json",
    "requirements_testing.txt",
    "TESTING_QUICKSTART.md",
    "TEST_HARNESS_USAGE.md"
]

for filename in required_files:
    if Path(filename).exists():
        print(f"  ✓ {filename}")
    else:
        print(f"  ✗ {filename} - MISSING!")
        all_ok = False

# Check question bank
print("\n2. Checking question bank...")
try:
    with open("test_question_bank.json", 'r') as f:
        data = json.load(f)
        questions = data.get('questions', [])
        print(f"  ✓ Question bank loaded: {len(questions)} questions")

        if len(questions) != 50:
            print(f"  ⚠ Warning: Expected 50 questions, found {len(questions)}")
except Exception as e:
    print(f"  ✗ Error loading question bank: {e}")
    all_ok = False

# Check Python dependencies
print("\n3. Checking Python dependencies...")
dependencies = {
    'requests': 'HTTP client',
    'json': 'JSON handling (built-in)',
    'pathlib': 'Path handling (built-in)',
}

for module, description in dependencies.items():
    try:
        __import__(module)
        print(f"  ✓ {module} - {description}")
    except ImportError:
        if module not in ['json', 'pathlib']:  # Skip built-ins
            print(f"  ✗ {module} - NOT INSTALLED")
            print(f"     Install with: pip install {module}")
            all_ok = False

# Optional dependencies
print("\n4. Checking optional dependencies (for analysis)...")
optional = {
    'pandas': 'Data analysis',
    'matplotlib': 'Visualization',
    'seaborn': 'Statistical plots'
}

for module, description in optional.items():
    try:
        __import__(module)
        print(f"  ✓ {module} - {description}")
    except ImportError:
        print(f"  ⚠ {module} - Not installed ({description})")
        print(f"     Install with: pip install {module}")

# Check configuration
print("\n5. Checking configuration...")
try:
    with open("test_config.json", 'r') as f:
        config = json.load(f)
        print(f"  ✓ Configuration loaded")
        print(f"     ClimateGPT: {config.get('climategpt', {}).get('url')}")
        print(f"     Llama: {config.get('llama', {}).get('url')}")
except Exception as e:
    print(f"  ✗ Error loading config: {e}")
    all_ok = False

# Check if scripts are executable
print("\n6. Checking script permissions...")
scripts = ["test_harness.py", "analyze_results.py"]
for script in scripts:
    path = Path(script)
    if path.exists():
        import os
        if os.access(path, os.X_OK):
            print(f"  ✓ {script} - Executable")
        else:
            print(f"  ⚠ {script} - Not executable (run: chmod +x {script})")
    else:
        print(f"  ✗ {script} - Not found")

# Test imports from test_harness
print("\n7. Testing test_harness imports...")
try:
    import requests
    print("  ✓ requests module works")
except ImportError as e:
    print(f"  ✗ requests import failed: {e}")
    print("     Install with: pip install requests")
    all_ok = False

# Summary
print("\n" + "=" * 80)
if all_ok:
    print("✅ SETUP VERIFICATION PASSED!")
    print("\nYou're ready to run tests!")
    print("\nNext steps:")
    print("  1. Start ClimateGPT: make serve")
    print("  2. Start LM Studio (manually)")
    print("  3. Run pilot test: python test_harness.py --pilot")
    print("  4. Analyze results: python analyze_results.py")
else:
    print("❌ SETUP VERIFICATION FAILED!")
    print("\nPlease fix the issues above before running tests.")
    print("\nCommon fixes:")
    print("  - Install dependencies: pip install -r requirements_testing.txt")
    print("  - Make scripts executable: chmod +x test_harness.py analyze_results.py")
    sys.exit(1)

print("=" * 80)
