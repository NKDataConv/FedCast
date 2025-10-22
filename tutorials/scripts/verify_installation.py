#!/usr/bin/env python3
"""
FedCast Installation Verification Script

This script verifies that FedCast is properly installed and all dependencies
are available. Run this after completing Tutorial 1: Installation & Setup.

Usage:
    python verify_installation.py
"""

import sys
import importlib
from typing import List, Tuple


def check_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """
    Check if a module can be imported successfully.
    
    Args:
        module_name: Name of the module to import
        package_name: Optional package name for version checking
        
    Returns:
        Tuple of (success, message)
    """
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"✅ {module_name} v{version}"
    except (ImportError, OSError, Exception) as e:
        return False, f"❌ {module_name}: {e}"


def check_fedcast_components() -> List[Tuple[bool, str]]:
    """Check all FedCast components and dependencies."""
    results = []
    
    # Core FedCast imports
    fedcast_imports = [
        ("fedcast", "FedCast core"),
        ("fedcast.datasets", "FedCast datasets"),
        ("fedcast.cast_models", "FedCast models"),
        ("fedcast.federated_learning_strategies", "FedCast strategies"),
        ("fedcast.telemetry", "FedCast telemetry"),
    ]
    
    # External dependencies
    external_imports = [
        ("torch", "PyTorch"),
        ("flwr", "Flower"),
        ("pandas", "Pandas"),
        ("mlflow", "MLflow"),
        ("numpy", "NumPy"),
    ]
    
    print("🔍 Checking FedCast components...")
    for module, description in fedcast_imports:
        success, message = check_import(module)
        results.append((success, f"{description}: {message}"))
    
    print("\n🔍 Checking external dependencies...")
    for module, description in external_imports:
        success, message = check_import(module)
        results.append((success, f"{description}: {message}"))
    
    return results


def test_basic_functionality() -> List[Tuple[bool, str]]:
    """Test basic FedCast functionality."""
    results = []
    
    try:
        # Test dataset loading
        from fedcast.datasets import load_sinus_dataset
        dataset = load_sinus_dataset(partition_id=0, num_examples=100)
        results.append((True, f"✅ Dataset loading: {type(dataset).__name__}"))
        
        # Test model creation
        from fedcast.cast_models import MLPModel
        model = MLPModel()  # Uses WINDOW_SIZE = 20 internally
        results.append((True, f"✅ Model creation: {type(model).__name__}"))
        
        # Test strategy import (skip if Ray issues)
        try:
            from fedcast.federated_learning_strategies import build_fedavg_strategy
            strategy = build_fedavg_strategy()
            results.append((True, f"✅ Strategy creation: {type(strategy).__name__}"))
        except Exception as e:
            results.append((False, f"⚠️  Strategy creation failed (Ray issue): {e}"))
        
    except Exception as e:
        results.append((False, f"❌ Basic functionality test failed: {e}"))
    
    return results


def main():
    """Main verification function."""
    print("🚀 FedCast Installation Verification")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 9):
        print("⚠️  Warning: Python 3.9+ is recommended")
    else:
        print("✅ Python version is compatible")
    
    print("\n" + "=" * 50)
    
    # Check imports
    import_results = check_fedcast_components()
    
    # Test functionality
    print("\n🔍 Testing basic functionality...")
    functionality_results = test_basic_functionality()
    
    # Combine results
    all_results = import_results + functionality_results
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    success_count = 0
    total_count = len(all_results)
    
    for success, message in all_results:
        print(message)
        if success:
            success_count += 1
    
    print(f"\nResults: {success_count}/{total_count} checks passed")
    
    # Check if failures are only Ray-related
    ray_failures = sum(1 for success, message in all_results if not success and "ray" in message.lower())
    non_ray_failures = total_count - success_count - ray_failures
    
    if success_count == total_count:
        print("\n🎉 Congratulations! FedCast is properly installed and ready to use!")
        print("📚 Continue with Tutorial 2: Your First Federated Experiment")
        return 0
    elif non_ray_failures == 0 and ray_failures > 0:
        print("\n✅ FedCast core functionality is working!")
        print("⚠️  Ray/Flower simulation features have issues (common on macOS)")
        print("💡 You can still use FedCast for basic federated learning")
        print("📖 See Tutorial 1 troubleshooting section for Ray issues")
        print("📚 Continue with Tutorial 2: Your First Federated Experiment")
        return 0
    else:
        print(f"\n⚠️  {non_ray_failures} core issues found. Please check the installation.")
        if ray_failures > 0:
            print(f"ℹ️  {ray_failures} Ray-related issues (see troubleshooting)")
        print("📖 Refer to Tutorial 1: Installation & Setup for troubleshooting")
        return 1


if __name__ == "__main__":
    sys.exit(main())
