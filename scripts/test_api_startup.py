"""Test script to verify API can start up correctly.

This script tests that the FastAPI application can be imported
and that all routes are registered correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_api_startup():
    """Test that API can be imported and routes are registered."""
    print("Testing API startup...")
    
    # Import the app
    print("  ✓ Importing FastAPI app...")
    from src.api import app
    
    # Check that app is FastAPI instance
    from fastapi import FastAPI
    assert isinstance(app, FastAPI), "App is not a FastAPI instance"
    print("  ✓ App is FastAPI instance")
    
    # Check routes are registered
    routes = [route.path for route in app.routes]
    print(f"  ✓ Found {len(routes)} routes")
    
    # Check expected routes exist
    expected_routes = [
        "/",
        "/health",
        "/metrics",
        "/predict",
        "/predict/top-k",
        "/predict/batch",
        "/predict/complete",
        "/model/info",
        "/model/vocabulary",
    ]
    
    for expected_route in expected_routes:
        if expected_route in routes:
            print(f"  ✓ Route {expected_route} registered")
        else:
            print(f"  ✗ Route {expected_route} NOT found")
            print(f"    Available routes: {routes}")
            return False
    
    # Check middleware is registered
    print(f"  ✓ Found {len(app.user_middleware)} middleware")
    
    # Check OpenAPI docs
    assert app.docs_url == "/docs", "Docs URL not set correctly"
    assert app.redoc_url == "/redoc", "ReDoc URL not set correctly"
    print("  ✓ OpenAPI documentation configured")
    
    print("\n✅ API startup test passed!")
    print("\nTo start the API server, run:")
    print("  uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
    print("\nOr use the run_api.py script (if available):")
    print("  python scripts/run_api.py")
    
    return True


if __name__ == "__main__":
    try:
        success = test_api_startup()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ API startup test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
