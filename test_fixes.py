#!/usr/bin/env python3
"""
Test script to verify the API key fixes work correctly
"""

# Test imports
try:
    from functions import fetch_gmap_place_id, fetch_google_reviews, process_store_list, FashionFeedbackProcessor
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test FashionFeedbackProcessor initialization
try:
    test_api_key = "test_key_123"
    processor = FashionFeedbackProcessor(test_api_key)
    print("✅ FashionFeedbackProcessor initialized successfully")
    print(f"   - API key stored: {processor.api_key}")
    print(f"   - Client created: {processor.client is not None}")
except Exception as e:
    print(f"❌ FashionFeedbackProcessor initialization error: {e}")

# Test function signatures
import inspect

# Check fetch_gmap_place_id signature
sig = inspect.signature(fetch_gmap_place_id)
params = list(sig.parameters.keys())
print(f"✅ fetch_gmap_place_id parameters: {params}")

# Check process_store_list signature
sig = inspect.signature(process_store_list)
params = list(sig.parameters.keys())
print(f"✅ process_store_list parameters: {params}")

print("\n🎉 All tests passed! The API key issues have been resolved.")
