#!/usr/bin/env python3

import asyncio
import subprocess
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

async def run_test_script(script_name):
    """Run a test script and return result"""
    try:
        print(f"\n{'='*60}")
        print(f"Running {script_name}...")
        print(f"{'='*60}")
        
        # Run the script
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.path.dirname(__file__)
        )
        
        stdout, _ = await process.communicate()
        output = stdout.decode('utf-8')
        
        print(output)
        
        if process.returncode == 0:
            print(f"✅ {script_name} PASSED")
            return True
        else:
            print(f"❌ {script_name} FAILED (exit code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {script_name} ERROR: {e}")
        return False

async def run_all_tests():
    """Run all test scripts in order"""
    
    print("🧪 Starting Comprehensive Test Suite")
    print("🎯 Testing Jaaz Backend Components")
    
    tests = [
        "test_api_connection.py",
        "test_anthropic_model.py", 
        "test_replicate_api.py",
        "test_chat_api.py",
        "test_context_debug.py"
    ]
    
    results = {}
    
    for test in tests:
        results[test] = await run_test_script(test)
        await asyncio.sleep(1)  # Brief pause between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("🏁 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    failed = 0
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The backend is working correctly.")
    else:
        print("🔧 Some tests failed. Check the output above for details.")
    
    return failed == 0

if __name__ == "__main__":
    asyncio.run(run_all_tests())