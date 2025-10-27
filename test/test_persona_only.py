#!/usr/bin/env python3
"""
Simple test to verify the persona string contains the correct identity information.
This test doesn't require all the dependencies to be installed.
"""

import sys
import os
import re

def test_persona_string():
    """Test that the persona string contains the correct identity information"""
    
    # Read the run.py file
    with open("run.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the persona string
    persona_match = re.search(r'self\.persona = \'\'\'(.*?)\'\'\'', content, re.DOTALL)
    
    if not persona_match:
        print("❌ Could not find persona string in run.py")
        return False
    
    persona = persona_match.group(1)
    
    print("Testing persona string...")
    print("=" * 50)
    print("Persona content:")
    print(persona)
    print("=" * 50)
    
    # Check for key identity elements
    checks = [
        ("Tomoki name", "tomoki" in persona.lower()),
        ("Not Toto", "not toto" in persona.lower() or "not be toto" in persona.lower()),
        ("Not Teeravist", "not teeravist" in persona.lower() or "not be teeravist" in persona.lower()),
        ("Name response", "name" in persona.lower() and "tomoki" in persona.lower()),
        ("Identity separation", "user" in persona.lower() and "tomoki" in persona.lower())
    ]
    
    all_passed = True
    for check_name, passed in checks:
        if passed:
            print(f"[PASS] {check_name}: PASSED")
        else:
            print(f"[FAIL] {check_name}: FAILED")
            all_passed = False
    
    # Check for the Tomoki identity file
    tomoki_file_exists = os.path.exists("kb/tomoki_identity.md")
    if tomoki_file_exists:
        print("[PASS] Tomoki identity file exists")
    else:
        print("[FAIL] Tomoki identity file missing")
        all_passed = False
    
    # Check for identity query detection function
    identity_func_exists = "_is_identity_query" in content
    if identity_func_exists:
        print("[PASS] Identity query detection function exists")
    else:
        print("[FAIL] Identity query detection function missing")
        all_passed = False
    
    # Check for modified RAG retrieval
    modified_rag = "if self._is_identity_query(user_text)" in content
    if modified_rag:
        print("[PASS] Modified RAG retrieval for identity queries exists")
    else:
        print("[FAIL] Modified RAG retrieval for identity queries missing")
        all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("[SUCCESS] All tests passed! The chatbot should now correctly identify as Tomoki.")
    else:
        print("[WARNING] Some tests failed. Please review the implementation.")
    
    return all_passed

if __name__ == "__main__":
    test_persona_string()