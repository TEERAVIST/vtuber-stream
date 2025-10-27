#!/usr/bin/env python3
"""
Test script to verify that the chatbot correctly identifies itself as Tomoki
and not as Toto/Teeravist when asked about its name.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import ChatLLM, LLMConfig, UserMemory

def test_identity_response():
    """Test that the chatbot correctly identifies itself as Tomoki"""
    
    # Initialize components
    llm_cfg = LLMConfig(model_name="Qwen/Qwen3-0.6B", enable_thinking=False)
    mem = UserMemory(mem_dir="test_memory", custom_data_path="data/sft_samples.jsonl")
    
    # Create chatbot instance
    chatbot = ChatLLM(llm_cfg, user_memory=mem, rag=None, reranker=None)
    
    # Test questions about identity
    identity_questions = [
        "What is your name?",
        "Who are you?",
        "What should I call you?",
        "Are you Toto?",
        "Are you Teeravist?",
        "Are you Tomoki?"
    ]
    
    print("Testing identity responses...")
    print("=" * 50)
    
    for question in identity_questions:
        print(f"\nQuestion: {question}")
        print("Response: ", end="", flush=True)
        
        # Get response from chatbot
        response_parts = []
        for part in chatbot.stream_reply(question):
            response_parts.append(part)
            print(part, end=" ", flush=True)
        
        response = " ".join(response_parts).strip()
        print()
        
        # Check if response correctly identifies as Tomoki (for identity questions)
        if any(keyword in question.lower() for keyword in ["what is your name", "who are you", "what should i call you"]):
            if "tomoki" in response.lower() and ("toto" not in response.lower() or "not toto" in response.lower()):
                print("✅ Correctly identified as Tomoki")
            else:
                print("❌ Failed to correctly identify as Tomoki")
        
        # Check if response correctly denies being Toto/Teeravist
        if "are you toto" in question.lower() or "are you teeravist" in question.lower():
            if "not" in response.lower() and ("toto" in response.lower() or "teeravist" in response.lower()):
                print("✅ Correctly denied being Toto/Teeravist")
            else:
                print("❌ Failed to deny being Toto/Teeravist")
        
        # Check if response correctly confirms being Tomoki
        if "are you tomoki" in question.lower():
            if "tomoki" in response.lower() and ("yes" in response.lower() or "i am" in response.lower()):
                print("✅ Correctly confirmed being Tomoki")
            else:
                print("❌ Failed to confirm being Tomoki")
    
    print("\n" + "=" * 50)
    print("Identity testing complete!")

if __name__ == "__main__":
    test_identity_response()