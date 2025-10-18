#!/usr/bin/env python3
"""
Test script for the chatbot with RAG functionality
"""
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import ChatLLM, LLMConfig, UserMemory, RAGPipeline, RAGReranker

def test_chatbot():
    print("=== Testing Chatbot with RAG ===")
    
    # Initialize memory
    mem = UserMemory(mem_dir="memory", custom_data_path="data/sft_samples.jsonl")
    print("[OK] Memory initialized")
    
    # Initialize RAG
    rag = None
    try:
        rag = RAGPipeline(
            kb_dir="kb",
            embed_model="models/embeds/bge-small-en-v1.5",
            device="cuda:0",
            local_only=True
        )
        rag.build_or_load(rebuild=False)  # Don't rebuild since we already have the index
        print("[OK] RAG initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize RAG: {e}")
        return False
    
    # Initialize reranker
    reranker = None
    try:
        reranker = RAGReranker(
            model_name="models/reranker/bge-reranker-base",
            device="cuda:0",
            local_only=True
        )
        print("[OK] Reranker initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize reranker: {e}")
    
    # Initialize LLM
    try:
        llm_cfg = LLMConfig(
            model_name="models/merged_model-20251017T181201Z-1-001/merged_model",
            enable_thinking=False,
            max_new_tokens=256
        )
        brain = ChatLLM(llm_cfg, user_memory=mem, rag=rag, reranker=reranker)
        print("[OK] LLM initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM: {e}")
        return False
    
    # Test questions
    test_questions = [
        "Who is Bomi?",
        "What is Bomi's educational background?",
        "What are Bomi's hobbies?",
        "Tell me about Bomi's personality"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("Answer: ", end="", flush=True)
        
        try:
            response_parts = []
            for part in brain.stream_reply(
                question,
                min_chars=40,
                topk=6,
                mmr_lambda=0.5,
                rerank_k=4
            ):
                print(part, end=" ", flush=True)
                response_parts.append(part)
            
            print("\n")
            brain.history.append({"role":"user","content": question})
            brain.history.append({"role":"assistant","content":" ".join(response_parts).strip()})
            
        except Exception as e:
            print(f"[ERROR] Failed to get response: {e}")
    
    print("\n=== Chatbot Test Complete ===")
    return True

if __name__ == "__main__":
    success = test_chatbot()
    sys.exit(0 if success else 1)