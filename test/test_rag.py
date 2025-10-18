#!/usr/bin/env python3
"""
Test script for RAG functionality
"""
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import RAGPipeline, RAGReranker

def test_rag():
    print("=== Testing RAG Functionality ===")
    
    # Initialize RAG pipeline
    try:
        rag = RAGPipeline(
            kb_dir="kb",
            embed_model="models/embeds/bge-small-en-v1.5",
            device="cuda:0",
            local_only=True
        )
        print("[OK] RAGPipeline initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize RAGPipeline: {e}")
        return False
    
    # Build or load index
    try:
        rag.build_or_load(rebuild=True)
        print("[OK] RAG index built/loaded")
    except Exception as e:
        print(f"[ERROR] Failed to build/load RAG index: {e}")
        return False
    
    # Test retrieval
    test_queries = [
        "Who is Bomi?",
        "What is Bomi's educational background?",
        "What are Bomi's hobbies?",
        "Tell me about Bomi's personality"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            results = rag.retrieve(query, topk=3)
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  Result {i}: {result['title']} - Score: {result['score']:.4f}")
                    print(f"    Preview: {result['chunk'][:100]}...")
            else:
                print("  No results found")
        except Exception as e:
            print(f"  [ERROR] Retrieval failed: {e}")
    
    # Test with reranker
    try:
        reranker = RAGReranker(
            model_name="models/reranker/bge-reranker-base",
            device="cuda:0",
            local_only=True
        )
        print("\n[OK] Reranker initialized")
        
        # Test one query with reranking
        query = "Who is Bomi?"
        print(f"\nQuery with reranking: {query}")
        results = rag.retrieve(query, topk=5)
        if results:
            reranked = reranker.rerank(query, results, topk=3)
            for i, result in enumerate(reranked, 1):
                print(f"  Reranked {i}: {result['title']} - Score: {result['score']:.4f}")
                print(f"    Preview: {result['chunk'][:100]}...")
        
    except Exception as e:
        print(f"\n[ERROR] Reranker test failed: {e}")
    
    print("\n=== RAG Test Complete ===")
    return True

if __name__ == "__main__":
    success = test_rag()
    sys.exit(0 if success else 1)