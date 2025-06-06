#!/usr/bin/env python3
"""Test script for memVid MCP server"""

import asyncio
import sys
from src.memvid_mcp.server import (
    store_memory,
    search_memories,
    store_conversation,
    get_memory_context,
    consolidate_memories,
    get_memory_stats
)


async def test_memvid_server():
    """Test all MCP server functions"""
    print("Testing memVid MCP Server...")
    user_id = "test_user"
    
    # Test 1: Store memories
    print("\n1. Testing store_memory...")
    memories = [
        ("User prefers dark mode interfaces", "preference"),
        ("User is learning Python programming", "fact"),
        ("User likes to code in the morning", "preference"),
        ("User's favorite IDE is VS Code", "preference"),
        ("User is working on a machine learning project", "fact")
    ]
    
    for content, mem_type in memories:
        result = await store_memory(
            content=content,
            user_id=user_id,
            memory_type=mem_type,
            metadata={"source": "test"}
        )
        print(f"  Stored: {content[:30]}... - Success: {result.get('success')}")
    
    # Test 2: Search memories
    print("\n2. Testing search_memories...")
    queries = [
        "What are the user's UI preferences?",
        "programming languages",
        "favorite tools"
    ]
    
    for query in queries:
        result = await search_memories(
            query=query,
            user_id=user_id,
            top_k=3
        )
        print(f"  Query: {query}")
        for mem in result.get("memories", []):
            print(f"    - {mem['content']} (score: {mem['relevance_score']:.2f})")
    
    # Test 3: Store conversation
    print("\n3. Testing store_conversation...")
    conversation = [
        {"role": "user", "content": "Can you help me debug my Python code?"},
        {"role": "assistant", "content": "Of course! What issue are you experiencing?"},
        {"role": "user", "content": "I'm getting a TypeError in my neural network implementation"}
    ]
    
    result = await store_conversation(
        messages=conversation,
        user_id=user_id,
        session_id="test_session_001"
    )
    print(f"  Conversation stored - Success: {result.get('success')}")
    
    # Test 4: Get memory context
    print("\n4. Testing get_memory_context...")
    context = await get_memory_context(
        user_id=user_id,
        max_tokens=500,
        query="coding preferences"
    )
    print(f"  Context (first 200 chars): {context[:200]}...")
    
    # Test 5: Get memory stats
    print("\n5. Testing get_memory_stats...")
    stats = await get_memory_stats(user_id=user_id)
    print(f"  Stats: {stats}")
    
    # Test 6: Consolidate memories
    print("\n6. Testing consolidate_memories...")
    result = await consolidate_memories(
        user_id=user_id,
        strategy="deduplicate"
    )
    print(f"  Consolidation result: {result}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_memvid_server())