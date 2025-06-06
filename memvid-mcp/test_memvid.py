#!/usr/bin/env python3
"""Test memvid library directly"""

import json
from pathlib import Path
from memvid import MemvidEncoder, MemvidRetriever

def test_memvid():
    """Test memvid library functionality"""
    print("Testing memvid library...")
    
    # Create test directory
    test_dir = Path("./test_memories")
    test_dir.mkdir(exist_ok=True)
    
    # Test data
    memories = [
        "User's name is Bob Redford",
        "User prefers dark mode interfaces", 
        "User is learning Python programming",
        "User likes to code in the morning",
        "User's favorite IDE is VS Code"
    ]
    
    print("\n1. Creating encoder...")
    try:
        # Try with config
        config = {
            'chunk_size': 512,
            'overlap': 50,
            'model_name': 'all-MiniLM-L6-v2'
        }
        encoder = MemvidEncoder(config=config)
    except:
        # Try without config
        encoder = MemvidEncoder()
    
    print("2. Adding memories...")
    for memory in memories:
        encoder.add_text(memory)
    
    print("3. Building video...")
    video_path = test_dir / "test.mp4"
    index_path = test_dir / "test_index.json"
    encoder.build_video(str(video_path), str(index_path))
    
    print(f"   Video size: {video_path.stat().st_size / 1024:.2f} KB")
    
    print("\n5. Testing retrieval...")
    retriever = MemvidRetriever(str(video_path), str(index_path))
    
    queries = ["Mark Angler", "dark mode", "programming"]
    for query in queries:
        results = retriever.search(query, top_k=3)
        print(f"\n   Query: '{query}'")
        print(f"   Results type: {type(results)}")
        if results:
            for result in results:
                print(f"   - Result: {result}")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_memvid()