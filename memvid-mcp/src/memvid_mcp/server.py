"""memVid MCP Server - Main server implementation"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastmcp import FastMCP, Context
from memvid import MemvidEncoder, MemvidRetriever

# Initialize FastMCP server
mcp = FastMCP("memVid Memory Server")

# Memory managers
encoder: Optional[MemvidEncoder] = None
retriever: Optional[MemvidRetriever] = None

# Configuration
MEMORY_PATH = os.getenv("MEMORY_PATH", "./memories")
Path(MEMORY_PATH).mkdir(exist_ok=True)


def get_memory_paths() -> tuple[str, str]:
    """Get video and index paths"""
    return (
        f"{MEMORY_PATH}/memories.mp4",
        f"{MEMORY_PATH}/memories_index.json"
    )


def get_or_create_encoder() -> MemvidEncoder:
    """Get or create encoder"""
    global encoder
    if encoder is None:
        config = {
            'chunk_size': 512,
            'overlap': 50,
            'model_name': 'all-MiniLM-L6-v2'
        }
        encoder = MemvidEncoder(config=config)
    return encoder


def get_or_create_retriever() -> Optional[MemvidRetriever]:
    """Get or create retriever"""
    global retriever
    if retriever is None:
        video_path, index_path = get_memory_paths()
        try:
            retriever = MemvidRetriever(video_path, index_path)
        except:
            return None
    return retriever


def load_memory_metadata() -> Dict:
    """Load metadata for memories"""
    try:
        metadata_path = f"{MEMORY_PATH}/metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
    except:
        pass
    return {"memories": []}


def save_memory_metadata(metadata: Dict) -> None:
    """Save metadata for memories"""
    metadata_path = f"{MEMORY_PATH}/metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


@mcp.tool()
async def store_memory(
    content: str,
    memory_type: str = "general",
    metadata: Optional[Dict] = None,
    ctx: Context = None
) -> Dict:
    """Store a new memory"""
    try:
        if ctx:
            ctx.info("Storing memory")
        
        # Get or create encoder
        encoder = get_or_create_encoder()
        
        # Prepare memory with metadata
        timestamp = datetime.now(timezone.utc).isoformat()
        memory_entry = {
            "content": content,
            "timestamp": timestamp,
            "memory_type": memory_type,
            "metadata": metadata or {}
        }
        
        # Load existing metadata
        metadata_obj = load_memory_metadata()
        metadata_obj["memories"].append(memory_entry)
        
        # Encode all memories to video
        encoder.clear()  # Clear any existing data
        for memory in metadata_obj["memories"]:
            encoder.add_text(memory["content"])
        
        video_path, index_path = get_memory_paths()
        encoder.build_video(video_path, index_path)
        
        # Save updated metadata
        save_memory_metadata(metadata_obj)
        
        # Update retriever
        global retriever
        retriever = MemvidRetriever(video_path, index_path)
        
        return {
            "success": True,
            "memory_id": len(metadata_obj["memories"]) - 1,
            "timestamp": timestamp
        }
    except Exception as e:
        if ctx:
            ctx.error(f"Failed to store memory: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def search_memories(
    query: str,
    top_k: int = 5,
    memory_type: Optional[str] = None,
    ctx: Context = None
) -> List[Dict]:
    """Search memories using semantic search
    
    Tips for better results:
    - Use keywords that might appear in the memory content
    - The search matches content, not intent
    """
    try:
        if ctx:
            ctx.info("Searching memories")
        
        # Get retriever
        retriever = get_or_create_retriever()
        if not retriever:
            return {"memories": []}
        
        # Search memories
        results = retriever.search(query, top_k=top_k * 2)  # Get extra to filter
        
        # Load metadata to match results with full memory info
        metadata_obj = load_memory_metadata()
        memories = []
        
        # Results are returned as list of text strings
        for result_text in results:
            # Find matching memory in metadata
            for memory in metadata_obj["memories"]:
                if memory["content"] == result_text:
                    # Filter by memory type if specified
                    if memory_type and memory.get("memory_type") != memory_type:
                        continue
                    
                    memories.append({
                        "content": memory["content"],
                        "relevance_score": 1.0,  # Memvid doesn't return scores
                        "timestamp": memory["timestamp"],
                        "memory_type": memory.get("memory_type", "general"),
                        "metadata": memory.get("metadata", {})
                    })
                    
                    if len(memories) >= top_k:
                        break
            
            if len(memories) >= top_k:
                break
        
        return {"memories": memories}
    except Exception as e:
        if ctx:
            ctx.error(f"Failed to search memories: {str(e)}")
        return {"memories": []}


@mcp.tool()
async def store_conversation(
    messages: List[Dict[str, str]], 
    session_id: str = None,
    metadata: Optional[Dict] = None,
    ctx: Context = None
) -> Dict:
    """Store a conversation in video memory"""
    try:
        # Generate session_id if not provided
        if session_id is None:
            session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        if ctx:
            ctx.info(f"Storing conversation, session {session_id}")
        
        # Format conversation as a single memory
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in messages
        ])
        
        # Store as a memory with conversation metadata
        conv_metadata = {
            "session_id": session_id,
            "message_count": len(messages),
            **(metadata or {})
        }
        
        return await store_memory(
            content=conversation_text,
            memory_type="conversation",
            metadata=conv_metadata,
            ctx=ctx
        )
    except Exception as e:
        if ctx:
            ctx.error(f"Failed to store conversation: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_memory_context(
    max_tokens: int = 2000,
    query: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Get contextual memory summary
    
    Returns a formatted text summary of memories. If query is provided,
    searches for relevant memories. Otherwise returns recent memories.
    
    For structured memory data, use search_memories instead.
    """
    try:
        if ctx:
            ctx.info("Getting memory context")
        
        if query:
            # Search for relevant memories
            results = await search_memories(
                query=query,
                top_k=10,
                ctx=ctx
            )
            memories = results.get("memories", [])
        else:
            # Get recent memories
            metadata_obj = load_memory_metadata()
            memories = [
                {
                    "content": m["content"],
                    "timestamp": m["timestamp"],
                    "memory_type": m.get("memory_type", "general")
                }
                for m in metadata_obj["memories"][-10:]
            ]
        
        # Build context string
        context_parts = []
        total_tokens = 0
        
        for memory in memories:
            memory_text = f"[{memory['memory_type']} - {memory['timestamp']}]\n{memory['content']}\n"
            # Simple token estimation (4 chars per token)
            estimated_tokens = len(memory_text) // 4
            
            if total_tokens + estimated_tokens > max_tokens:
                break
            
            context_parts.append(memory_text)
            total_tokens += estimated_tokens
        
        return "\n".join(context_parts)
    except Exception as e:
        if ctx:
            ctx.error(f"Failed to get memory context: {str(e)}")
        return ""


@mcp.tool()
async def consolidate_memories(
    strategy: str = "deduplicate",
    ctx: Context = None
) -> Dict:
    """Consolidate and optimize memories"""
    try:
        if ctx:
            ctx.info(f"Consolidating memories with strategy {strategy}")
        
        metadata_obj = load_memory_metadata()
        original_count = len(metadata_obj["memories"])
        
        if strategy == "deduplicate":
            # Remove duplicate memories based on content similarity
            seen_content = set()
            unique_memories = []
            
            for memory in metadata_obj["memories"]:
                # Simple deduplication based on exact content match
                content_key = memory["content"].strip().lower()
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_memories.append(memory)
            
            metadata_obj["memories"] = unique_memories
        
        elif strategy == "compress":
            # Group similar memories together
            # This is a simplified version - could use embeddings for better grouping
            grouped_memories = {}
            
            for memory in metadata_obj["memories"]:
                mem_type = memory.get("memory_type", "general")
                if mem_type not in grouped_memories:
                    grouped_memories[mem_type] = []
                grouped_memories[mem_type].append(memory)
            
            # Keep representative memories from each group
            compressed_memories = []
            for mem_type, memories in grouped_memories.items():
                # Keep most recent N memories of each type
                memories.sort(key=lambda m: m["timestamp"], reverse=True)
                compressed_memories.extend(memories[:100])  # Keep max 100 per type
            
            metadata_obj["memories"] = compressed_memories
        
        # Re-encode memories
        if metadata_obj["memories"]:
            encoder = get_or_create_encoder()
            video_path, index_path = get_memory_paths()
            
            encoder.clear()
            for memory in metadata_obj["memories"]:
                encoder.add_text(memory["content"])
            encoder.build_video(video_path, index_path)
            
            # Update retriever
            global retriever
            retriever = MemvidRetriever(video_path, index_path)
        
        # Save updated metadata
        save_memory_metadata(metadata_obj)
        
        final_count = len(metadata_obj["memories"])
        
        return {
            "success": True,
            "original_count": original_count,
            "final_count": final_count,
            "removed": original_count - final_count,
            "strategy": strategy
        }
    except Exception as e:
        if ctx:
            ctx.error(f"Failed to consolidate memories: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.resource("memory://stats")
async def get_memory_stats() -> Dict:
    """Get memory statistics"""
    try:
        video_path, index_path = get_memory_paths()
        metadata_obj = load_memory_metadata()
        
        stats = {
            "chunk_count": len(metadata_obj["memories"]),
            "video_exists": os.path.exists(video_path),
            "index_exists": os.path.exists(index_path)
        }
        
        if os.path.exists(video_path):
            stats["video_size_mb"] = os.path.getsize(video_path) / (1024 * 1024)
            
            # Calculate compression ratio
            total_text_size = sum(len(m["content"]) for m in metadata_obj["memories"])
            if total_text_size > 0:
                stats["compression_ratio"] = total_text_size / os.path.getsize(video_path)
            else:
                stats["compression_ratio"] = 0
        
        # Memory type breakdown
        memory_types = {}
        for memory in metadata_obj["memories"]:
            mem_type = memory.get("memory_type", "general")
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
        stats["memory_types"] = memory_types
        
        return stats
    except Exception as e:
        return {
            "error": str(e)
        }


@mcp.prompt()
def memory_extraction_prompt(conversation: str) -> str:
    """Generate a prompt for extracting memories from conversation"""
    return f"""Analyze the following conversation and extract key memories about the user.
Focus on:
- User preferences and settings
- Personal information shared
- Important decisions or choices made
- Behavioral patterns
- Stated goals or intentions

Conversation:
{conversation}

Extract memories as a list of concise statements, each representing a distinct piece of information about the user.
Format each memory as a clear, standalone fact that would be useful for future interactions."""


def main():
    """Main entry point for the MCP server"""
    import sys
    
    # Ensure memories directory exists
    os.makedirs(MEMORY_PATH, exist_ok=True)
    
    # Run the server
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\nShutting down memVid MCP server...")
        sys.exit(0)


if __name__ == "__main__":
    main()