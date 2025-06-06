"""memVid MCP Server - Main server implementation"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastmcp import FastMCP, Context
from memvid import MemvidEncoder, MemvidRetriever

# Initialize FastMCP server
mcp = FastMCP("memVid Memory Server")

# User-specific managers
encoders: Dict[str, MemvidEncoder] = {}
retrievers: Dict[str, MemvidRetriever] = {}

# Configuration
MEMORY_PATH = os.getenv("MEMORY_PATH", "./memories")
Path(MEMORY_PATH).mkdir(exist_ok=True)


def get_memory_paths(user_id: str) -> tuple[str, str]:
    """Get video and index paths for a user"""
    return (
        f"{MEMORY_PATH}/{user_id}.mp4",
        f"{MEMORY_PATH}/{user_id}_index.json"
    )


def get_or_create_encoder(user_id: str) -> MemvidEncoder:
    """Get or create encoder for user"""
    if user_id not in encoders:
        encoders[user_id] = MemvidEncoder(
            chunk_size=512,
            overlap=50,
            model_name='all-MiniLM-L6-v2'
        )
    return encoders[user_id]


def get_or_create_retriever(user_id: str) -> Optional[MemvidRetriever]:
    """Get or create retriever for user"""
    if user_id not in retrievers:
        video_path, index_path = get_memory_paths(user_id)
        try:
            retrievers[user_id] = MemvidRetriever(video_path, index_path)
        except:
            return None
    return retrievers[user_id]


def load_memory_metadata(user_id: str) -> Dict:
    """Load metadata for user memories"""
    try:
        metadata_path = f"{MEMORY_PATH}/{user_id}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
    except:
        pass
    return {"memories": []}


def save_memory_metadata(user_id: str, metadata: Dict) -> None:
    """Save metadata for user memories"""
    metadata_path = f"{MEMORY_PATH}/{user_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


@mcp.tool()
async def store_memory(
    content: str,
    user_id: str,
    memory_type: str = "general",
    metadata: Optional[Dict] = None,
    ctx: Context = None
) -> Dict:
    """Store a new memory for the user"""
    try:
        if ctx:
            ctx.info(f"Storing memory for user {user_id}")
        
        # Get or create encoder
        encoder = get_or_create_encoder(user_id)
        
        # Prepare memory with metadata
        timestamp = datetime.utcnow().isoformat() + "Z"
        memory_entry = {
            "content": content,
            "timestamp": timestamp,
            "memory_type": memory_type,
            "metadata": metadata or {}
        }
        
        # Load existing metadata
        user_metadata = load_memory_metadata(user_id)
        user_metadata["memories"].append(memory_entry)
        
        # Encode all memories to video
        all_content = [m["content"] for m in user_metadata["memories"]]
        video_path, index_path = get_memory_paths(user_id)
        
        encoder.fit(all_content)
        encoder.encode_to_video(video_path)
        encoder.create_index(index_path)
        
        # Save updated metadata
        save_memory_metadata(user_id, user_metadata)
        
        # Update retriever
        retrievers[user_id] = MemvidRetriever(video_path, index_path)
        
        return {
            "success": True,
            "memory_id": len(user_metadata["memories"]) - 1,
            "timestamp": timestamp
        }
    except Exception as e:
        if ctx:
            ctx.error(f"Failed to store memory: {str(e)}")
        return {"success": False, "error": str(e)}


@mcp.tool()
async def search_memories(
    query: str,
    user_id: str,
    top_k: int = 5,
    memory_type: Optional[str] = None,
    ctx: Context = None
) -> List[Dict]:
    """Search user memories using semantic search"""
    try:
        if ctx:
            ctx.info(f"Searching memories for user {user_id}")
        
        # Get retriever
        retriever = get_or_create_retriever(user_id)
        if not retriever:
            return {"memories": []}
        
        # Search memories
        results = retriever.search(query, top_k=top_k * 2)  # Get extra to filter
        
        # Load metadata to match results with full memory info
        user_metadata = load_memory_metadata(user_id)
        memories = []
        
        for idx, score in results:
            if idx < len(user_metadata["memories"]):
                memory = user_metadata["memories"][idx]
                
                # Filter by memory type if specified
                if memory_type and memory.get("memory_type") != memory_type:
                    continue
                
                memories.append({
                    "content": memory["content"],
                    "relevance_score": float(score),
                    "timestamp": memory["timestamp"],
                    "memory_type": memory.get("memory_type", "general"),
                    "metadata": memory.get("metadata", {})
                })
                
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
    user_id: str,
    session_id: str,
    metadata: Optional[Dict] = None,
    ctx: Context = None
) -> Dict:
    """Store a conversation in video memory"""
    try:
        if ctx:
            ctx.info(f"Storing conversation for user {user_id}, session {session_id}")
        
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
            user_id=user_id,
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
    user_id: str,
    max_tokens: int = 2000,
    query: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Get contextual memory summary for user"""
    try:
        if ctx:
            ctx.info(f"Getting memory context for user {user_id}")
        
        if query:
            # Search for relevant memories
            results = await search_memories(
                query=query,
                user_id=user_id,
                top_k=10,
                ctx=ctx
            )
            memories = results.get("memories", [])
        else:
            # Get recent memories
            user_metadata = load_memory_metadata(user_id)
            memories = [
                {
                    "content": m["content"],
                    "timestamp": m["timestamp"],
                    "memory_type": m.get("memory_type", "general")
                }
                for m in user_metadata["memories"][-10:]
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
    user_id: str,
    strategy: str = "deduplicate",
    ctx: Context = None
) -> Dict:
    """Consolidate and optimize user memories"""
    try:
        if ctx:
            ctx.info(f"Consolidating memories for user {user_id} with strategy {strategy}")
        
        user_metadata = load_memory_metadata(user_id)
        original_count = len(user_metadata["memories"])
        
        if strategy == "deduplicate":
            # Remove duplicate memories based on content similarity
            seen_content = set()
            unique_memories = []
            
            for memory in user_metadata["memories"]:
                # Simple deduplication based on exact content match
                content_key = memory["content"].strip().lower()
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_memories.append(memory)
            
            user_metadata["memories"] = unique_memories
        
        elif strategy == "compress":
            # Group similar memories together
            # This is a simplified version - could use embeddings for better grouping
            grouped_memories = {}
            
            for memory in user_metadata["memories"]:
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
            
            user_metadata["memories"] = compressed_memories
        
        # Re-encode memories
        if user_metadata["memories"]:
            encoder = get_or_create_encoder(user_id)
            all_content = [m["content"] for m in user_metadata["memories"]]
            video_path, index_path = get_memory_paths(user_id)
            
            encoder.fit(all_content)
            encoder.encode_to_video(video_path)
            encoder.create_index(index_path)
            
            # Update retriever
            retrievers[user_id] = MemvidRetriever(video_path, index_path)
        
        # Save updated metadata
        save_memory_metadata(user_id, user_metadata)
        
        final_count = len(user_metadata["memories"])
        
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


@mcp.resource("memory://stats/{user_id}")
async def get_memory_stats(user_id: str) -> Dict:
    """Get memory statistics for a user"""
    try:
        video_path, index_path = get_memory_paths(user_id)
        user_metadata = load_memory_metadata(user_id)
        
        stats = {
            "user_id": user_id,
            "chunk_count": len(user_metadata["memories"]),
            "video_exists": os.path.exists(video_path),
            "index_exists": os.path.exists(index_path)
        }
        
        if os.path.exists(video_path):
            stats["video_size_mb"] = os.path.getsize(video_path) / (1024 * 1024)
            
            # Calculate compression ratio
            total_text_size = sum(len(m["content"]) for m in user_metadata["memories"])
            if total_text_size > 0:
                stats["compression_ratio"] = total_text_size / os.path.getsize(video_path)
            else:
                stats["compression_ratio"] = 0
        
        # Memory type breakdown
        memory_types = {}
        for memory in user_metadata["memories"]:
            mem_type = memory.get("memory_type", "general")
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
        stats["memory_types"] = memory_types
        
        return stats
    except Exception as e:
        return {
            "user_id": user_id,
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