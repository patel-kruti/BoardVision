"""
Simple startup script for BoardVision API
Starts the server WITHOUT Qwen model (faster startup)
"""

print("=" * 70)
print("BoardVision API Server")
print("=" * 70)
print("\nStarting FastAPI server...")
print("Note: Qwen-VL model will load lazily when first requested")
print("\nServer will be available at: http://127.0.0.1:8000")
print("API docs at: http://127.0.0.1:8000/docs")
print("\nPress Ctrl+C to stop the server")
print("=" * 70)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "local_mcp_new:app",
        host="127.0.0.1",
        port=8000,
        reload=False  # Set to True for development
    )


