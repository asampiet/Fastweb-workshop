"""T-303: FastAPI backend serving the Strands agent."""
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agent import TelcoRCAAgent
from config import API_PORT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("APIServer")

app = FastAPI(title="Telco RCA Agent API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

agent = TelcoRCAAgent()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    response: str
    metrics: dict | None = None
    error: str | None = None


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = agent.process_query(req.message)
        return ChatResponse(response=result["response"], metrics=result["metrics"])
    except ConnectionError:
        raise HTTPException(503, detail="MCP_CONNECTION_FAILED")
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(502, detail="BEDROCK_API_ERROR")


@app.get("/api/health")
async def health():
    return {"status": "healthy", "bedrock_model": agent.model.model_id}


if __name__ == "__main__":
    logger.info(f"Starting API server on port {API_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
