from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from src.models.schemas import QueryRequest, QueryResponse
from src.rag.pipeline import RagPipeline
import gradio as gr
from ui.app import demo


app = FastAPI()
rag = RagPipeline()

# ask
@app.post("/query", response_model = QueryResponse)
async def query(request: QueryRequest):
    result = await rag.run(request.question)
    return QueryResponse(
        answer=result["answer"],
        source=result["source"]
    )


# Mount Gradio
app = gr.mount_gradio_app(app, demo, path="/ui")
