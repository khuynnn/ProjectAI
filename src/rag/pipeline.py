from src.rag.ingestion.ingest_pipeline import IngestionPipeline
from src.rag.retrieval.retriever import Retriever
from src.rag.generation.generator import LMClient


class RagPipeline:
    def __init__(self):
        self.generator = LMClient()

    async def run(self, query: str, thread_id: str = "default") -> str:
        return await self.generator.ask(query, thread_id=thread_id)



