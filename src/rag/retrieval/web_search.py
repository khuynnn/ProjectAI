from dotenv import load_dotenv
import os
load_dotenv()

from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str, k: int = 3):
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=k
    )

    results = response.get("results", [])

    context = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "")
        
        if len(content) > 50:
            context.append(
                f"Tiêu đề: {title}\n"
                f"URL: {url}\n"
                f"Nội dung: {content}"
            )

    return "\n\n".join(context)