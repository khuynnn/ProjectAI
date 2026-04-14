# ui/app.py
import gradio as gr
import asyncio
from src.rag.pipeline import RagPipeline
import re

rag = RagPipeline()


def fix_latex(text: str) -> str:
    # \(...\) → $...$
    text = re.sub(r'\\\((.+?)\\\)', r'$\1$', text, flags=re.DOTALL)
    # \[...\] → $$...$$
    text = re.sub(r'\\\[(.+?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    return text

def run_rag(question: str) -> str:
    loop = asyncio.new_event_loop()
    try:
        result =  loop.run_until_complete(rag.run(question))
        return fix_latex(result)
    finally:
        loop.close()

custom_css = """
.message-bubble-border {
    max-width: 90% !important;
    width: 90% !important;
}

/* Text thường */
.message p {
    line-height: 1.8;
    font-size: 15px;
}

/* Công thức block căn giữa */
.message .mjx-container[display="true"] {
    margin: 16px auto !important;
    display: block !important;
    text-align: center !important;
}

/* Công thức inline thẳng hàng với text */
.message .mjx-container[display="false"] {
    display: inline !important;
    margin: 0 2px !important;
}

/* Khoảng cách giữa các đoạn */
.message p + p {
    margin-top: 12px;
}
"""

with gr.Blocks(title="RAG Chatbot") as demo:
    gr.Markdown("## 💬 RAG Chatbot")

    chatbot = gr.Chatbot(
        latex_delimiters=[
            {"left": "$$", "right": "$$", "display": True},
            {"left": "$",  "right": "$",  "display": False},
        ],
        render_markdown=True,
        height=600,
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Nhập câu hỏi...",
            scale=9,
            container=False
        )
        send = gr.Button("Gửi", scale=1)

    def respond(question, history):
        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "⏳ Đang xử lý..."})
        yield "", history

        answer = run_rag(question)

        history[-1] = {"role": "assistant", "content": answer}
        yield "", history
        

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send.click(respond, [msg, chatbot], [msg, chatbot])