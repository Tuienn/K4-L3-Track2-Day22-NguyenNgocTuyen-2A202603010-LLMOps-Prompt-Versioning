"""
Bước 1 — RAG Pipeline với LangSmith Tracing
=============================================
NHIỆM VỤ:
  1. Tải knowledge base, chia chunks, index với FAISS
  2. Xây dựng RAG chain: retriever → prompt → LLM → output parser
  3. Trang trí hàm query với @traceable để LangSmith ghi lại mỗi lần gọi
  4. Chạy 50 câu hỏi → tạo ≥ 50 traces trên LangSmith

DELIVERABLE: Mở https://smith.langchain.com → project của bạn → xác nhận ≥ 50 traces.
"""
import sys
import os
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ⚠️ QUAN TRỌNG: Import config TRƯỚC KHI import bất kỳ thư viện LangChain nào.
# config.py tự động đặt LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, ... vào os.environ
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Thiết lập Vectorstore ───────────────────────────────────────────────
def setup_vectorstore():
    """
    Tải knowledge base, chia chunks và tạo FAISS vectorstore.

    Gợi ý:
        embeddings  = get_embeddings()
        text        = load_knowledge_base()
        chunks      = split_text(text, chunk_size=500, chunk_overlap=50)
        vectorstore = build_vectorstore(chunks, embeddings)
    """
    embeddings = get_embeddings()

    text = load_knowledge_base()

    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"📚 Đã chia thành {len(chunks)} chunks")

    vectorstore = build_vectorstore(chunks, embeddings)
    return vectorstore


# ── 2. RAG Prompt Template ─────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Answer in the language of the question using only the context. "
     "If the context is insufficient, say so. Do not add unsupported facts.\n\nContext:\n{context}"),
    ("human", "{question}"),
])


# ── 3. Build RAG Chain ─────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Xây dựng LCEL RAG chain theo cấu trúc pipe:
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()

    Trả về: (chain, retriever)
    """
    llm = get_llm()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT | llm | StrOutputParser()
    )

    return chain, retriever


# ── 4. Hàm Query có LangSmith Tracing ─────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Chạy RAG chain với một câu hỏi.
    Decorator @traceable sẽ gửi mỗi lần gọi lên LangSmith như một trace riêng.
    """
    return chain.invoke(question)


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    chain, retriever = build_rag_chain(vectorstore)

    evidence = Path(__file__).resolve().parents[1] / "evidence"
    progress_path = evidence / "01_answers.json"
    answers = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    pending = [q for q in SAMPLE_QUESTIONS if q not in answers]
    workers = int(os.getenv("LAB_WORKERS", "5"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for question, answer in zip(pending, pool.map(lambda q: ask(chain, q), pending)):
            answers[question] = answer
            progress_path.write_text(json.dumps(answers, indent=2), encoding="utf-8")
            print(f"Completed {len(answers)}/50: {question}", flush=True)
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question}")
        print(f"       A: {answers[question]}\n")

    from langchain_core.tracers.langchain import wait_for_all_tracers
    wait_for_all_tracers()
    print(f"\n✅ {len(SAMPLE_QUESTIONS)} câu hỏi đã chạy; kiểm tra traces trong project '{config.LANGSMITH_PROJECT}'")
    print("   Mở https://smith.langchain.com để xem traces.")


if __name__ == "__main__":
    main()
