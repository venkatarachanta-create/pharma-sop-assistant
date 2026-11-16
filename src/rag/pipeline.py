import time
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from src.config.settings import settings
from src.guardrails.input_rail import check_input_guardrail
from src.guardrails.output_rail import enforce_output_guardrail
from src.rag.vectorstore import load_vectorstore
from src.rag.prompts import get_qa_prompt
from src.utils.logger import log_audit_event

SUMMARY_KEYWORDS = (
    "summary", "summarize", "outline", "outlines",
    "highlights", "heighlights", "overview", "brief", "about the doc"
)


def _get_llm():
    if settings.llm_provider.lower() == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.temperature,
            api_key=SecretStr(settings.gemini_api_key),
        )
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.temperature,
        api_key=SecretStr(settings.openai_api_key),
    )


def _route_retrieval(vectorstore, query: str):
    query_lower = query.lower()
    is_summary = any(kw in query_lower for kw in SUMMARY_KEYWORDS)

    if is_summary:
        all_docs = [
            d for d in vectorstore.docstore._dict.values()
            if len(d.page_content.strip()) > 100
        ]
        target_doc = None

        for pdf_path in settings.raw_pdf_dir.glob("*.pdf"):
            if pdf_path.name.lower() in query_lower or pdf_path.stem.lower() in query_lower:
                target_doc = pdf_path.name
                break

        if target_doc:
            matched = [d for d in all_docs if d.metadata.get("doc_name") == target_doc]
            matched.sort(key=lambda d: d.metadata.get("page_num", 99))
            return matched[: settings.top_k_results], True

        by_doc = {}
        for d in sorted(all_docs, key=lambda x: x.metadata.get("page_num", 99)):
            name = d.metadata.get("doc_name", "Unknown")
            if len(by_doc.get(name, [])) < 2:
                by_doc.setdefault(name, []).append(d)
        combined = [chunk for chunks in by_doc.values() for chunk in chunks]
        return combined[: settings.top_k_results], True

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.top_k_results, "fetch_k": 20},
    )
    return retriever.invoke(query), False


def run_pharma_rag(query: str) -> dict:
    start = time.perf_counter()

    allowed, msg_or_clean_query = check_input_guardrail(query)
    if not allowed:
        latency = (time.perf_counter() - start) * 1000
        log_audit_event(query, "BLOCKED_INPUT_RAIL", msg_or_clean_query, [], latency)
        return {"answer": msg_or_clean_query, "sources": [], "contexts": [], "blocked": True}

    vectorstore = load_vectorstore()
    docs, is_summary = _route_retrieval(vectorstore, msg_or_clean_query)

    sources = [
        f"[Source: {d.metadata.get('doc_name', 'Unknown')}, Page {d.metadata.get('page_num', '?')}]"
        for d in docs
    ]
    contexts = [d.page_content for d in docs]
    formatted_context = "\n\n---\n\n".join(
        f"{src}\n{doc.page_content}" for src, doc in zip(sources, docs)
    )

    chain = get_qa_prompt(is_summary=is_summary) | _get_llm()
    raw_response = chain.invoke({"context": formatted_context, "question": msg_or_clean_query})
    final_answer = enforce_output_guardrail(str(raw_response.content), sources)

    latency = (time.perf_counter() - start) * 1000
    log_audit_event(msg_or_clean_query, "SUCCESS", final_answer, list(dict.fromkeys(sources)), latency)

    return {
        "answer": final_answer,
        "sources": list(dict.fromkeys(sources)),
        "contexts": contexts,
        "blocked": False,
    }
