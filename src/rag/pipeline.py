import time
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import settings
from src.guardrails.input_rail import check_input_guardrail
from src.guardrails.output_rail import enforce_output_guardrail
from src.rag.vectorstore import load_vectorstore
from src.rag.prompts import get_qa_prompt
from src.utils.logger import log_audit_event


def _get_llm():
    if settings.llm_provider.lower() == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.temperature,
            google_api_key=settings.gemini_api_key,
        )
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )


def run_pharma_rag(query: str) -> dict:
    start = time.perf_counter()

    allowed, msg_or_clean_query = check_input_guardrail(query)
    if not allowed:
        latency = (time.perf_counter() - start) * 1000
        log_audit_event(query, "BLOCKED_INPUT_RAIL", msg_or_clean_query, [], latency)
        return {"answer": msg_or_clean_query, "sources": [], "contexts": [], "blocked": True}

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k_results})
    docs = retriever.invoke(msg_or_clean_query)

    sources = [
        f"[Source: {d.metadata.get('doc_name', 'Unknown')}, Page {d.metadata.get('page_num', '?')}]"
        for d in docs
    ]
    contexts = [d.page_content for d in docs]
    formatted_context = "\n\n---\n\n".join(
        f"{src}\n{doc.page_content}" for src, doc in zip(sources, docs)
    )

    chain = get_qa_prompt() | _get_llm()
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
