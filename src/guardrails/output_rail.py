REFUSAL_PHRASE = "I cannot find this information in the uploaded regulatory documents."


def enforce_output_guardrail(answer: str, retrieved_sources: list[str]) -> str:
    if REFUSAL_PHRASE.lower() in answer.lower():
        return REFUSAL_PHRASE

    if retrieved_sources and "[Source:" not in answer:
        unique_sources = ", ".join(dict.fromkeys(retrieved_sources))
        answer = f"{answer.strip()}\n\n**Verified Sources:** {unique_sources}"

    return answer
