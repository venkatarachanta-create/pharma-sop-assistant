import json
import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Workaround for ragas 0.4.x importing removed vertexai path in langchain-community 0.4+
mock_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
mock_vertexai.ChatVertexAI = object
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertexai

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config.settings import settings
from src.rag.pipeline import run_pharma_rag


def get_judge_models():
    if settings.llm_provider.lower() == "gemini":
        llm = ChatGoogleGenerativeAI(model=settings.llm_model, google_api_key=settings.gemini_api_key)
        emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=settings.gemini_api_key)
        return llm, emb
    llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)
    emb = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    return llm, emb


def main():
    golden_path = Path("data/eval_datasets/cdsco_golden_qa.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_items = json.load(f)

    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    print(f"🧪 Running Pharma RAG pipeline on {len(golden_items)} golden test cases...")
    for item in golden_items:
        q = item["question"]
        result = run_pharma_rag(q)
        records["question"].append(q)
        records["answer"].append(result["answer"])
        records["contexts"].append(result["contexts"])
        records["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(records)
    judge_llm, judge_emb = get_judge_models()

    print("📊 Computing RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision)...")
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_emb,
    )

    print("\n=== PHARMA RAGAS BENCHMARK RESULTS ===")
    print(scores)

    report_path = Path("data/logs/ragas_benchmark_report.csv")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_pandas().to_csv(report_path, index=False)
    print(f"✅ Detailed report saved to {report_path}")


if __name__ == "__main__":
    main()
