import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.rag.work_classes import Retriever


QUESTIONS_PATH = ROOT / "tests" / "questions.json"
RESULTS_PATH = ROOT / "tests" / "results.json"


def is_hit(result, expected_doc_ids, expected_urls):
    chunk = result.chunk
    if expected_doc_ids and chunk.doc_id in expected_doc_ids:
        return True
    if expected_urls and chunk.source_url in expected_urls:
        return True
    return False


def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    retriever = Retriever()

    rows = list()
    hits = 0

    for item in questions:
        question = item["question"]
        expected_doc_ids = item.get("expected_doc_ids", [])
        expected_urls = item.get("expected_source_urls", [])

        results = retriever.retrieve(question, top_k=3)
        hit = any(is_hit(result, expected_doc_ids, expected_urls) for result in results)
        hits += int(hit)

        rows.append(
            {
                "question": question,
                "hit@3": hit,
                "expected_source_urls": expected_urls,
                "retrieved": [
                    {
                        "doc_title": result.chunk.doc_title,
                        "source_url": result.chunk.source_url,
                        "score": round(result.score, 4),
                    }
                    for result in results
                ],
            }
        )

    recall_at_3 = hits / len(questions) if questions else 0.0
    report = {
        "metric": "Recall@3",
        "value": round(recall_at_3, 4),
        "total_questions": len(questions),
        "hits": hits,
        "rows": rows,
    }

    RESULTS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Recall@3: {recall_at_3:.3f} ({hits}/{len(questions)})")
    print(f"Saved report: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
