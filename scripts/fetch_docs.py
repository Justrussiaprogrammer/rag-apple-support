import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.config import settings


URLS_PATH = ROOT / "data" / "source_urls.json"
OUTPUT_PATH = settings.raw_docs_path


def make_doc_id(url):
    parsed = urlparse(url)
    slug = parsed.path.strip("/").replace("/", "_").replace("-", "_")
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    return re.sub(r"[^a-zA-Z0-9_]+", "_", f"{slug}_{digest}").strip("_")


def clean_text(value):
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def extract_document(html, url):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()

    title_tag = soup.find("h1")
    if title_tag:
        title = clean_text(title_tag.get_text(" "))
    elif soup.title:
        title = clean_text(soup.title.get_text(" ")).replace(" - Служба поддержки Apple (RU)", "")
    else:
        title = url

    main = soup.find("main") or soup.find("article") or soup.body or soup

    lines = list()
    seen = set()

    for tag in main.find_all(["h1", "h2", "h3", "p", "li"]):
        text = clean_text(tag.get_text(" "))
        if not text or text in seen:
            continue

        seen.add(text)

        if tag.name == "h1":
            lines.append(f"# {text}")
        elif tag.name == "h2":
            lines.append(f"\n## {text}")
        elif tag.name == "h3":
            lines.append(f"\n### {text}")
        elif tag.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    content = "\n\n".join(lines)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()

    return {
        "doc_id": make_doc_id(url),
        "title": title,
        "source_url": url,
        "text": content,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_url(url: str) -> dict:
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0 RAG test assignment crawler; educational prototype"
        },
    )
    response.raise_for_status()
    doc = extract_document(response.text, url)

    if len(doc["text"]) < 800:
        raise ValueError(f"Too little extracted text: {len(doc['text'])} chars")

    return doc


def main() -> None:
    urls = json.loads(URLS_PATH.read_text(encoding="utf-8"))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    docs = []
    errors = []

    for idx, item in enumerate(urls, start=1):
        url = item["url"]
        print(f"[{idx}/{len(urls)}] Fetching {url}")

        try:
            doc = fetch_url(url)
            docs.append(doc)
            print(f"  OK: {doc['title']} ({len(doc['text'])} chars)")
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})
            print(f"  ERROR: {exc}")

        time.sleep(0.8)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    errors_path = OUTPUT_PATH.with_suffix(".errors.json")
    errors_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"Saved docs: {len(docs)} -> {OUTPUT_PATH}")
    print(f"Errors: {len(errors)} -> {errors_path}")

    if len(docs) < 15:
        raise SystemExit("Скачано меньше 15 документов. Добавьте/замените URL в data/source_urls.json.")


if __name__ == "__main__":
    main()
