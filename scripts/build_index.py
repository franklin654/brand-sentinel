"""One-shot script: embed and index the incident corpus into .chroma/

Reads every JSON file from brand_risk/incident_corpus/, creates one
LangChain Document per incident (narrative as page_content, all other fields
as metadata), and indexes them into the persistent ChromaDB collection.

Run once before starting the app, or whenever the corpus changes:
    python scripts/build_index.py

No jq dependency — documents are built directly from json.load().
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Resolve project root (scripts/ is one level below)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CORPUS_DIR = PROJECT_ROOT / "brand_risk" / "incident_corpus"


def main() -> None:
    from langchain_core.documents import Document
    from brand_risk.rag import index_documents

    if not CORPUS_DIR.exists():
        logger.error("Corpus directory not found: %s", CORPUS_DIR)
        sys.exit(1)

    json_files = sorted(CORPUS_DIR.glob("*.json"))
    if not json_files:
        logger.error("No JSON files found in %s", CORPUS_DIR)
        sys.exit(1)

    documents: list[Document] = []
    for path in json_files:
        try:
            incident = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s — %s", path.name, exc)
            continue

        narrative = incident.get("narrative", "")
        if not narrative:
            logger.warning("Skipping %s — missing 'narrative' field", path.name)
            continue

        metadata = {k: v for k, v in incident.items() if k != "narrative"}
        documents.append(Document(page_content=narrative, metadata=metadata))

    logger.info("Loaded %d incidents from %s", len(documents), CORPUS_DIR)
    index_documents(documents)
    logger.info("Done — indexed %d incidents into .chroma/", len(documents))


if __name__ == "__main__":
    main()
