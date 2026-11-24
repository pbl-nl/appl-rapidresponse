"""Step 4: turn every fact into an embedding and save them with their metadata.

Only the fact text is embedded. The vectors go to vectors.npy and the metadata to
facts_meta.jsonl, in the same order. Results go to outputs/04_embedding/.
OPENAI_API_KEY and OPENAI_EMBEDDINGS_MODEL come from .env.

Usage:
    python src/04_embedding.py --dry-run                 # no API calls, count facts and calls
    python src/04_embedding.py --limit 20                # test on 20 facts, saved in a test folder
    python src/04_embedding.py --search "text" --test    # search the test folder
    python src/04_embedding.py                           # all facts (asks first)
    python src/04_embedding.py --search "Hoeveel sloopmateriaal komt terug in de woningbouw?"
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

FACTS_FILE = config.STEP3_DIR / "facts_checked.jsonl"  # made by step 3b
TEST_DIR = config.STEP4_DIR / "test"  # a run with --limit goes here, so the full store stays as it is
VECTORS_NAME = "vectors.npy"
META_NAME = "facts_meta.jsonl"
INFO_NAME = "info.json"

# Fields kept next to each vector
META_FIELDS = ["fact_id", "fact", "number", "year", "evidence", "source", "source_file",
               "page", "passage_id", "flagged", "flag_reasons", "needs_chart_review"]


def load_facts():
    with open(FACTS_FILE, encoding="utf-8") as f:
        facts = [json.loads(line) for line in f]
    # Empty facts give no useful vector
    return [r for r in facts if r["fact"].strip()]


def embed(client, texts):
    """Embeddings for a list of texts, in the same order. Tries again after an API error."""
    import openai
    for attempt in range(1, config.EXTRACTION_RETRIES + 1):
        try:
            response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
            return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as error:
            wait = 2 ** attempt
            print(f"    {type(error).__name__}, try {attempt} of {config.EXTRACTION_RETRIES}, waiting {wait} s")
            time.sleep(wait)
    raise RuntimeError(f"no answer after {config.EXTRACTION_RETRIES} tries")


def normalise_rows(vectors):
    """Scale every vector to length 1, so a dot product is the cosine similarity."""
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(lengths == 0, 1, lengths)


def make_client():
    if not config.EMBEDDING_MODEL:
        sys.exit("Set OPENAI_EMBEDDINGS_MODEL in .env first.")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY in .env first.")
    from openai import OpenAI
    return OpenAI()  # reads OPENAI_API_KEY from the environment


def build(facts, yes, out_dir):
    batches = [facts[i:i + config.EMBEDDING_BATCH] for i in range(0, len(facts), config.EMBEDDING_BATCH)]
    print(f"{len(facts)} facts, {len(batches)} API calls, model {config.EMBEDDING_MODEL}")
    if len(batches) > 20 and not yes:
        answer = input(f"This makes {len(batches)} API calls. Go on? (yes/no) ")
        if answer.strip().lower() != "yes":
            return

    client = make_client()
    vectors = []
    for i, batch in enumerate(batches, start=1):
        vectors += embed(client, [r["fact"] for r in batch])
        print(f"[{i}/{len(batches)}] {len(vectors)} facts done")
    vectors = normalise_rows(np.array(vectors, dtype=np.float32))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / VECTORS_NAME, vectors)
    with open(out_dir / META_NAME, "w", encoding="utf-8") as f:
        for r in facts:
            f.write(json.dumps({k: r.get(k) for k in META_FIELDS}, ensure_ascii=False) + "\n")
    info = {"model": config.EMBEDDING_MODEL, "facts": len(facts), "dimensions": int(vectors.shape[1]),
            "facts_file": str(FACTS_FILE.relative_to(config.PROJECT_DIR))}
    (out_dir / INFO_NAME).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    print(f"\nvectors -> {out_dir / VECTORS_NAME} ({vectors.shape[0]} x {vectors.shape[1]})")
    print(f"metadata -> {out_dir / META_NAME}")


def load_store(store_dir):
    """The saved vectors and metadata. Checks that they belong together."""
    vectors = np.load(store_dir / VECTORS_NAME)
    with open(store_dir / META_NAME, encoding="utf-8") as f:
        meta = [json.loads(line) for line in f]
    info = json.loads((store_dir / INFO_NAME).read_text(encoding="utf-8"))
    if len(meta) != len(vectors):
        sys.exit("vectors.npy and facts_meta.jsonl do not match. Run step 4 again.")
    if info["model"] != config.EMBEDDING_MODEL:
        sys.exit(f"The vectors were made with {info['model']}, but .env says {config.EMBEDDING_MODEL}. "
                 f"Run step 4 again.")
    return vectors, meta


def search(query, top, store_dir):
    """Print the facts closest in meaning to a query."""
    vectors, meta = load_store(store_dir)
    query_vector = normalise_rows(np.array(embed(make_client(), [query]), dtype=np.float32))[0]
    scores = vectors @ query_vector
    for i in np.argsort(-scores)[:top]:
        r = meta[i]
        print(f"{scores[i]:.3f}  {r['fact_id']}  {r['source']} p{r['page']}\n       {r['fact']}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="only the first facts, for a test")
    parser.add_argument("--dry-run", action="store_true", help="no API calls, only count the facts and calls")
    parser.add_argument("--yes", action="store_true", help="do not ask before a run of more than 20 calls")
    parser.add_argument("--search", help="show the facts closest to this text")
    parser.add_argument("--top", type=int, default=5, help="number of facts shown by --search")
    parser.add_argument("--test", action="store_true", help="search the test folder")
    args = parser.parse_args()

    if args.search:
        search(args.search, args.top, TEST_DIR if args.test else config.STEP4_DIR)
        return

    facts = load_facts()
    if args.limit:
        facts = facts[:args.limit]
    if args.dry_run:
        batches = -(-len(facts) // config.EMBEDDING_BATCH)
        chars = sum(len(r["fact"]) for r in facts)
        print(f"{len(facts)} facts, {batches} API calls")
        print(f"input: about {chars / 4 / 1e6:.2f} million tokens (estimate: 4 characters per token)")
        return
    build(facts, args.yes, TEST_DIR if args.limit else config.STEP4_DIR)


if __name__ == "__main__":
    main()
