"""Project settings for the BLO chapter pipeline.

All paths are relative to the project folder. Local settings come from .env.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Folders
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
REPORTS_DIR = PROJECT_DIR / "report"        # BLO 2025 report PDF (read only)
REFERENCES_DIR = PROJECT_DIR / "references"  # source files the chapter cites (read only)
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# One output folder per pipeline step
STEP0_DIR = OUTPUTS_DIR / "00_cited_sentences"
STEP1_DIR = OUTPUTS_DIR / "01_collect_sources"
STEP2_DIR = OUTPUTS_DIR / "02_text_processing"
STEP3_DIR = OUTPUTS_DIR / "03_extraction"
STEP4_DIR = OUTPUTS_DIR / "04_embedding"
STEP8_DIR = OUTPUTS_DIR / "08_evaluation"

REPORT_PDF = REPORTS_DIR / "pbl-2025-balans-van-de-leefomgeving-2025-5719.pdf"

# Chapter pages in the report PDF
CHAPTER_PAGES = (49, 71)          # chapter 3 "Grondstoffen"

# Text processing (step 2)
PASSAGE_MAX_CHARS = 1500  # a passage is closed when the next paragraph would make it longer
PASSAGE_MIN_CHARS = 300   # a shorter passage at the end of a page is added to the one before
# OCR of scanned and garbage pages with PaddleOCR (all models read Dutch and English).
# Time per page on CPU, measured on a two-column scan and a table page:
#   PP-OCRv6_tiny,   200 dpi:  3-5 s   right column order, tables row by row (about 30 min for all OCR pages)
#   PP-OCRv6_small,  200 dpi:  9-15 s  sometimes puts the right column first, can drop spaces
#   PP-OCRv6_medium, 300 dpi: 40-72 s  far too slow (about 6 hours for all OCR pages)
# Tips: start with tiny. If OCR pages in page_report.csv read badly, try small.
# A higher OCR_DPI can help with small print but makes every page slower.
OCR_MODEL = "PP-OCRv6_tiny"
OCR_DPI = 200

# Models
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL")

# Extraction (step 3)
FACT_LANGUAGE = "the language of the passage"  # facts in the same language as the passage
EXTRACTION_RETRIES = 5    # tries per passage when the API gives an error
EXTRACTION_WORKERS = 4    # passages processed concurrently; use 1 for sequential calls

# Embedding (step 4)
EMBEDDING_BATCH = 100     # facts per API call

# Retrieval and evaluation
TOP_K_VALUES = [5, 10, 20, 50]
EXTRACTION_CANDIDATES = 5
