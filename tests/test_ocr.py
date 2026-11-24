"""Tests for the OCR part of step 2 (src/02_text_processing.py).

The unit tests replace PaddleOCR with a fake model that returns fixed boxes, so they are
fast and test our own code: joining boxes into lines, lines into paragraphs, reading
order, and pixel to page coordinates. The last test runs the real PaddleOCR model on a
generated page and is skipped when PaddleOCR is not installed.

Usage:
    python -m unittest tests.test_ocr -v
"""
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

# The file name starts with a number, so it cannot be imported with a normal import
_spec = importlib.util.spec_from_file_location("text_processing", ROOT / "src" / "02_text_processing.py")
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


class FakeModel:
    """Stands in for PaddleOCR. Returns the given (text, x0, y0, x1, y1) boxes."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.images = []

    def predict(self, image):
        self.images.append(image)
        return [{"rec_texts": [b[0] for b in self.boxes], "rec_boxes": [b[1:] for b in self.boxes]}]


def line(text, x0, top, x1, bottom):
    return {"text": text, "x0": x0, "top": top, "x1": x1, "bottom": bottom}


def texts(paragraphs):
    return [" ".join(p["lines"]) for p in paragraphs]


class OcrLinesTest(unittest.TestCase):
    def run_lines(self, boxes):
        with mock.patch.object(tp, "_ocr_model", FakeModel(boxes)):
            return tp.ocr_lines(image=None)

    def test_close_boxes_on_one_row_become_one_line(self):
        # Given right to left, joined left to right
        lines = self.run_lines([("world", 105, 101, 160, 111), ("Hello", 50, 100, 100, 110)])
        self.assertEqual([l["text"] for l in lines], ["Hello world"])
        self.assertEqual((lines[0]["x0"], lines[0]["x1"]), (50, 160))
        self.assertEqual((lines[0]["top"], lines[0]["bottom"]), (100, 111))

    def test_far_apart_boxes_on_one_row_stay_separate(self):
        # Two columns: the space between them (40 px) is more than 1.5 line heights
        lines = self.run_lines([("left", 50, 100, 280, 110), ("right", 320, 100, 550, 110)])
        self.assertEqual(sorted(l["text"] for l in lines), ["left", "right"])

    def test_boxes_on_different_rows_stay_separate(self):
        lines = self.run_lines([("one", 50, 100, 100, 110), ("two", 105, 130, 160, 140)])
        self.assertEqual(len(lines), 2)

    def test_empty_text_is_dropped(self):
        lines = self.run_lines([("  ", 50, 100, 100, 110), ("text", 50, 130, 100, 140)])
        self.assertEqual([l["text"] for l in lines], ["text"])


class OcrParagraphsTest(unittest.TestCase):
    def test_lines_of_running_text_are_joined(self):
        pars = tp.ocr_paragraphs([
            line("first line", 50, 100, 350, 110),
            line("second line", 50, 115, 350, 125),
            line("last line", 50, 130, 200, 140),
        ])
        self.assertEqual(texts(pars), ["first line second line last line"])

    def test_empty_space_starts_a_new_paragraph(self):
        pars = tp.ocr_paragraphs([
            line("paragraph one", 50, 100, 350, 110),
            line("paragraph two", 50, 125, 350, 135),  # 15 px space, more than one line height
        ])
        self.assertEqual(texts(pars), ["paragraph one", "paragraph two"])

    def test_heading_is_not_joined_with_body_text(self):
        pars = tp.ocr_paragraphs([
            line("Big heading", 50, 50, 450, 70),
            line("body text", 50, 80, 350, 90),
        ])
        self.assertEqual(texts(pars), ["Big heading", "body text"])

    def test_short_line_ends_the_paragraph(self):
        pars = tp.ocr_paragraphs([
            line("short end.", 50, 100, 100, 110),  # 50 px wide, less than 12 line heights
            line("new paragraph", 50, 115, 350, 125),
        ])
        self.assertEqual(texts(pars), ["short end.", "new paragraph"])

    def test_two_columns_are_read_one_after_the_other(self):
        lines = []
        for i, top in enumerate((100, 115, 130)):
            lines.append(line(f"L{i}", 50, top, 280, top + 10))
            lines.append(line(f"R{i}", 320, top, 550, top + 10))
        self.assertEqual(texts(tp.ocr_paragraphs(lines)), ["L0 L1 L2", "R0 R1 R2"])

    def test_table_is_read_row_by_row(self):
        lines = []
        for row, top in ((1, 100), (2, 115)):
            lines.append(line(f"A{row}", 50, top, 80, top + 10))
            lines.append(line(f"B{row}", 200, top, 230, top + 10))
        self.assertEqual(texts(tp.ocr_paragraphs(lines)), ["A1", "B1", "A2", "B2"])


class ReadingOrderTest(unittest.TestCase):
    def box(self, name, x0, top, x1, bottom):
        return {"name": name, "x0": x0, "top": top, "x1": x1, "bottom": bottom}

    def order(self, boxes):
        return [b["name"] for b in tp.reading_order(boxes)]

    def test_header_columns_footer(self):
        boxes = [
            self.box("footer", 50, 700, 550, 710),
            self.box("right", 320, 100, 550, 600),
            self.box("header", 50, 20, 550, 30),
            self.box("left", 50, 100, 280, 600),
        ]
        self.assertEqual(self.order(boxes), ["header", "left", "right", "footer"])

    def test_overlapping_boxes_fall_back_to_top_then_left(self):
        boxes = [self.box("b", 100, 10, 300, 50), self.box("a", 50, 0, 200, 40)]
        self.assertEqual(self.order(boxes), ["a", "b"])

    def test_empty_and_single(self):
        self.assertEqual(tp.reading_order([]), [])
        self.assertEqual(self.order([self.box("only", 0, 0, 1, 1)]), ["only"])


class OcrRegionTest(unittest.TestCase):
    def setUp(self):
        self.doc = pymupdf.open()
        self.page = self.doc.new_page(width=600, height=800)

    def tearDown(self):
        self.doc.close()

    def test_blocks_have_page_coordinates_and_joined_text(self):
        fake = FakeModel([
            ("De circulaire eco-", 50, 100, 350, 110),
            ("nomie is belangrijk.", 50, 115, 350, 125),
        ])
        # At 72 dpi one pixel is one PDF point, so the numbers are easy to check
        with mock.patch.object(tp, "_ocr_model", fake), mock.patch.object(config, "OCR_DPI", 72):
            blocks = tp.ocr_region(self.page, self.page.rect)

        self.assertEqual(len(blocks), 1)
        b = blocks[0]
        self.assertEqual(b["text"], "De circulaire economie is belangrijk.")  # hyphen joined
        self.assertAlmostEqual(b["size"], 10)
        self.assertFalse(b["bold"])
        self.assertAlmostEqual(b["top"], 100 / 800)
        self.assertAlmostEqual(b["bottom"], 125 / 800)
        # The model gets a BGR image of the whole page
        self.assertEqual(fake.images[0].shape, (800, 600, 3))

    def test_region_offset_is_added(self):
        fake = FakeModel([("text in the lower half of the page", 0, 0, 300, 10)])
        lower_half = pymupdf.Rect(0, 400, 600, 800)
        with mock.patch.object(tp, "_ocr_model", fake), mock.patch.object(config, "OCR_DPI", 72):
            blocks = tp.ocr_region(self.page, lower_half)
        self.assertAlmostEqual(blocks[0]["top"], 400 / 800)
        self.assertEqual(fake.images[0].shape, (400, 600, 3))

    def test_text_is_unicode_normalized(self):
        fake = FakeModel([("ﬁets", 50, 100, 350, 110)])  # 'fi' ligature
        with mock.patch.object(tp, "_ocr_model", fake), mock.patch.object(config, "OCR_DPI", 72):
            blocks = tp.ocr_region(self.page, self.page.rect)
        self.assertEqual(blocks[0]["text"], "fiets")

    def test_page_without_text_gives_no_blocks(self):
        with mock.patch.object(tp, "_ocr_model", FakeModel([])):
            self.assertEqual(tp.ocr_region(self.page, self.page.rect), [])


@unittest.skipUnless(tp.HAS_OCR, "PaddleOCR is not installed")
class PaddleOcrIntegrationTest(unittest.TestCase):
    """Runs the real model (slow: loads the model, then a few seconds per page)."""

    def test_reads_dutch_text_from_a_page_image(self):
        # Write text on a page, then keep only a picture of it, like a scanned page
        src = pymupdf.open()
        src_page = src.new_page(width=595, height=842)
        src_page.insert_text((72, 100), "Circulaire economie", fontsize=24)
        src_page.insert_text((72, 160), "Het gebruik van grondstoffen moet omlaag.", fontsize=14)
        pix = src_page.get_pixmap(dpi=200)
        doc = pymupdf.open()
        page = doc.new_page(width=595, height=842)
        page.insert_image(page.rect, pixmap=pix)
        self.assertEqual(page.get_text().strip(), "")  # no text layer: only OCR can read it

        blocks = tp.ocr_blocks(page)
        text = " ".join(b["text"] for b in blocks).lower()
        self.assertIn("circulaire", text)
        self.assertIn("grondstoffen", text)
        # The heading comes first and is bigger than the body text
        self.assertIn("circulaire", blocks[0]["text"].lower())
        self.assertGreater(blocks[0]["size"], blocks[-1]["size"])


if __name__ == "__main__":
    unittest.main()
