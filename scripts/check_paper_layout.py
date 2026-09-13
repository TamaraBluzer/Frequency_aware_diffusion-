from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--max-body-pages", type=int)
    args = parser.parse_args()
    document = pymupdf.open(args.pdf)
    if args.render_dir:
        args.render_dir.mkdir(parents=True, exist_ok=True)
    reference_page = None
    body_pages = None
    errors = []
    for number, page in enumerate(document, 1):
        text = page.get_text(sort=True)
        if "??" in text:
            errors.append(f"Page {number}: unresolved reference")
        spans = [span for block in page.get_text("dict")["blocks"] if "lines" in block
                 for line in block["lines"] for span in line["spans"] if span["text"].strip()]
        for index, span in enumerate(spans):
            if span["text"].strip() == "References" and reference_page is None:
                reference_page = number
                body_pages = number if index else number - 1
        outside = [s["text"] for s in spans if not (page.rect + (-1, -1, 1, 1)).contains(pymupdf.Rect(s["bbox"]))]
        if outside:
            errors.append(f"Page {number}: text outside page: {outside}")
        print(f"Page {number}: {len(text.split())} words; {page.rect.width:.1f} x {page.rect.height:.1f} pt")
        if args.extract:
            print(text)
        if args.render_dir:
            page.get_pixmap(dpi=130).save(args.render_dir / f"{args.pdf.stem}-{number}.png")
    print(f"Pages: {len(document)}; body pages: {body_pages}; references start: {reference_page or 'not found'}")
    if args.max_body_pages and (body_pages is None or body_pages > args.max_body_pages):
        errors.append(f"Body exceeds {args.max_body_pages} pages or References heading is missing")
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
