#!/usr/bin/env python3
"""Build Sentiment10k dataset artifacts in an isolated workspace.

Pipeline in one file:
1) download FinanceBench metadata into workspace/
2) download missing PDFs into workspace/pdfs
3) convert PDFs to workspace/pdfs_text
4) compute sentiment labels into workspace/sentiment.json
5) write final examples JSON (metadata + text_path, no embedded context)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

PdfBackend = Literal["auto", "fitz", "pdfplumber", "pypdf"]

DEFAULT_OPEN_SOURCE_URL = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl"
)
DEFAULT_DOC_INFO_URL = (
    "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_document_information.jsonl"
)

DATE_CANDIDATE_PATTERN = re.compile(
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}",
    re.IGNORECASE,
)
DATE_PARSE_FORMATS = ["%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d"]
DATE_MARKERS = ["for the fiscal year ended", "for the quarterly period ended", "date of report"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_workspace_dir() -> Path:
    return _repo_root() / "data" / "sentiment10k"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def ensure_metadata_files(
    data_dir: Path,
    open_source_url: str = DEFAULT_OPEN_SOURCE_URL,
    doc_info_url: str = DEFAULT_DOC_INFO_URL,
    force: bool = False,
    timeout: float = 45.0,
) -> Tuple[Path, Path]:
    import requests  # type: ignore

    data_dir.mkdir(parents=True, exist_ok=True)
    open_source_path = data_dir / "financebench_open_source.jsonl"
    doc_info_path = data_dir / "financebench_document_information.jsonl"

    if force or not open_source_path.exists():
        r = requests.get(open_source_url, timeout=timeout)
        r.raise_for_status()
        open_source_path.write_text(r.text, encoding="utf-8")

    if force or not doc_info_path.exists():
        r = requests.get(doc_info_url, timeout=timeout)
        r.raise_for_status()
        doc_info_path.write_text(r.text, encoding="utf-8")

    return open_source_path, doc_info_path


def _normalize_text(text: str) -> str:
    return text.replace("\x00", " ")


def _parse_date_candidate(raw: str) -> Optional[date]:
    s = " ".join(raw.strip().split())
    for fmt in DATE_PARSE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _extract_date_near_marker(text: str, marker: str, lookahead_chars: int = 100) -> Optional[date]:
    lower = text.lower()
    start = 0
    while True:
        idx = lower.find(marker, start)
        if idx == -1:
            return None
        window = text[idx : idx + len(marker) + lookahead_chars]
        for match in DATE_CANDIDATE_PATTERN.finditer(window):
            parsed = _parse_date_candidate(match.group(0))
            if parsed:
                return parsed
        start = idx + len(marker)


def extract_filing_date(text: str) -> Optional[date]:
    cleaned = _normalize_text(text)
    for marker in DATE_MARKERS:
        dt = _extract_date_near_marker(cleaned, marker)
        if dt:
            return dt
    return None


def extract_pdf_pages(pdf_path: Path, backend: PdfBackend = "auto") -> List[str]:
    def _extract_with_fitz(path: Path) -> List[str]:
        import fitz  # type: ignore

        doc = fitz.open(path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return pages

    def _extract_with_pdfplumber(path: Path) -> List[str]:
        import pdfplumber  # type: ignore

        with pdfplumber.open(path) as pdf:
            return [(page.extract_text() or "") for page in pdf.pages]

    def _extract_with_pypdf(path: Path) -> List[str]:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        return [(page.extract_text() or "") for page in reader.pages]

    extractors = {
        "fitz": _extract_with_fitz,
        "pdfplumber": _extract_with_pdfplumber,
        "pypdf": _extract_with_pypdf,
    }
    order = ["fitz", "pdfplumber", "pypdf"] if backend == "auto" else [backend]
    errors: Dict[str, Exception] = {}
    last_exc: Exception | None = None

    for name in order:
        try:
            return extractors[name](pdf_path)
        except Exception as exc:
            errors[name] = exc
            last_exc = exc

    err_summary = "; ".join(
        f"{name} error: {type(errors[name]).__name__ if name in errors else 'N/A'}"
        for name in ["fitz", "pdfplumber", "pypdf"]
    )
    raise RuntimeError(f"Failed to extract PDF text for {pdf_path}. {err_summary}.") from last_exc


def page_text_to_single_text(pages: Sequence[str]) -> str:
    out: List[str] = []
    for i, page in enumerate(pages, start=1):
        out.append(f"[PAGE {i}]")
        out.append(page)
    return "\n\n".join(out).strip()


def ensure_pdf_text_cache(
    pdf_dir: Path,
    pdf_text_dir: Path,
    overwrite: bool = False,
    pdf_backend: PdfBackend = "auto",
) -> Tuple[Dict[str, Path], List[Path]]:
    from tqdm import tqdm  # type: ignore

    pdf_text_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, Path] = {}
    failed_pdfs: List[Path] = []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    for pdf_path in tqdm(pdf_files, desc="Converting PDFs", unit="pdf"):
        doc_name = pdf_path.stem
        text_path = pdf_text_dir / f"{doc_name}.txt"
        mapping[doc_name] = text_path

        if text_path.exists() and not overwrite:
            continue

        try:
            pages = extract_pdf_pages(pdf_path, backend=pdf_backend)
            text = page_text_to_single_text(pages)
            text_path.write_text(text, encoding="utf-8")
        except Exception:
            failed_pdfs.append(pdf_path)
            continue

    return mapping, failed_pdfs


def _load_doc_company_names(doc_info_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not doc_info_path.exists():
        return mapping
    for row in _load_jsonl(doc_info_path):
        doc_name = str(row.get("doc_name", "")).strip()
        company = str(row.get("company", "")).strip()
        if doc_name and company:
            mapping[doc_name] = company
    return mapping


def _load_doc_links(doc_info_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not doc_info_path.exists():
        return mapping
    for row in _load_jsonl(doc_info_path):
        doc_name = str(row.get("doc_name", "")).strip()
        link = str(row.get("doc_link", "")).strip()
        if doc_name and link:
            mapping[doc_name] = link
    return mapping


def _required_doc_names(open_source_path: Path, only_10k: bool = False) -> Set[str]:
    required: Set[str] = set()
    for row in _load_jsonl(open_source_path):
        doc_name = str(row.get("doc_name", "")).strip()
        if not doc_name:
            continue
        if only_10k and "_10K" not in doc_name.upper():
            continue
        required.add(doc_name)
    return required


def ensure_financebench_pdfs(
    pdf_dir: Path,
    doc_links: Dict[str, str],
    required_doc_names: Iterable[str],
    timeout: float = 45.0,
) -> Tuple[List[str], List[str]]:
    import requests  # type: ignore

    pdf_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[str] = []
    failed: List[str] = []

    for doc_name in sorted(set(required_doc_names)):
        out_path = pdf_dir / f"{doc_name}.pdf"
        if out_path.exists():
            continue
        link = doc_links.get(doc_name)
        if not link:
            failed.append(doc_name)
            continue
        try:
            resp = requests.get(link, timeout=timeout)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            downloaded.append(doc_name)
        except Exception:
            failed.append(doc_name)

    return downloaded, failed


def _clean_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _score_quote_match(company_name: str, quote: Dict[str, Any]) -> float:
    import difflib

    symbol = str(quote.get("symbol", "")).strip().upper()
    quote_type = str(quote.get("quoteType", "")).strip().upper()
    exchange = str(quote.get("exchange", "")).strip().upper()
    short_name = str(quote.get("shortname", "") or quote.get("longname", "")).strip()

    if not symbol:
        return -999.0

    company_clean = _clean_name(company_name)
    name_clean = _clean_name(short_name)

    ratio = difflib.SequenceMatcher(None, company_clean, name_clean).ratio() if name_clean else 0.0
    score = ratio

    if quote_type == "EQUITY":
        score += 1.0
    elif quote_type in {"ETF", "MUTUALFUND"}:
        score -= 0.2
    else:
        score -= 1.2

    if exchange in {"NMS", "NYQ", "ASE", "BTS", "PCX"}:
        score += 0.15
    if "-" in symbol:
        score -= 0.5

    return score


def _pick_best_symbol(company_name: str, quotes: List[Dict[str, Any]]) -> Optional[str]:
    if not quotes:
        return None
    ranked = sorted(quotes, key=lambda q: _score_quote_match(company_name, q), reverse=True)
    best = ranked[0]
    symbol = str(best.get("symbol", "")).strip().upper()
    return symbol or None


def search_yahoo_symbol(company_name: str, timeout: float = 10.0) -> Optional[str]:
    import requests  # type: ignore
    import yfinance as yf  # type: ignore

    try:
        result = yf.Search(company_name)
        quotes = list(result.quotes or [])
        symbol = _pick_best_symbol(company_name, quotes)
        if symbol:
            return symbol
    except Exception:
        pass

    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": company_name, "quotesCount": 10, "newsCount": 0}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None

    quotes = payload.get("quotes") or []
    if not isinstance(quotes, list):
        return None
    return _pick_best_symbol(company_name, quotes)


def validate_ticker_has_prices(ticker: str, around_date: date) -> bool:
    import yfinance as yf  # type: ignore

    start = around_date - timedelta(days=7)
    end = around_date + timedelta(days=30)
    try:
        hist = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
            progress=False,
            actions=False,
            threads=False,
        )
    except Exception:
        return False

    if hist is None or len(hist) == 0:
        return False
    return "Close" in hist.columns and hist["Close"].dropna().shape[0] > 0


def resolve_ticker(company_name: str, around_date: date, cache: Dict[str, Optional[str]]) -> Optional[str]:
    key = company_name.strip().lower()
    if key in cache:
        return cache[key]

    symbol = search_yahoo_symbol(company_name)
    if symbol and validate_ticker_has_prices(symbol, around_date):
        cache[key] = symbol
        return symbol

    cache[key] = None
    return None


def fetch_close_prices(ticker: str, anchor_date: date):
    import pandas as pd  # type: ignore
    import yfinance as yf  # type: ignore

    start_date = anchor_date - timedelta(days=370)
    end_date = anchor_date + timedelta(days=370)
    df = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    if df is None or len(df) == 0 or "Close" not in df.columns:
        return pd.Series(dtype=float)

    close = df["Close"].dropna().copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def trend_direction(close, from_date: date, to_date: date) -> Optional[str]:
    import pandas as pd  # type: ignore

    if close.empty:
        return None

    start_ts = pd.Timestamp(min(from_date, to_date))
    end_ts = pd.Timestamp(max(from_date, to_date))
    window = close.loc[(close.index >= start_ts) & (close.index <= end_ts)]
    if window.shape[0] < 2:
        return None

    y = window.values.astype(float)
    return "up" if (y[-1] - y[0]) > 0 else "down"


def compute_trends(close, start_date: date) -> Dict[str, Any]:
    q_date = start_date + timedelta(days=91)
    h_date = start_date + timedelta(days=182)
    y_date = start_date + timedelta(days=365)
    return {
        "future": {
            "quarter": trend_direction(close, start_date, q_date),
            "half_year": trend_direction(close, start_date, h_date),
            "year": trend_direction(close, start_date, y_date),
        },
        "past": {
            "quarter": trend_direction(close, start_date - timedelta(days=91), start_date),
            "half_year": trend_direction(close, start_date - timedelta(days=182), start_date),
            "year": trend_direction(close, start_date - timedelta(days=365), start_date),
        },
    }


def build_sentiment_records(
    pdf_text_dir: Path,
    doc_company_names: Dict[str, str],
    existing_docs: Optional[Set[str]] = None,
    max_files: Optional[int] = None,
    sleep_seconds: float = 0.3,
) -> List[Dict[str, Any]]:
    from tqdm import tqdm  # type: ignore

    txt_files = sorted(pdf_text_dir.glob("*.txt"))
    if existing_docs:
        txt_files = [p for p in txt_files if p.stem not in existing_docs]
    if max_files is not None:
        txt_files = txt_files[:max_files]

    ticker_cache: Dict[str, Optional[str]] = {}
    rows: List[Dict[str, Any]] = []

    for txt_path in tqdm(txt_files, desc="Building sentiment", unit="doc"):
        doc_name = txt_path.stem
        company_name = doc_company_names.get(doc_name, doc_name.split("_", 1)[0])

        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        filing_date = extract_filing_date(text)

        row: Dict[str, Any] = {
            "document_name": doc_name,
            "company_name": company_name,
            "filing_date": filing_date.isoformat() if filing_date else None,
            "ticker": None,
            "price_trend": {
                "future": {"quarter": None, "half_year": None, "year": None},
                "past": {"quarter": None, "half_year": None, "year": None},
            },
            "status": "ok",
        }

        if filing_date is None:
            row["status"] = "date_not_found"
            rows.append(row)
            continue

        ticker = resolve_ticker(company_name, filing_date, cache=ticker_cache)
        row["ticker"] = ticker
        if not ticker:
            row["status"] = "ticker_not_found"
            rows.append(row)
            time.sleep(sleep_seconds)
            continue

        close = fetch_close_prices(ticker, filing_date)
        if close.empty:
            row["status"] = "no_price_data"
            rows.append(row)
            time.sleep(sleep_seconds)
            continue

        row["price_trend"] = compute_trends(close, filing_date)
        rows.append(row)
        time.sleep(sleep_seconds)

    return rows


def load_existing_records(output_path: Path) -> List[Dict[str, Any]]:
    if not output_path.exists():
        return []
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    records: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("document_name"), str):
            records.append(item)
    return records


def build_sentiment10k_data(
    output_path: str,
    workspace_dir: Optional[str] = None,
    direction: str = "forward",
    horizon: str = "quarter",
    only_10k: bool = True,
    max_context_chars: int = 300000,
    overwrite_text: bool = False,
    force_override_sentiment: bool = False,
    max_files: Optional[int] = None,
    sleep_seconds: float = 0.3,
    pdf_backend: PdfBackend = "auto",
    download_missing_pdfs: bool = True,
    refresh_metadata: bool = False,
    open_source_url: str = DEFAULT_OPEN_SOURCE_URL,
    doc_info_url: str = DEFAULT_DOC_INFO_URL,
) -> List[Dict[str, Any]]:
    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be one of: forward, backward")
    if horizon not in {"quarter", "half_year", "year"}:
        raise ValueError("horizon must be one of: quarter, half_year, year")

    if workspace_dir:
        ws_candidate = Path(workspace_dir)
        ws_dir = ws_candidate if ws_candidate.is_absolute() else (_repo_root() / ws_candidate)
        ws_dir = ws_dir.resolve()
    else:
        ws_dir = _default_workspace_dir()
    ws_dir.mkdir(parents=True, exist_ok=True)

    data_dir = ws_dir
    open_source_path, doc_info_path = ensure_metadata_files(
        data_dir=data_dir,
        open_source_url=open_source_url,
        doc_info_url=doc_info_url,
        force=refresh_metadata,
    )

    pdf_dir = ws_dir / "pdfs"
    pdf_text_dir = ws_dir / "pdfs_text"
    sentiment_json_path = ws_dir / "sentiment.json"

    doc_company_names = _load_doc_company_names(doc_info_path)
    doc_links = _load_doc_links(doc_info_path)
    required_docs = _required_doc_names(open_source_path, only_10k=only_10k)

    if download_missing_pdfs:
        ensure_financebench_pdfs(
            pdf_dir=pdf_dir,
            doc_links=doc_links,
            required_doc_names=required_docs,
        )

    ensure_pdf_text_cache(
        pdf_dir=pdf_dir,
        pdf_text_dir=pdf_text_dir,
        overwrite=overwrite_text,
        pdf_backend=pdf_backend,
    )

    existing_records: List[Dict[str, Any]] = []
    existing_docs: Set[str] = set()
    if not force_override_sentiment:
        existing_records = load_existing_records(sentiment_json_path)
        existing_docs = {str(r["document_name"]) for r in existing_records}

    new_rows = build_sentiment_records(
        pdf_text_dir=pdf_text_dir,
        doc_company_names=doc_company_names,
        existing_docs=existing_docs,
        max_files=max_files,
        sleep_seconds=sleep_seconds,
    )

    if force_override_sentiment:
        sentiment_rows = new_rows
    else:
        merged: Dict[str, Dict[str, Any]] = {str(r["document_name"]): r for r in existing_records}
        for row in new_rows:
            merged[str(row["document_name"])] = row
        sentiment_rows = [merged[k] for k in sorted(merged.keys())]

    sentiment_json_path.parent.mkdir(parents=True, exist_ok=True)
    sentiment_json_path.write_text(json.dumps(sentiment_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    direction_key = "future" if direction == "forward" else "past"
    examples: List[Dict[str, Any]] = []
    for row in sentiment_rows:
        if row.get("status") != "ok":
            continue

        doc_name = str(row.get("document_name", "")).strip()
        if not doc_name:
            continue
        if only_10k and "_10K" not in doc_name.upper():
            continue

        trend = row.get("price_trend", {}).get(direction_key, {}).get(horizon)
        if trend not in {"up", "down"}:
            continue

        text_path = (pdf_text_dir / f"{doc_name}.txt").resolve()
        if not text_path.exists():
            continue

        filing_date = row.get("filing_date")
        year = None
        if isinstance(filing_date, str) and len(filing_date) >= 4 and filing_date[:4].isdigit():
            year = int(filing_date[:4])

        examples.append(
            {
                "id": doc_name,
                "doc_name": doc_name,
                "year": year,
                "company_name": row.get("company_name"),
                "ticker": row.get("ticker"),
                "filing_date": filing_date,
                "direction": direction,
                "horizon": horizon,
                "question": (
                    "Classify the stock-movement sentiment using the filing context. "
                    "Return one token only: up or down."
                ),
                "text_path": str(text_path),
                "context_char_limit": int(max_context_chars),
                "answer": trend,
            }
        )

    output = Path(output_path)
    if not output.is_absolute():
        output = (_repo_root() / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Sentiment10k data")
    parser.add_argument("--output", type=str, default="data/sentiment10k/sentiment10k.json")
    parser.add_argument("--workspace-dir", type=str, default=None, help="Defaults to data/sentiment10k")
    parser.add_argument("--direction", type=str, default="forward", choices=["forward", "backward"])
    parser.add_argument("--horizon", type=str, default="quarter", choices=["quarter", "half_year", "year"])
    parser.add_argument("--all-doc-types", action="store_true", help="Include non-10K docs (default is 10K only)")
    parser.add_argument("--max-context-chars", type=int, default=300000)
    parser.add_argument("--overwrite-text", action="store_true")
    parser.add_argument("--force-override-sentiment", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--pdf-backend", type=str, default="auto", choices=["auto", "fitz", "pdfplumber", "pypdf"])
    parser.add_argument("--no-download-missing-pdfs", action="store_true")
    parser.add_argument("--refresh-metadata", action="store_true")
    parser.add_argument("--open-source-url", type=str, default=DEFAULT_OPEN_SOURCE_URL)
    parser.add_argument("--doc-info-url", type=str, default=DEFAULT_DOC_INFO_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    examples = build_sentiment10k_data(
        output_path=args.output,
        workspace_dir=args.workspace_dir,
        direction=args.direction,
        horizon=args.horizon,
        only_10k=not args.all_doc_types,
        max_context_chars=args.max_context_chars,
        overwrite_text=args.overwrite_text,
        force_override_sentiment=args.force_override_sentiment,
        max_files=args.max_files,
        sleep_seconds=args.sleep_seconds,
        pdf_backend=args.pdf_backend,
        download_missing_pdfs=not args.no_download_missing_pdfs,
        refresh_metadata=args.refresh_metadata,
        open_source_url=args.open_source_url,
        doc_info_url=args.doc_info_url,
    )

    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
