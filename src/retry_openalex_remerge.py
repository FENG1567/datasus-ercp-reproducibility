from __future__ import annotations

"""Retry the four failed OpenAlex queries and rebuild the v2 candidate list
from the persisted raw files so the merge is reproducible without re-querying
the other sources."""

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from search_ercp_literature import (
    QUERIES,
    fetch,
    merge_records,
    normalize_doi,
    normalize_title,
    openalex,
    save_csv,
    utc_now,
)

RETRY_QUERY_IDS = [
    "q3_datasus_ercp_stones",
    "q4_brazil_cholecystectomy_datasus",
    "q8_datasus_cpre",
    "q10_brazil_choledocholithiasis_treatment",
]


def load_raw_records(raw_root: Path) -> list[dict]:
    records: list[dict] = []
    for query_id, query in QUERIES.items():
        # PubMed: esearch JSON plus efetch XML batches
        pubmed_dir = raw_root / "pubmed"
        for efetch in sorted(pubmed_dir.glob(f"{query_id}_efetch_*.xml")):
            for article in ET.fromstring(efetch.read_bytes()).findall(".//PubmedArticle"):
                citation = article.find("MedlineCitation")
                journal_article = citation.find("Article") if citation is not None else None
                if citation is None or journal_article is None:
                    continue
                pmid = citation.findtext("PMID", default="")
                title = "".join(journal_article.find("ArticleTitle").itertext()) if journal_article.find("ArticleTitle") is not None else ""
                abstract = " ".join("".join(node.itertext()) for node in journal_article.findall("Abstract/AbstractText"))
                authors = []
                for author in journal_article.findall("AuthorList/Author"):
                    collective = author.findtext("CollectiveName")
                    name = collective or " ".join(filter(None, [author.findtext("ForeName"), author.findtext("LastName")]))
                    if name:
                        authors.append(name)
                journal = journal_article.findtext("Journal/Title", default="")
                year = journal_article.findtext("Journal/JournalIssue/PubDate/Year", default="")
                if not year:
                    medline_date = journal_article.findtext("Journal/JournalIssue/PubDate/MedlineDate", default="")
                    match = re.search(r"(?:19|20)\d{2}", medline_date or "")
                    year = match.group(0) if match else ""
                doi = ""
                for article_id in article.findall("PubmedData/ArticleIdList/ArticleId"):
                    if article_id.attrib.get("IdType") == "doi":
                        doi = article_id.text or ""
                records.append(
                    {"source": "PubMed", "query_ids": query_id, "title": title, "authors": "; ".join(authors),
                     "journal": journal, "year": year, "doi": normalize_doi(doi), "pmid": pmid,
                     "cited_by_count": "", "abstract": abstract, "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}
                )
        # Europe PMC: one search JSON per query
        epmc_path = raw_root / "europe_pmc" / f"{query_id}.json"
        if epmc_path.exists():
            data = json.loads(epmc_path.read_text(encoding="utf-8"))
            for item in data.get("resultList", {}).get("result", []):
                records.append(
                    {"source": "Europe PMC", "query_ids": query_id, "title": item.get("title", ""),
                     "authors": item.get("authorString", ""), "journal": item.get("journalTitle", ""),
                     "year": str(item.get("pubYear", "")), "doi": normalize_doi(item.get("doi")),
                     "pmid": str(item.get("pmid", "")), "cited_by_count": str(item.get("citedByCount", "")),
                     "abstract": item.get("abstractText", ""),
                     "source_url": f"https://europepmc.org/article/MED/{item.get('pmid')}" if item.get("pmid") else ""}
                )
        # OpenAlex: one works JSON per query
        oa_path = raw_root / "openalex" / f"{query_id}.json"
        if oa_path.exists():
            data = json.loads(oa_path.read_text(encoding="utf-8"))
            for item in data.get("results", []):
                authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])]
                location = item.get("primary_location") or {}
                source = location.get("source") or {}
                inverted = item.get("abstract_inverted_index") or {}
                positions = [(pos, word) for word, values in inverted.items() for pos in values]
                abstract = " ".join(word for _, word in sorted(positions))
                ids = item.get("ids") or {}
                records.append(
                    {"source": "OpenAlex", "query_ids": query_id, "title": item.get("title", ""),
                     "authors": "; ".join(filter(None, authors)), "journal": source.get("display_name", ""),
                     "year": str(item.get("publication_year", "")), "doi": normalize_doi(item.get("doi")),
                     "pmid": str(ids.get("pmid", "")).rsplit("/", 1)[-1] if ids.get("pmid") else "",
                     "cited_by_count": str(item.get("cited_by_count", "")), "abstract": abstract,
                     "source_url": item.get("id", "")}
                )
        # Crossref: one works JSON per query
        cr_path = raw_root / "crossref" / f"{query_id}.json"
        if cr_path.exists():
            data = json.loads(cr_path.read_text(encoding="utf-8"))
            for item in data.get("message", {}).get("items", []):
                title = (item.get("title") or [""])[0]
                authors = [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in item.get("author", [])]
                journal = (item.get("container-title") or [""])[0]
                date = item.get("published-print") or item.get("published-online") or {}
                parts = date.get("date-parts") or [[]]
                year = str(parts[0][0]) if parts and parts[0] else ""
                records.append(
                    {"source": "Crossref", "query_ids": query_id, "title": title, "authors": "; ".join(authors),
                     "journal": journal, "year": year, "doi": normalize_doi(item.get("DOI")), "pmid": "",
                     "cited_by_count": str(item.get("is-referenced-by-count", "")),
                     "abstract": re.sub(r"<[^>]+>", " ", item.get("abstract", "")), "source_url": item.get("URL", "")}
                )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("provenance/literature_raw_v2"))
    parser.add_argument("--output", type=Path, default=Path("provenance/ercp_literature_candidates_v2.csv"))
    parser.add_argument("--audit", type=Path, default=Path("provenance/ercp_literature_search_audit_v2.json"))
    parser.add_argument("--event-log", type=Path, default=Path("provenance/literature_retry_events.jsonl"))
    args = parser.parse_args()

    events = []
    retry_failures = []
    for query_id in RETRY_QUERY_IDS:
        try:
            rows = openalex(query_id, QUERIES[query_id], args.raw_root)
            events.append({"timestamp": utc_now(), "status": "retry_pass", "source": "openalex",
                           "query_id": query_id, "records": len(rows)})
            print(f"PASS {query_id}: {len(rows)} records", flush=True)
        except Exception as exc:
            retry_failures.append({"source": "openalex", "query_id": query_id, "error_class": type(exc).__name__})
            events.append({"timestamp": utc_now(), "status": "retry_fail", "source": "openalex",
                           "query_id": query_id, "error_class": type(exc).__name__})
            print(f"FAIL {query_id}: {type(exc).__name__}", flush=True)

    all_records = load_raw_records(args.raw_root)
    merged = merge_records(all_records)
    save_csv(args.output, merged)

    source_totals: dict[str, int] = {}
    for row in all_records:
        source_totals[row["source"]] = source_totals.get(row["source"], 0) + 1
    audit = {
        "schema_version": "1.0",
        "accessed_at": utc_now(),
        "queries": QUERIES,
        "sources": source_totals,
        "openalex_retry_failures": retry_failures,
        "retrieved_records": len(all_records),
        "deduplicated_records": len(merged),
        "output": str(args.output),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.event_log.parent.mkdir(parents=True, exist_ok=True)
    with args.event_log.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 1 if retry_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())