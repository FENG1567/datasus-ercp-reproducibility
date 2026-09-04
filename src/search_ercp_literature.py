from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


USER_AGENT = "DATASUS-ERCP-literature-audit/1.0"
QUERIES = {
    "q1_brazil_ercp": '(Brazil OR Brasil) AND (ERCP OR "endoscopic retrograde cholangiopancreatography")',
    "q2_brazil_choledocholithiasis_access": '(Brazil OR Brasil) AND (choledocholithiasis OR "common bile duct stones") AND (access OR equity OR referral OR regional OR network)',
    "q3_datasus_ercp_stones": '(DATASUS OR "SIH-SUS" OR "Sistema Único de Saúde") AND (ERCP OR choledocholithiasis OR gallstone)',
    "q4_brazil_cholecystectomy_datasus": '(Brazil OR Brasil) AND (cholecystectomy OR cholelithiasis OR gallstone) AND (DATASUS OR SUS OR hospitalization)',
    "q5_brazil_ercp_diffusion_network": '(Brazil OR Brasil) AND (ERCP OR cholangiopancreatography) AND (diffusion OR adoption OR implementation OR network OR referral OR geography)',
    "q6_therapeutic_ercp_equity": '"therapeutic ERCP" AND (access OR equity OR disparity OR rural OR regionalization)',
    "q7_brazil_cpre": '(Brazil OR Brasil) AND (CPRE OR "colangiopancreatografia retrógrada endoscópica")',
    "q8_datasus_cpre": '(DATASUS OR SUS) AND (CPRE OR "colangiopancreatografia retrógrada endoscópica")',
    "q9_sigtap_ercp_code": '("0407030255" OR "04.07.03.025-5")',
    "q10_brazil_choledocholithiasis_treatment": '(Brazil OR Brasil) AND ("tratamento de coledocolitíase" OR "manejo da coledocolitíase" OR "choledocholithiasis treatment")',
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch(url: str, timeout: int = 120, attempts: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, application/xml;q=0.9, */*;q=0.1"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Request failed after {attempts} attempts: {type(last_error).__name__}") from last_error


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def save_raw(root: Path, source: str, query_id: str, suffix: str, payload: bytes) -> Path:
    path = root / source / f"{safe_name(query_id)}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi:\s*", "", value)
    return value.rstrip(". ,;)")


def normalize_title(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def pubmed(query_id: str, query: str, raw_root: Path) -> list[dict]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    search_url = base + "/esearch.fcgi?" + urllib.parse.urlencode(
        {"db": "pubmed", "term": query, "retmode": "json", "retmax": 500, "sort": "relevance"}
    )
    search_payload = fetch(search_url)
    save_raw(raw_root, "pubmed", query_id + "_esearch", "json", search_payload)
    search = json.loads(search_payload)
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    articles = []
    for offset in range(0, len(ids), 200):
        batch = ids[offset:offset + 200]
        fetch_url = base + "/efetch.fcgi?" + urllib.parse.urlencode(
            {"db": "pubmed", "id": ",".join(batch), "retmode": "xml"}
        )
        payload = fetch(fetch_url)
        save_raw(raw_root, "pubmed", f"{query_id}_efetch_{offset // 200 + 1:03d}", "xml", payload)
        articles.extend(ET.fromstring(payload).findall(".//PubmedArticle"))
        time.sleep(0.4)
    rows = []
    for article in articles:
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
            match = re.search(r"(?:19|20)\d{2}", medline_date)
            year = match.group(0) if match else ""
        doi = ""
        for article_id in article.findall("PubmedData/ArticleIdList/ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = article_id.text or ""
        rows.append(
            {"source": "PubMed", "query_ids": query_id, "title": title, "authors": "; ".join(authors),
             "journal": journal, "year": year, "doi": normalize_doi(doi), "pmid": pmid,
             "cited_by_count": "", "abstract": abstract, "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}
        )
    return rows


def europe_pmc(query_id: str, query: str, raw_root: Path) -> list[dict]:
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
        {"query": query, "format": "json", "pageSize": 1000, "resultType": "core"}
    )
    payload = fetch(url)
    save_raw(raw_root, "europe_pmc", query_id, "json", payload)
    data = json.loads(payload)
    rows = []
    for item in data.get("resultList", {}).get("result", []):
        rows.append(
            {"source": "Europe PMC", "query_ids": query_id, "title": item.get("title", ""),
             "authors": item.get("authorString", ""), "journal": item.get("journalTitle", ""),
             "year": str(item.get("pubYear", "")), "doi": normalize_doi(item.get("doi")),
             "pmid": str(item.get("pmid", "")), "cited_by_count": str(item.get("citedByCount", "")),
             "abstract": item.get("abstractText", ""),
             "source_url": f"https://europepmc.org/article/MED/{item.get('pmid')}" if item.get("pmid") else ""}
        )
    return rows


def openalex(query_id: str, query: str, raw_root: Path) -> list[dict]:
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": query, "per-page": 200, "select": "id,doi,title,publication_year,authorships,primary_location,cited_by_count,abstract_inverted_index,ids"}
    )
    payload = fetch(url)
    save_raw(raw_root, "openalex", query_id, "json", payload)
    data = json.loads(payload)
    rows = []
    for item in data.get("results", []):
        authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])]
        location = item.get("primary_location") or {}
        source = location.get("source") or {}
        inverted = item.get("abstract_inverted_index") or {}
        positions = [(pos, word) for word, values in inverted.items() for pos in values]
        abstract = " ".join(word for _, word in sorted(positions))
        ids = item.get("ids") or {}
        rows.append(
            {"source": "OpenAlex", "query_ids": query_id, "title": item.get("title", ""),
             "authors": "; ".join(filter(None, authors)), "journal": source.get("display_name", ""),
             "year": str(item.get("publication_year", "")), "doi": normalize_doi(item.get("doi")),
             "pmid": str(ids.get("pmid", "")).rsplit("/", 1)[-1] if ids.get("pmid") else "",
             "cited_by_count": str(item.get("cited_by_count", "")), "abstract": abstract,
             "source_url": item.get("id", "")}
        )
    return rows


def crossref(query_id: str, query: str, raw_root: Path) -> list[dict]:
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": query, "rows": 200, "select": "DOI,title,author,published-print,published-online,container-title,is-referenced-by-count,URL,abstract"}
    )
    payload = fetch(url)
    save_raw(raw_root, "crossref", query_id, "json", payload)
    data = json.loads(payload)
    rows = []
    for item in data.get("message", {}).get("items", []):
        title = (item.get("title") or [""])[0]
        authors = [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in item.get("author", [])]
        journal = (item.get("container-title") or [""])[0]
        date = item.get("published-print") or item.get("published-online") or {}
        parts = date.get("date-parts") or [[]]
        year = str(parts[0][0]) if parts and parts[0] else ""
        rows.append(
            {"source": "Crossref", "query_ids": query_id, "title": title, "authors": "; ".join(authors),
             "journal": journal, "year": year, "doi": normalize_doi(item.get("DOI")), "pmid": "",
             "cited_by_count": str(item.get("is-referenced-by-count", "")),
             "abstract": re.sub(r"<[^>]+>", " ", item.get("abstract", "")), "source_url": item.get("URL", "")}
        )
    return rows


def merge_records(records: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        if row["doi"]:
            key = ("doi", row["doi"])
        elif row["pmid"]:
            key = ("pmid", row["pmid"])
        else:
            key = ("title", normalize_title(row["title"]))
        groups[key].append(row)
    merged = []
    preference = {"PubMed": 0, "Europe PMC": 1, "OpenAlex": 2, "Crossref": 3}
    for rows in groups.values():
        rows.sort(key=lambda row: preference.get(row["source"], 99))
        base = dict(rows[0])
        for field in ["title", "authors", "journal", "year", "doi", "pmid", "abstract", "source_url"]:
            if not base.get(field):
                base[field] = next((row[field] for row in rows if row.get(field)), "")
        counts = [int(row["cited_by_count"]) for row in rows if str(row.get("cited_by_count", "")).isdigit()]
        base["cited_by_count"] = max(counts) if counts else ""
        base["sources"] = "; ".join(sorted({row["source"] for row in rows}))
        base["query_ids"] = "; ".join(sorted({qid for row in rows for qid in row["query_ids"].split("; ")}))
        merged.append(base)
    merged.sort(key=lambda row: (int(row["year"]) if str(row["year"]).isdigit() else 0, int(row["cited_by_count"] or 0)), reverse=True)
    return merged


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["title", "authors", "journal", "year", "doi", "pmid", "cited_by_count", "sources", "query_ids", "abstract", "source_url"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("provenance/literature_raw"))
    parser.add_argument("--output", type=Path, default=Path("provenance/ercp_literature_candidates.csv"))
    parser.add_argument("--audit", type=Path, default=Path("provenance/ercp_literature_search_audit.json"))
    args = parser.parse_args()

    all_records = []
    audit = {"schema_version": "1.0", "accessed_at": utc_now(), "queries": QUERIES, "sources": {}, "failures": []}
    sources = [("pubmed", pubmed), ("europe_pmc", europe_pmc), ("openalex", openalex), ("crossref", crossref)]
    for source_name, function in sources:
        source_total = 0
        for query_id, query in QUERIES.items():
            try:
                rows = function(query_id, query, args.raw_root)
                all_records.extend(rows)
                source_total += len(rows)
            except Exception as exc:
                audit["failures"].append({"source": source_name, "query_id": query_id, "error_class": type(exc).__name__})
        audit["sources"][source_name] = source_total
    merged = merge_records(all_records)
    save_csv(args.output, merged)
    audit["retrieved_records"] = len(all_records)
    audit["deduplicated_records"] = len(merged)
    audit["output"] = str(args.output)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # ASCII-safe console output avoids Windows legacy code-page failures; the
    # persisted audit remains UTF-8 with the original Portuguese terms.
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    return 1 if len(audit["failures"]) == len(sources) * len(QUERIES) else 0


if __name__ == "__main__":
    raise SystemExit(main())
