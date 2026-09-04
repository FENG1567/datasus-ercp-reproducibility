from src.search_ercp_literature import merge_records, normalize_doi, normalize_title


def test_doi_normalization():
    assert normalize_doi("https://doi.org/10.1000/ABC.1.") == "10.1000/abc.1"


def test_title_normalization():
    assert normalize_title("ERCP in Brazil: Access & Equity") == "ercp in brazil access equity"


def test_merge_prefers_pubmed_and_unions_queries():
    common = {"authors": "", "journal": "", "year": "2024", "pmid": "", "abstract": "", "source_url": ""}
    rows = [
        {**common, "source": "OpenAlex", "query_ids": "q2", "title": "A", "doi": "10.1/x", "cited_by_count": "5"},
        {**common, "source": "PubMed", "query_ids": "q1", "title": "A full title", "doi": "10.1/x", "cited_by_count": ""},
    ]
    merged = merge_records(rows)
    assert len(merged) == 1
    assert merged[0]["source"] == "PubMed"
    assert merged[0]["cited_by_count"] == 5
    assert merged[0]["query_ids"] == "q1; q2"
