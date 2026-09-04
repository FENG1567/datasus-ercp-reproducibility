import hashlib

from src.download_datasus_portal import file_batches, raw_file_is_complete


def test_file_batches_preserve_order_and_bound_size():
    files = [{"filename": f"PF{i:03d}.dbc"} for i in range(37)]
    batches = file_batches(files, 18)
    assert [len(batch) for batch in batches] == [18, 18, 1]
    assert [row["filename"] for batch in batches for row in batch] == [
        row["filename"] for row in files
    ]


def test_file_batches_reject_nonpositive_size():
    try:
        file_batches([], 0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_raw_file_complete_requires_matching_size_and_hash(tmp_path):
    row = {"filename": "PFAC2101.dbc", "year": 2021, "uf": "AC"}
    path = tmp_path / "2021" / "AC" / row["filename"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"dbc")
    manifest = {
        row["filename"]: {
            "status": "complete",
            "local_size_bytes": "3",
            "sha256": hashlib.sha256(b"dbc").hexdigest(),
        }
    }
    assert raw_file_is_complete(row, tmp_path, manifest)
    manifest[row["filename"]]["sha256"] = "0" * 64
    assert not raw_file_is_complete(row, tmp_path, manifest)
