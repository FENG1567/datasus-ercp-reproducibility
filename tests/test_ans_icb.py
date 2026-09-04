import zipfile

from src.download_ans_icb import FILE_RE, validate_archive


def test_ans_filename_contract():
    match = FILE_RE.match("pda-024-icb-SP-2025_12.zip")
    assert match is not None
    assert match.groups() == ("SP", "2025", "12")


def test_validate_archive_requires_exact_csv(tmp_path):
    archive = tmp_path / "pda-024-icb-AC-2021_12.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("pda-024-icb-AC-2021_12.csv", "a;b\n1;2\n")
    validate_archive(archive, "pda-024-icb-AC-2021_12.csv")
