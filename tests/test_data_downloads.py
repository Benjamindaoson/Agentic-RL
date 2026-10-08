import io
import json
import zipfile

import pytest

from scripts.download_spider import extract_verified_archive, sha256, validate_archive
from scripts.validate_bird_metadata import audit_metadata


def test_spider_download_verifies_real_zip_bytes_offline(tmp_path):
    archive = tmp_path / "spider.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("spider/train_spider.json", "[]")
        zf.writestr("spider/dev.json", "[]")
    with zipfile.ZipFile(archive) as zf:
        metadata = validate_archive(zf)
    assert metadata["members"] == 2
    assert len(sha256(archive)) == 64
    destination = tmp_path / "data"
    destination.mkdir()
    extract_verified_archive(archive, destination)
    assert (destination / "spider" / "dev.json").is_file()


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.txt", "X:/evil.txt", "..\\win-escape.txt"])
def test_spider_rejects_archive_escape(tmp_path, name):
    archive = tmp_path / "malicious.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(name, "BAD")
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(ValueError, match="unsafe"):
            validate_archive(zf)


def test_bird_metadata_checks_row_count_and_fields(tmp_path):
    file = tmp_path / "bird.jsonl"
    file.write_text(json.dumps({"db_id": "music", "question": "How many?", "SQL": "SELECT COUNT(*) FROM tracks"}) + "\n")
    report = audit_metadata(file, expected_rows=1)
    assert report["rows"] == 1
    assert report["databases_downloaded_and_verified"] is False
    with pytest.raises(ValueError, match="expected 6601"):
        audit_metadata(file)
