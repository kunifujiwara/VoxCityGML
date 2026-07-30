"""Parse failures must be visible (warnings + summary), not silent."""
import logging
from pathlib import Path

import numpy as np

from voxcitygml.citygml.parser import _parse_single_file, parse_citygml_directory


def _write_corrupt_gml(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_text("<not-xml", encoding="utf-8")
    return p


def test_parse_single_file_warns_on_failure(tmp_path, caplog):
    bad = _write_corrupt_gml(tmp_path, "53393671_bldg_6697_op.gml")
    failures = []
    with caplog.at_level(logging.WARNING):
        result = _parse_single_file(bad, 'building', None, None,
                                    failures=failures)
    assert result == []
    assert failures == [str(bad)]
    assert any("53393671_bldg_6697_op.gml" in r.getMessage() for r in caplog.records)


def test_directory_parse_reports_summary(tmp_path, capsys):
    # PLATEAU layout with one corrupt building file
    _write_corrupt_gml(tmp_path / "udx" / "bldg", "53393671_bldg_6697_op.gml")
    collection = parse_citygml_directory(str(tmp_path),
                                         feature_types=['building'])
    assert collection.buildings == []
    out = capsys.readouterr().out
    assert "1 file(s) failed to parse" in out
