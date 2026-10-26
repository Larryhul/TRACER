from sage.experiments.audit_external_baselines import _finite_csv


def test_finite_csv_accepts_required_numeric_rows(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("success,cost\n1,0.5\n0,1.25\n")

    result = _finite_csv(path, ("success", "cost"))

    assert result["rows"] == 2
    assert result["missing_fields"] == []
    assert result["nonfinite_cells"] == 0
    assert result["complete"] is True


def test_finite_csv_rejects_nan_and_missing_fields(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("success,cost\n1,nan\n")

    result = _finite_csv(path, ("success", "cost", "progress"))

    assert result["missing_fields"] == ["progress"]
    assert result["nonfinite_cells"] == 2
    assert result["complete"] is False
