"""Static checks on the saved notebook, results and README. These do not retrain a model."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "bank_marketing_classification.ipynb").read_text(encoding="utf-8"))
README = (ROOT / "README.md").read_text(encoding="utf-8")
BASELINES = json.loads((ROOT / "analysis" / "baseline_results.json").read_text(encoding="utf-8"))
CODE = [cell for cell in NOTEBOOK["cells"] if cell["cell_type"] == "code"]


def _stream_text():
    return "".join(
        "".join(output.get("text", []))
        for cell in CODE
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def test_notebook_has_saved_outputs_and_no_errors():
    assert any(cell.get("outputs") for cell in CODE)
    assert not [o for c in CODE for o in c.get("outputs", []) if o.get("output_type") == "error"]


def test_notebook_has_no_machine_specific_paths():
    blob = json.dumps(NOTEBOOK)
    assert "/content/" not in blob and "AppData" not in blob and "C:\\\\Users" not in blob


def test_readme_test_figures_match_the_saved_notebook():
    text = _stream_text()
    for line, value in (("TEST AUC: 0.9268", "0.9268"), ("TEST ACC: 0.9051", "0.9051"), ("TEST F1 : 0.8919", "0.8919")):
        assert line in text
        assert value in README


def test_readme_makes_no_stratified_claim_about_the_notebook_split():
    assert "not stratified" in README
    assert "weighted F1" in README
    assert "CrossValidator" in README


def test_baseline_shows_the_majority_class_rate():
    assert "0.8830" in README
    assert "45,211" in README
    assert BASELINES


def test_dataset_is_not_committed():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/" in ignored or "bank-full.csv" in ignored
