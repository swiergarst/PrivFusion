"""Smoke test: the Streamlit app renders a synthetic workspace without errors."""

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "viz" / "app.py"
RUN_ID = "123.5"


def _build_workspace(root: Path) -> None:
    (root / "datasets").mkdir(parents=True)
    (root / "dataset_modifications").mkdir()
    (root / "dataset_checkpoints").mkdir()
    (root / "logs").mkdir()
    (root / "config.yaml").write_text(
        "rounds: 1\ndataset_config:\n  - name: dsA\n    path: datasets/ds_a.csv\n", encoding="utf-8"
    )

    # base dataset (written with the index, like datasets/*.csv)
    base = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    base.to_csv(root / "datasets" / "ds_a.csv", index=True)

    # round-0 checkpoint: agent loaded the base, renamed a column, wrote with the index
    mod = pd.read_csv(root / "datasets" / "ds_a.csv").rename(columns={"a": "alpha"})
    mod.to_csv(root / "dataset_checkpoints" / f"dsA_modified_round0_{RUN_ID}.csv", index=True)

    # instructions: one normalized row (rename a -> alpha), one dropped row
    pd.DataFrame(
        {
            "cluster_id": [1, 9999],
            "feature_name": ["a", "b"],
            "dataset": ["dsA", "dsA"],
            "norm_uri": ["http://example/xsd", ""],
            "norm_feature_name": ["alpha", ""],
            "norm_value_structure": ["", ""],
            "transformation": ["rename", ""],
        }
    ).to_csv(root / "dataset_modifications" / f"harmonization_instructions_round_0_{RUN_ID}.csv", index=False)

    (root / "logs" / f"fed_harmonization_logs_round0_id_{RUN_ID}.log").write_text(
        "2026-10-19 10:00:00,000 fedLogger INFO input messages: [{'role': 'user', 'content': 'hi'}]\n"
        "2026-10-19 10:00:02,000 fedLogger INFO response: {'model': 'm', "
        "'choices': [{'message': {'content': 'ok'}}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 2}}\n",
        encoding="utf-8",
    )
    (root / "logs" / f"local_modif_logs_round0_id_{RUN_ID}.log").write_text(
        "2026-10-19 11:00:01,000 localLogger INFO >>> LLM REQUEST <<<\n"
        "Columns/dtypes:\n"
        "Unnamed: 0    int64\n"
        "a    int64\n"
        "b    str\n"
        "\n"
        "2026-10-19 11:00:05,000 localLogger INFO tool call: apply_transformation(description='rename', code='x')\n"
        "2026-10-19 11:00:05,500 localLogger INFO tool result: apply_transformation -> OK\n"
        "2026-10-19 11:00:06,000 localLogger INFO >>> LLM RESPONSE <<<\n"
        "done\n",
        encoding="utf-8",
    )
    with (root / "logs" / f"local_results_dsA_round_0_id_{RUN_ID}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "success": True,
                "summary": "renamed a -> alpha",
                "steps": [{"description": "rename a -> alpha", "code": "df = df.rename(columns=dict(a='alpha'))"}],
                "replay_ok": True,
                "replay_message": "replay ok",
            },
            f,
        )


def test_app_renders_synthetic_workspace(tmp_path: Path) -> None:
    _build_workspace(tmp_path)

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception

    # point the app at the synthetic workspace
    at.text_input[0].set_value(str(tmp_path)).run()
    assert not at.exception, [str(e.value) for e in at.exception]

    # the synthetic run is discovered and selected by default
    run_box = [s for s in at.selectbox if s.label == "Run"][0]
    assert run_box.options == ["123.5"]  # formatted display value

    # the diff for round 0 / dsA should have run (metrics for the diff view exist)
    labels = {m.label: m.value for m in at.metric}
    assert labels.get("Comparison") == "identity"
    assert labels.get("Rows") == "3 → 3"

    # the rename declared in the instructions is reported as a rename, not add+remove
    col_tables = [el.value for el in at.dataframe if "change" in getattr(el.value, "columns", [])]
    assert col_tables, "per-column table not found"
    row = col_tables[0][col_tables[0]["column"] == "a -> alpha"]
    assert len(row) == 1
    assert row.iloc[0]["change"] == "renamed"


def test_app_without_artifacts_shows_warning(tmp_path: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception

    at.text_input[0].set_value(str(tmp_path)).run()
    assert not at.exception
    assert any("No runs found" in (w.value or "") for w in at.warning)
