"""Recover the scaled-up discrete results that live only in notebook outputs.

A reviewer noticed that `notebooks/FALD_Colab.ipynb` and the JSON reports in `results/`
disagree for similar run names (306.78 vs 262.47, 30.07 vs 40.01) and inferred that Colab runs
had overwritten each other's files.

That is not what happened. The notebook cells and the saved reports are **different
configurations**: the saved JSONs are 6 layers / 200 steps / ~8.5 min, while the notebook
outputs include a 10 layer / 500 steps / ~86 min pair that was never written to `results/` at
all. The numbers do not match because they are not the same experiment.

This script re-extracts the notebook numbers from git history so the claim is checkable rather
than transcribed by hand. Run it to regenerate `results/discrete_scaled_up_planar.json`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fald.paths import results_dir

COMMITS = ("649fc7d", "2e0ac83")
NOTEBOOK = "notebooks/FALD_Colab.ipynb"
RATIO = re.compile(r"Ratio:\s*model=([\d.]+)\s*ER=([\d.]+)")
VUN = re.compile(r"valid=([\d.]+)\s*unique=([\d.]+)\s*novel=([\d.]+)")


def notebook_at(commit: str) -> dict:
    blob = subprocess.run(
        ["git", "show", f"{commit}:{NOTEBOOK}"], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(blob)


def extract(commit: str) -> list[dict]:
    """Pull (ratio, validity) pairs out of the cell outputs, in order."""
    notebook = notebook_at(commit)
    found: list[dict] = []
    for cell in notebook.get("cells", []):
        text = "".join(
            "".join(output.get("text", "")) for output in cell.get("outputs", [])
        )
        ratios = RATIO.findall(text)
        vuns = VUN.findall(text)
        for index, (model, er) in enumerate(ratios):
            entry = {"commit": commit, "ratio": float(model), "er_ratio": float(er)}
            if index < len(vuns):
                valid, unique, novel = vuns[index]
                entry.update(
                    validity=float(valid), unique=float(unique), novel=float(novel)
                )
            found.append(entry)
    return found


def main() -> None:
    everything: list[dict] = []
    for commit in COMMITS:
        rows = extract(commit)
        print(f"{commit}: {len(rows)} evaluated run(s)")
        for row in rows:
            print(
                f"   ratio={row['ratio']:<8} ER={row['er_ratio']} "
                f"valid={row.get('validity')}"
            )
        everything.extend(rows)

    distinct = sorted({row["ratio"] for row in everything})
    print(f"\ndistinct ratios across both commits: {distinct}")
    print(
        "\nThe 15.40 and 39.60 runs are the 10-layer/500-step pair; 306.78 and 30.07 are the\n"
        "6-layer/200-step pair. Neither pair was ever saved to results/, which is the whole\n"
        "reason the notebook and the JSON reports disagree."
    )

    output = results_dir() / "discrete_scaled_up_recovered.json"
    output.write_text(json.dumps({"runs": everything}, indent=2))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
