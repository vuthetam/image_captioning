"""Add official COCO IDs to predictions without changing Karpathy imgid values."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import PREDICTIONS_PATH, TEST_DF_PATH


def add_cocoids(predictions: list[dict], test_df: pd.DataFrame) -> list[dict]:
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Predictions must be a non-empty JSON list.")
    if not {"imgid", "cocoid"}.issubset(test_df.columns):
        raise ValueError("Test split must contain imgid and cocoid columns.")

    mapping = test_df[["imgid", "cocoid"]].drop_duplicates()
    for column in ("imgid", "cocoid"):
        if mapping[column].isna().any() or not pd.api.types.is_integer_dtype(mapping[column]):
            raise ValueError(f"Test split {column} values must be non-null integers.")
        if (mapping[column] < 0).any():
            raise ValueError(f"Test split {column} values must be non-negative.")
    if mapping["imgid"].duplicated().any():
        raise ValueError("An imgid maps to multiple cocoid values in the test split.")
    if mapping["cocoid"].duplicated().any():
        raise ValueError("A cocoid maps to multiple imgid values in the test split.")
    cocoid_lookup = dict(mapping.itertuples(index=False, name=None))

    enriched_predictions = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise ValueError(f"Prediction {index} must be a JSON object.")
        imgid = prediction.get("imgid")
        if not isinstance(imgid, int):
            raise ValueError(f"Prediction {index} must contain an integer imgid.")
        if imgid not in cocoid_lookup:
            raise ValueError(f"No cocoid found for prediction imgid={imgid} in the test split.")
        cocoid = int(cocoid_lookup[imgid])
        if "cocoid" in prediction and (
            not isinstance(prediction["cocoid"], int)
            or prediction["cocoid"] != cocoid
        ):
            raise ValueError(f"Existing cocoid does not match the test split for imgid={imgid}.")
        enriched_prediction = dict(prediction)
        enriched_prediction["cocoid"] = cocoid
        enriched_predictions.append(enriched_prediction)

    return enriched_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--test-df", type=Path, default=TEST_DF_PATH)
    parser.add_argument("--output", type=Path, help="Default: <predictions_stem>_cocoid.json alongside the input.")
    args = parser.parse_args()
    output_path = args.output
    if output_path is None:
        output_path = args.predictions.with_name(f"{args.predictions.stem}_cocoid.json")
    if output_path.resolve() == args.predictions.resolve():
        raise ValueError("Output must differ from the original predictions file.")

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    test_df = pd.read_parquet(args.test_df, columns=["imgid", "cocoid"])
    enriched_predictions = add_cocoids(predictions, test_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(enriched_predictions, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved {len(enriched_predictions):,} predictions with cocoid to {output_path}")


if __name__ == "__main__":
    main()
