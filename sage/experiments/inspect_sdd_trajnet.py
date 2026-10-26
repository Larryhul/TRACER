from __future__ import annotations

import argparse
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("pandas is required to inspect SDD trajnet pickle files") from exc

try:
    import dill as pickle_backend
except ModuleNotFoundError:
    import pickle as pickle_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect columns and IDs in SDD trajnet pickle files.")
    parser.add_argument("--path", default="raw_data/stanford/train_trajnet.pkl")
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--scenes", type=int, default=20)
    args = parser.parse_args()

    with Path(args.path).open("rb") as f:
        try:
            df = pickle_backend.load(f, encoding="latin1")
        except TypeError:
            f.seek(0)
            df = pickle_backend.load(f)

    print(f"path={args.path}")
    print(f"type={type(df)}")
    print(f"shape={getattr(df, 'shape', None)}")
    print(f"columns={list(df.columns)}")
    print("head=")
    print(df.head(args.rows).to_string(index=False))

    for col in ["sceneId", "metaId", "trackId", "frame"]:
        if col in df.columns:
            values = pd.Series(df[col]).dropna().drop_duplicates().head(args.scenes).tolist()
            print(f"unique_{col}_sample={values}")
            print(f"num_unique_{col}={df[col].nunique(dropna=True)}")

    if "sceneId" in df.columns and "metaId" in df.columns:
        pairs = df[["sceneId", "metaId"]].drop_duplicates().head(args.scenes)
        print("scene_meta_pairs=")
        print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
