import argparse
import json
import random
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


def set_seed(seed: int = 24):
    """Set random seed for reproducibility."""
    random.seed(seed)


def save_samples(records: list[dict], output_path: Path, format: str = "json"):
    """Save sampled records to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    elif format == "parquet" and pd is not None:
        df = pd.DataFrame(records)
        df.to_parquet(output_path, index=False)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def sample_parquet(input_path: Path, output_path: Path, sample_ratio: float = 0.5, seed: int = 42):
    """Sample records from a parquet file while preserving parquet schema."""
    if pd is None:
        raise ModuleNotFoundError("pandas is required for parquet files")

    set_seed(seed)
    df = pd.read_parquet(input_path)

    total = len(df)
    if total == 0:
        raise ValueError(f"No valid records found in {input_path}")

    n_sample = max(1, int(total * sample_ratio))

    sampled_df = df.sample(n=n_sample, random_state=seed).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_parquet(output_path, index=False)
    return total, n_sample


def sample_json(input_path: Path, output_path: Path, sample_ratio: float = 0.5, seed: int = 42):
    """Sample records from a JSON file."""
    set_seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # Convert dict to list of records
        records = []
        for k, v in data.items():
            if isinstance(v, dict):
                v = dict(v)
                v.setdefault("id", k)
            records.append(v)
    else:
        raise ValueError(f"Unsupported JSON structure in {input_path}")

    n_sample = max(1, int(len(records) * sample_ratio))
    sampled = random.sample(records, n_sample)

    save_samples(sampled, output_path, "json")
    return len(records), n_sample


def sample_jsonl(input_path: Path, output_path: Path, sample_ratio: float = 0.5, seed: int = 42):
    """Sample records from a JSONL file."""
    set_seed(seed)

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    n_sample = max(1, int(len(records) * sample_ratio))
    sampled = random.sample(records, n_sample)

    save_samples(sampled, output_path, "jsonl")
    return len(records), n_sample


def sample_scienceqa_directory(input_dir: Path, output_dir: Path, sample_ratio: float = 0.5, seed: int = 42):
    
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    problems_path = input_dir / "problems.json"
    if not problems_path.exists():
        raise FileNotFoundError(f"ScienceQA problems.json not found: {problems_path}")

    with open(problems_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = list(data.items())
        total = len(items)
        if total == 0:
            raise ValueError(f"No valid records found in {problems_path}")

        n_sample = max(1, int(total * sample_ratio))
        sampled_items = random.sample(items, n_sample)
        sampled_data = {k: v for k, v in sampled_items}

    elif isinstance(data, list):
        total = len(data)
        if total == 0:
            raise ValueError(f"No valid records found in {problems_path}")

        n_sample = max(1, int(total * sample_ratio))
        sampled_data = random.sample(data, n_sample)

    else:
        raise ValueError(f"Unsupported JSON structure in {problems_path}")

    output_path = output_dir / "problems.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)

    return total, n_sample


def sample_directory(input_dir: Path, output_dir: Path, sample_ratio: float = 0.5, seed: int = 42):
    """Sample records from a directory."""
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if parquet_files:
        if pd is None:
            raise ModuleNotFoundError("pandas is required for parquet files")

        all_records = []

        for pf in parquet_files:
            df = pd.read_parquet(pf)
            records = df.to_dict(orient="records")
            all_records.extend(records)

        n_sample = max(1, int(len(all_records) * sample_ratio))
        sampled = random.sample(all_records, n_sample)

        # Save as multiple parquet files (same behavior as original code)
        batch_size = max(1, len(sampled) // len(parquet_files)) if parquet_files else len(sampled)

        for i, pf in enumerate(parquet_files[:max(1, (len(sampled) + batch_size - 1) // batch_size)]):
            start = i * batch_size
            end = min(start + batch_size, len(sampled))
            if start < len(sampled):
                batch_df = pd.DataFrame(sampled[start:end])
                batch_df.to_parquet(output_dir / pf.name, index=False)

        return len(all_records), n_sample

    problems_json = input_dir / "problems.json"
    if problems_json.exists():
        return sample_scienceqa_directory(input_dir, output_dir, sample_ratio, seed)

    raise ValueError(f"Unsupported directory format: {input_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sample 50% of questions from datasets")
    parser.add_argument("input", type=Path, help="Input file or directory path")
    parser.add_argument("--output", type=Path, required=True, help="Output file or directory path")
    parser.add_argument("--ratio", type=float, default=0.5, help="Sampling ratio (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    if args.input.is_dir():
        total, sampled = sample_directory(args.input, args.output, args.ratio, args.seed)
    elif args.input.suffix.lower() == ".parquet":
        total, sampled = sample_parquet(args.input, args.output, args.ratio, args.seed)
    elif args.input.suffix.lower() == ".json":
        total, sampled = sample_json(args.input, args.output, args.ratio, args.seed)
    elif args.input.suffix.lower() == ".jsonl":
        total, sampled = sample_jsonl(args.input, args.output, args.ratio, args.seed)
    else:
        raise ValueError(f"Unsupported file format: {args.input.suffix}")

    print(f"Sampled {sampled}/{total} records ({args.ratio*100:.0f}%) from {args.input}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()