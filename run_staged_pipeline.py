import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from run_pipeline import (
    format_cuda_memory,
    generate_summary,
    load_qwen_model,
    load_translation_model,
    translate_en_to_bn,
)


RESULT_FIELDS = [
    "image_id",
    "image_name",
    "image_path",
    "split",
    "family",
    "numeric_id",
    "english_summary",
    "bangla_summary",
    "status",
    "error",
    "seconds",
    "completed_at",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume-safe staged chart summarization for large local datasets."
    )
    parser.add_argument(
        "--manifest",
        default="manifests/full_dataset_manifest.csv",
        help="Manifest CSV created by scripts/build_manifest.py.",
    )
    parser.add_argument(
        "--run-id",
        default="qwen25vl_full",
        help="Run folder name under --runs-dir.",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Parent folder for staged run state and part files.",
    )
    parser.add_argument(
        "--stage-size",
        type=int,
        default=500,
        help="Maximum pending images to process in this invocation.",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Optional split filter. Can be repeated, e.g. --split valid --split test.",
    )
    parser.add_argument(
        "--family",
        action="append",
        default=[],
        help="Optional family filter. Can be repeated, e.g. --family k --family s.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Select failed rows as well as pending rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the next stage without loading models or writing results.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print state counts and exit.",
    )
    parser.add_argument(
        "--vl-model-name",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Vision-language model name or local path.",
    )
    parser.add_argument(
        "--quantization",
        choices=["bnb4", "none"],
        default="bnb4",
        help="Qwen quantization mode.",
    )
    parser.add_argument(
        "--max-gpu-memory",
        default="7GiB",
        help="Per-GPU memory budget for --device-map auto.",
    )
    parser.add_argument(
        "--device-map",
        choices=["auto", "cuda"],
        default="cuda",
        help="Use auto for GPU/CPU placement or cuda to force Qwen onto GPU 0.",
    )
    parser.add_argument(
        "--qwen-min-pixels",
        type=int,
        default=128 * 28 * 28,
        help="Minimum visual token pixel budget.",
    )
    parser.add_argument(
        "--qwen-max-pixels",
        type=int,
        default=512 * 28 * 28,
        help="Maximum visual token pixel budget. Use 401408 for the tested 8 GB path.",
    )
    parser.add_argument(
        "--translate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Bangla translations after English summaries.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=250,
        help="Max new tokens for each Qwen summary.",
    )
    return parser.parse_args()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_schema(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            image_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            split TEXT,
            family TEXT,
            numeric_id INTEGER,
            size_bytes INTEGER,
            width INTEGER,
            height INTEGER,
            sha256 TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS run_parts (
            part_index INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.commit()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def import_manifest_if_needed(conn: sqlite3.Connection, manifest_path: Path):
    existing = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    if existing:
        return
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run scripts/build_manifest.py first."
        )

    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                (
                    row["image_id"],
                    row["image_name"],
                    row["image_path"],
                    row.get("split", ""),
                    row.get("family", ""),
                    int(row["numeric_id"]) if row.get("numeric_id") else None,
                    int(row["size_bytes"]) if row.get("size_bytes") else None,
                    int(row["width"]) if row.get("width") else None,
                    int(row["height"]) if row.get("height") else None,
                    row.get("sha256", ""),
                    now_iso(),
                )
            )

    conn.executemany(
        """
        INSERT INTO images (
            image_id, image_name, image_path, split, family, numeric_id,
            size_bytes, width, height, sha256, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    print(f"Imported {len(rows)} manifest rows into staged state.")


def requeue_interrupted_rows(conn: sqlite3.Connection):
    conn.execute(
        """
        UPDATE images
        SET status = 'pending', last_error = 'Requeued after interrupted run', updated_at = ?
        WHERE status = 'running'
        """,
        (now_iso(),),
    )
    conn.commit()


def state_counts(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT split, family, status, COUNT(*) AS count
        FROM images
        GROUP BY split, family, status
        ORDER BY split, family, status
        """
    ).fetchall()


def print_status(conn: sqlite3.Connection):
    rows = state_counts(conn)
    if not rows:
        print("No rows in staged state.")
        return
    for row in rows:
        print(f"{row['split']}_{row['family']}\t{row['status']}\t{row['count']}")


def select_stage(conn: sqlite3.Connection, args) -> list[sqlite3.Row]:
    params: list[object] = []
    statuses = ["pending"]
    if args.retry_failed:
        statuses.append("failed")
    where = [f"status IN ({','.join('?' for _ in statuses)})"]
    params.extend(statuses)

    if args.split:
        where.append(f"split IN ({','.join('?' for _ in args.split)})")
        params.extend(args.split)
    if args.family:
        where.append(f"family IN ({','.join('?' for _ in args.family)})")
        params.extend(args.family)

    query = f"""
        SELECT *
        FROM images
        WHERE {' AND '.join(where)}
        ORDER BY
            CASE split
                WHEN 'valid' THEN 0
                WHEN 'test' THEN 1
                WHEN 'train' THEN 2
                ELSE 3
            END,
            family,
            numeric_id,
            image_name
        LIMIT ?
    """
    params.append(args.stage_size)
    return conn.execute(query, params).fetchall()


def next_part_path(conn: sqlite3.Connection, parts_dir: Path) -> tuple[int, Path]:
    current = conn.execute("SELECT COALESCE(MAX(part_index), 0) FROM run_parts").fetchone()[0]
    part_index = current + 1
    return part_index, parts_dir / f"part_{part_index:06d}.csv"


def write_config(run_dir: Path, args):
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        return
    config = vars(args).copy()
    config["created_at"] = now_iso()
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def append_result(part_path: Path, row: dict):
    file_exists = part_path.exists()
    with part_path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def mark_running(conn: sqlite3.Connection, image_id: str):
    conn.execute(
        """
        UPDATE images
        SET status = 'running', attempts = attempts + 1, last_error = '', updated_at = ?
        WHERE image_id = ?
        """,
        (now_iso(), image_id),
    )
    conn.commit()


def mark_final(conn: sqlite3.Connection, image_id: str, status: str, error: str = ""):
    conn.execute(
        """
        UPDATE images
        SET status = ?, last_error = ?, updated_at = ?
        WHERE image_id = ?
        """,
        (status, error, now_iso(), image_id),
    )
    conn.commit()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    run_dir = Path(args.runs_dir) / args.run_id
    parts_dir = run_dir / "parts"
    db_path = run_dir / "state.sqlite"
    run_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    write_config(run_dir, args)

    conn = connect(db_path)
    ensure_schema(conn)
    import_manifest_if_needed(conn, manifest_path)
    requeue_interrupted_rows(conn)

    if args.status:
        print_status(conn)
        return

    stage_rows = select_stage(conn, args)
    if not stage_rows:
        print("No pending rows match the requested filters.")
        print_status(conn)
        return

    print(f"Selected {len(stage_rows)} rows for this stage.")
    print(f"First: {stage_rows[0]['image_name']}")
    print(f"Last:  {stage_rows[-1]['image_name']}")
    if args.dry_run:
        for row in stage_rows[:20]:
            print(f"{row['image_id']}\t{row['image_path']}")
        if len(stage_rows) > 20:
            print(f"... {len(stage_rows) - 20} more")
        return

    print(f"CUDA memory before loading Qwen: {format_cuda_memory()}")
    processor, qwen_model = load_qwen_model(
        args.vl_model_name,
        args.quantization,
        args.max_gpu_memory,
        args.device_map,
        args.qwen_min_pixels,
        args.qwen_max_pixels,
    )
    trans_tokenizer = trans_model = None
    if args.translate:
        trans_tokenizer, trans_model = load_translation_model()

    part_index, part_path = next_part_path(conn, parts_dir)
    conn.execute(
        "INSERT INTO run_parts (part_index, path, created_at, row_count) VALUES (?, ?, ?, 0)",
        (part_index, str(part_path), now_iso()),
    )
    conn.commit()

    done_count = 0
    progress = tqdm(stage_rows, desc=f"Stage {part_index}", unit="img")
    for row in progress:
        started = perf_counter()
        image_id = row["image_id"]
        mark_running(conn, image_id)
        result = {
            "image_id": image_id,
            "image_name": row["image_name"],
            "image_path": row["image_path"],
            "split": row["split"],
            "family": row["family"],
            "numeric_id": row["numeric_id"],
            "english_summary": "",
            "bangla_summary": "",
            "status": "done",
            "error": "",
            "seconds": "",
            "completed_at": "",
        }
        try:
            with Image.open(row["image_path"]) as img:
                image = img.convert("RGB")
                english_summary = generate_summary(
                    processor, qwen_model, image, max_new_tokens=args.max_new_tokens
                )
            bangla_summary = ""
            if args.translate:
                bangla_summary = translate_en_to_bn(
                    trans_tokenizer, trans_model, english_summary
                )
            result["english_summary"] = english_summary
            result["bangla_summary"] = bangla_summary
            result["seconds"] = f"{perf_counter() - started:.2f}"
            result["completed_at"] = now_iso()
            append_result(part_path, result)
            mark_final(conn, image_id, "done")
            done_count += 1
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result["status"] = "failed"
            result["error"] = error
            result["seconds"] = f"{perf_counter() - started:.2f}"
            result["completed_at"] = now_iso()
            append_result(part_path, result)
            mark_final(conn, image_id, "failed", error)
        conn.execute(
            "UPDATE run_parts SET row_count = row_count + 1 WHERE part_index = ?",
            (part_index,),
        )
        conn.commit()
        progress.set_postfix({"done": done_count, "part": part_index})

    print(f"Wrote stage results to {part_path}")
    print_status(conn)


if __name__ == "__main__":
    main()
