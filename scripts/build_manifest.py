import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a manifest for a local chart-image dataset."
    )
    parser.add_argument(
        "--input",
        default="chart images",
        help="Folder containing chart images.",
    )
    parser.add_argument(
        "--output",
        default="manifests/full_dataset_manifest.csv",
        help="Output CSV manifest path.",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip SHA-256 hashing. Faster, but duplicate detection is unavailable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for quick manifest smoke tests.",
    )
    return parser.parse_args()


def parse_name(path: Path) -> tuple[str, str, int | None]:
    parts = path.stem.split("_")
    split = parts[0] if parts else ""
    family = parts[1] if len(parts) >= 3 else ""
    numeric_id = int(parts[-1]) if parts and parts[-1].isdigit() else None
    return split, family, numeric_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_header(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format or ""
    return width, height, image_format


def main():
    args = parse_args()
    input_root = Path(args.input)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    image_paths = sorted(
        path for path in input_root.rglob("*") if path.suffix.lower() in IMAGE_EXTS
    )
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No images found under {input_root}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "image_name",
        "image_path",
        "split",
        "family",
        "numeric_id",
        "extension",
        "size_bytes",
        "width",
        "height",
        "image_format",
        "sha256",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, image_path in enumerate(tqdm(image_paths, desc="Building manifest"), start=1):
            split, family, numeric_id = parse_name(image_path)
            width, height, image_format = read_image_header(image_path)
            writer.writerow(
                {
                    "image_id": f"{split}_{family}_{numeric_id}"
                    if numeric_id is not None and split and family
                    else image_path.stem,
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "split": split,
                    "family": family,
                    "numeric_id": numeric_id if numeric_id is not None else "",
                    "extension": image_path.suffix.lower(),
                    "size_bytes": image_path.stat().st_size,
                    "width": width,
                    "height": height,
                    "image_format": image_format,
                    "sha256": "" if args.no_hash else sha256_file(image_path),
                }
            )

    print(f"Wrote {len(image_paths)} rows to {output_path}")


if __name__ == "__main__":
    main()
