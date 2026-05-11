import sys

from run_pipeline import main


def add_default_arg(flag: str, value: str | None = None):
    if flag in sys.argv:
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


if __name__ == "__main__":
    add_default_arg("--quantization", "none")
    add_default_arg("--device-map", "cuda")
    add_default_arg("--limit", "1")
    main()
