import sys

from run_staged_pipeline import main


def add_default_arg(flag: str, value: str | None = None):
    if flag in sys.argv:
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


if __name__ == "__main__":
    add_default_arg("--run-id", "qwen25vl_3b_fp16")
    add_default_arg("--vl-model-name", "Qwen/Qwen2.5-VL-3B-Instruct")
    add_default_arg("--quantization", "none")
    add_default_arg("--device-map", "cuda")
    add_default_arg("--qwen-max-pixels", "401408")
    main()
