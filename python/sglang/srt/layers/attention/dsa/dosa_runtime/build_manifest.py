"""Build Full or Few service dependencies and write an explicit native manifest.

The build uses the packaged source and the caller's CUDA/Torch toolchain. It
does not construct a backend, reserve model memory, or run a numerical kernel.
"""
import argparse
import json
from pathlib import Path
from .native_binding import load_modules, manifest_from_modules


def build_manifest(build_directory, output, *, mode="full", verbose=False, score_heads=32):
    """Optionally export an immutable binding for controlled reproductions."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new immutable manifest path: {output}")
    heads = {} if score_heads == 32 else {"score_heads": score_heads}
    modules = load_modules(mode, service=True, build_directory=build_directory,
                           verbose=verbose, **heads)
    result = manifest_from_modules(modules, mode=mode, **heads)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a concurrent build from replacing an existing
    # manifest. Start the service only after this command completes successfully.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "few"), default="full")
    parser.add_argument("--score-heads", choices=(32, 64), type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = build_manifest(
        args.build_directory, args.output, mode=args.mode, verbose=args.verbose,
        score_heads=args.score_heads,
    )
    print(json.dumps({"manifest": str(args.output.resolve()),
                      "mode": args.mode,
                      "modules": {key: entry["path"] for key, entry in result["modules"].items()},
                      "numerical_kernel_run": False}, indent=2))


if __name__ == "__main__":
    main()
