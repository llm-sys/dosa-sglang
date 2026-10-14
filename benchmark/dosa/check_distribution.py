"""Check DoSA's source payload in a standard wheel, without importing CUDA code."""
import argparse
import email
import json
from pathlib import Path
import posixpath
import re
import subprocess
import zipfile

RUNTIME = "sglang/srt/layers/attention/dsa/dosa_runtime/"
PRIMARY = "sglang/kernels/jit/csrc/attention/dosa_primary/"
LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"\n]+)"', re.MULTILINE)


def check_distribution(wheel, source_root, expected_version):
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "python/" + RUNTIME, "python/" + PRIMARY],
        cwd=source_root,
    ).decode().split("\0")
    expected = {name.removeprefix("python/"): (source_root / name).read_bytes()
                for name in tracked if name}
    errors = []
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = email.message_from_bytes(archive.read(metadata_name))
        if metadata["Name"] != "sglang" or metadata["Version"] != expected_version:
            errors.append(f"Expected sglang {expected_version}, got {metadata['Name']} {metadata['Version']}")
        payload = {name: archive.read(name) for name in names
                   if name.startswith((RUNTIME, PRIMARY)) and not name.endswith("/")}
        for name, contents in expected.items():
            if payload.get(name) != contents:
                errors.append("Missing or changed source: " + name)
        for name in payload.keys() - expected.keys():
            errors.append("Unexpected source payload: " + name)
        includes = []
        for name, contents in payload.items():
            if Path(name).suffix not in (".cu", ".cuh", ".inl"):
                continue
            for target in LOCAL_INCLUDE.findall(contents.decode()):
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
                includes.append(dict(source=name, include=target, resolved=resolved))
                if not resolved.startswith("sglang/") or resolved not in names:
                    errors.append(f"Missing local include: {name} -> {target}")
        private = [name for name in payload if Path(name).name in
                   ("DIRECT_ABS_PLANNER.json", "KEY_RHO_NATIVE.json") or "__pycache__" in name]
        if private:
            errors.append("Private pins or bytecode found: " + ", ".join(sorted(private)))
    return dict(status="PASS" if not errors else "FAIL", wheel=str(wheel),
                distribution_version=metadata["Version"], expected_version=expected_version,
                runtime_files=sum(name.startswith(RUNTIME) for name in payload),
                primary_files=sum(name.startswith(PRIMARY) for name in payload),
                source_files_compared=len(expected), local_include_edges=len(includes),
                errors=errors, include_edges=includes,
                scope="Python/native-source packaging only; external toolchain, AOT selector and Rust are separate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-version", default="0.5.18")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_distribution(args.wheel.resolve(), args.source_root.resolve(), args.expected_version)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "include_edges"}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
