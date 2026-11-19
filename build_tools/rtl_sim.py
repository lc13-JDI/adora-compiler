#!/usr/bin/env python3
"""Build a byte-locked ADORA RTL target and run a named cocotb regression."""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("cstore", "intvecadd")
CASES = {"short-tail": "cstore", "ownership": "cstore",
         "vector-add": "intvecadd", "tcp-client": "intvecadd", "tcp-server": "intvecadd"}
VERILATOR_FLAGS = ["--cc", "--exe", "--threads", "1", "--timescale", "1ns/1ns",
                   "--vpi", "--public-flat-rw", "--prefix", "Vtop", "-o", "Vtop",
                   "-DCOCOTB_SIM=1", "-Wno-SELRANGE", "-Wno-WIDTH", "-Wno-CASEINCOMPLETE"]
MAKE_FLAGS = ["OPT_FAST=-O0", "OPT_SLOW=-O0", "OPT_GLOBAL=-O0"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def validated(target, assets, source=None):
    spec = json.loads((ROOT / "test/spec/rtl_sim" / target / "target.json").read_text())
    paths = {}
    for role, entry in spec["files"].items():
        if "repository_path" in entry:
            path = ROOT / entry["repository_path"]
            supplied = (source or assets / target) / path.name
            if supplied.is_file() and sha256(supplied) != entry["sha256"]:
                raise ValueError(f"{target} {role}: SHA-256 mismatch for supplied {supplied}")
        else:
            path = (source or assets / target) / entry["asset"]
        if not path.is_file():
            raise ValueError(f"{target} {role}: missing {path}, expected SHA-256 {entry['sha256']}")
        digest = sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"{target} {role}: SHA-256 mismatch for {path}, expected "
                             f"{entry['sha256']}, got {digest}")
        paths[role] = path
    return spec, paths


def which(name):
    exe = shutil.which(name)
    if not exe:
        raise ValueError(f"missing executable: {name}")
    return str(Path(exe).absolute())


def command_version(exe):
    return subprocess.check_output([exe, "--version"], text=True).splitlines()[0]


def build_model(spec, paths, cache, jobs):
    import cocotb.config
    if not cocotb.__version__.startswith("1.9."):
        raise ValueError(f"cocotb 1.9.x required, found {cocotb.__version__}")
    driver = Path(cocotb.config.share_dir) / "lib/verilator/verilator.cpp"
    libs = Path(cocotb.config.libs_dir)
    plugin = libs / "libcocotbvpi_verilator.so"
    verilator, make, cxx, ar = (which(name) for name in ("verilator", "make", os.environ.get("CXX", "c++"), "ar"))
    ldflags = f"-Wl,-rpath,{libs} -L{libs} -lcocotbvpi_verilator"
    vflags = [*VERILATOR_FLAGS, "--top-module", spec["top"], "-LDFLAGS", ldflags]
    vflags += [f"-G{k}={v}" for k, v in spec["parameters"].items()]
    mflags = [*MAKE_FLAGS, f"CXX={cxx}", f"LINK={cxx}", f"AR={ar}"]
    env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LD_LIBRARY_PATH",
                                        "SYSTEMROOT") if k in os.environ}
    identity = {"format": 1, "spec": spec, "verilator_flags": vflags, "make_flags": mflags,
                "tools": {p: command_version(p) for p in (verilator, make, cxx, ar)},
                "machine": platform.machine(), "python": sys.executable,
                "cocotb": cocotb.__version__, "driver_sha256": sha256(driver),
                "plugin_sha256": sha256(plugin)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    with (cache / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        binary = destination / "Vtop"
        if destination.exists():
            receipt = json.loads((destination / "model.json").read_text())
            if receipt["identity"] != identity or sha256(binary) != receipt["binary_sha256"]:
                raise ValueError(f"model cache integrity failure: {destination}")
            print(f"model cache hit {key}", flush=True)
            return binary, receipt, True
        print(f"building {spec['target']} model {key}", flush=True)
        with tempfile.TemporaryDirectory(dir=cache, prefix=key[:10] + "-") as scratch:
            stage = Path(scratch)
            obj = stage / "obj"
            commands = [[verilator, *vflags, "--Mdir", str(obj), str(paths["wrapper"]),
                         str(paths["rtl"]), str(driver)],
                        [make, "-C", str(obj), "-f", "Vtop.mk", f"-j{jobs}", *mflags]]
            log_path = cache / (key + ".build.log")
            with log_path.open("w") as log:
                for cmd in commands:
                    print(shlex.join(cmd), file=log, flush=True)
                    result = subprocess.run(cmd, cwd=stage, env=env, stdout=log,
                                            stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise ValueError(f"model build exited {result.returncode}, see {log_path}")
            complete = stage / "model"
            complete.mkdir()
            (obj / "Vtop").rename(complete / "Vtop")
            receipt = {"key": key, "identity": identity, "commands": commands,
                       "binary_sha256": sha256(complete / "Vtop")}
            save(complete / "model.json", receipt)
            complete.rename(destination)
    return binary, receipt, False


def import_assets(target, assets, source):
    """Import an unpacked immutable artifact directory, then re-check installed bytes."""
    spec, _ = validated(target, assets, source)
    assets.mkdir(parents=True, exist_ok=True)
    with (assets / (target + ".import.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (assets / target).exists():
            with tempfile.TemporaryDirectory(dir=assets, prefix=target + "-") as scratch:
                stage = Path(scratch) / target
                stage.mkdir()
                for entry in spec["files"].values():
                    if "asset" in entry:
                        shutil.copyfile(source / entry["asset"], stage / entry["asset"])
                validated(target, Path(scratch))
                stage.rename(assets / target)
    return validated(target, assets)


def run(args, target, paths, binary, receipt, hit, case_env=None):
    import find_libpython
    import site

    runtime_dir = ROOT / "runtime/python"
    for name in ("test_runif.py", "rtl_device.py"):
        if not (runtime_dir / name).is_file():
            raise ValueError(f"missing repository simulation runtime: {runtime_dir / name}")
    if args.output:
        work = args.output
        work.mkdir(parents=True)
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=args.output_root, prefix=args.case + "-"))
    print(f"running {args.case} in {work}", flush=True)
    vector = target == "intvecadd"
    bench = ROOT / "test/rtl" / ("vector" if vector else "tail")
    module = "vector_bench" if vector else "tail_bench"
    if case_env is None and vector:
        case_env = {"TESTCASE": args.case.replace("-", "_")}
    elif case_env is None:
        sys.path.insert(0, str(bench))
        prepare = importlib.import_module("tail_prepare").prepare
        case_env = prepare(paths, work, args.build_dir, args.cgeist, args.case)
    env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LD_LIBRARY_PATH", "CXX")
           if k in os.environ}
    env.update({f"ADORA_RTL_{k.upper()}": str(p) for k, p in paths.items()})
    env.update({f"PARAM_{k}": str(v) for k, v in receipt["identity"]["spec"]["parameters"].items()})
    libs = str(Path(sys.prefix) / "lib")
    env.update(
        ADORA_RTL_TARGET=target, ADORA_RTL_CASE=args.case, ADORA_RTL_PORT=str(args.port),
        ADORA_RTL_WORK=str(work), ADORA_RTL_OUTPUT=str(work), PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONHOME=sys.base_prefix,
        PYTHONPATH=os.pathsep.join([str(runtime_dir), str(bench), *site.getsitepackages()]),
        LD_LIBRARY_PATH=os.pathsep.join(filter(None, [libs, env.get("LD_LIBRARY_PATH")])),
        LIBPYTHON_LOC=find_libpython.find_libpython() or "", PYGPI_PYTHON_BIN=sys.executable,
        TOPLEVEL=receipt["identity"]["spec"]["top"], TOPLEVEL_LANG="verilog", MODULE=module,
        COCOTB_RESULTS_FILE=str(work / "results.xml"), COCOTB_LOG_LEVEL="INFO",
        COCOTB_RESOLVE_X="ZEROS", RANDOM_SEED="7", **case_env)
    record = {"case": args.case, "target": target, "cache_hit": hit, "model": str(binary),
              "key": receipt["key"], "binary_sha256": receipt["binary_sha256"],
              "source_paths": {k: str(v) for k, v in paths.items()},
              "python": sys.executable, "module": module, "pythonpath": env["PYTHONPATH"],
              "runtime": str(runtime_dir / "test_runif.py"),
              "libpython": env["LIBPYTHON_LOC"], "testcase": env["TESTCASE"]}
    save(work / "run.json", record)
    with (work / "simulator.log").open("w") as log:
        proc = subprocess.run([str(binary)], cwd=work, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    xml = work / "results.xml"
    if proc.returncode or not xml.is_file():
        raise ValueError(f"RTL run failed (exit {proc.returncode}), see {log.name}")
    cases = list(ET.parse(xml).iter("testcase"))
    executed = [tc.get("name") for tc in cases]
    if (sorted(executed) != sorted(env["TESTCASE"].split(",")) or
            any(any(child.tag in ("skipped", "failure", "error") for child in tc) for tc in cases)):
        raise ValueError(f"incomplete or failed RTL tests {executed}, see {xml} and {log.name}")
    print(f"PASS {args.case}: {', '.join(executed)}")


def absolute(value):
    return Path(value).expanduser().resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("import", "verify", "build", "run"):
        cmd = subs.add_parser(name)
        cmd.add_argument("--assets", type=absolute, required=True, help="offline target asset store")
        if name == "run":
            cmd.add_argument("--case", choices=CASES, required=True)
            dest = cmd.add_mutually_exclusive_group(required=True)
            dest.add_argument("--output", type=absolute, help="new run directory")
            dest.add_argument("--output-root", type=absolute, help="parent of a unique run directory")
            cmd.add_argument("--build-dir", type=absolute, default=ROOT / "build")
            cmd.add_argument("--cgeist", type=absolute)
            cmd.add_argument("--port", type=int, default=0)
        else:
            cmd.add_argument("--target", choices=TARGETS, required=True)
        if name == "import":
            cmd.add_argument("--from-dir", type=absolute, required=True,
                             help="directory containing the original files named by target.json")
        if name in ("build", "run"):
            cmd.add_argument("--cache", type=absolute, default=ROOT / "build/rtl-models")
            cmd.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    target = CASES[args.case] if args.command == "run" else args.target
    if args.command == "import":
        _, paths = import_assets(target, args.assets, args.from_dir)
    else:
        spec, paths = validated(target, args.assets)
    if args.command in ("verify", "import"):
        print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
        return
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    binary, receipt, hit = build_model(spec, paths, args.cache, args.jobs)
    if args.command == "run":
        run(args, target, paths, binary, receipt, hit)
    else:
        print(binary)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, ValueError, OSError, RuntimeError, subprocess.SubprocessError, ET.ParseError) as error:
        sys.exit(f"rtl_sim: {error}")
