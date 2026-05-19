#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/public/sdc/yl_GigaTok_Loss")
REPO = Path("/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation")
PERSIST = ROOT / "gigatok_persist"
PY = "/data/duoduo_25/anaconda3/envs/RiemDiff_img/bin/python"
NPU = "root@139.9.157.110"
NPU_CONTAINER = "node1new"
NPU_PERSIST = "/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist"
OUT = PERSIST / "outputs/overnight_allin"
PREP = OUT / "prepared_inputs"
SEL = OUT / "selection"
LOGROOT = ROOT / f"logs/repair_text_cache_readable_eval_{time.strftime('%Y%m%d_%H%M%S')}"
DONE = LOGROOT / "done"
FAILED = LOGROOT / "failed"
NOTREADY = LOGROOT / "not_ready"
STATE = LOGROOT / "state.json"
LOCK = ROOT / "logs/repair_text_cache_readable_eval.lock"
PID = ROOT / "logs/repair_text_cache_readable_eval.pid"
TEXT_CONFIG = OUT / "configs/ocr_box_gate_v2_conf_proxy.realworld40_5k.cuda.yaml"
SETS = [
    "holdout_v2_readable_balanced_200",
    "holdout_v2_readable_medium_200",
    "readable50_textscenes40_balanced",
    "readable50_textscenes40_medium",
    "readable50_textscenes40_dense_lite",
]
READABLE = [s for s in SETS if s.startswith("readable50_")]
METHODS = [
    "w001",
    "w001_ocrvis_only",
    "w001_hr_only",
    "w001_local_only",
    "w001_hr_local",
    "ocrvis_hr_local",
    "ocr_box_gate_v1",
    "ocr_box_gate_v2_conf_proxy",
]

for d in [LOGROOT, DONE, FAILED, NOTREADY]:
    d.mkdir(parents=True, exist_ok=True)
state = {"stamp": LOGROOT.name, "events": []}


def save():
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def log(msg: str):
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    state.setdefault("events", []).append(line)
    save()


def mark(root: Path, name: str, data):
    p = root / f"{name}.json"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def run(cmd, log_path: Path, cwd: Path | None = None) -> int:
    shell = isinstance(cmd, str)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        display = cmd if shell else " ".join(map(str, cmd))
        f.write(f"\n==== CMD {time.strftime('%F %T')} ====\n{display}\n")
        f.flush()
        p = subprocess.Popen(cmd, shell=shell, cwd=str(cwd) if cwd else None, stdout=f, stderr=subprocess.STDOUT)
        rc = p.wait()
        f.write(f"\n==== EXIT {rc} {time.strftime('%F %T')} ====\n")
        return rc


def live(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def lock():
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
            if live(old):
                print(f"repair already running pid={old}", flush=True)
                sys.exit(0)
        except Exception:
            pass
    LOCK.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    PID.write_text(str(os.getpid()) + "\n", encoding="utf-8")


def sync_glyph_assets():
    target = PERSIST / "models/Glyph-SDXL-v2/assets/color_idx.json"
    if target.exists():
        mark(DONE, "glyph_assets_present", {"path": str(target)})
        return
    log("sync missing Glyph-SDXL-v2 assets from NPU")
    dst = PERSIST / "models/Glyph-SDXL-v2"
    dst.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"ssh {NPU} 'docker exec {NPU_CONTAINER} tar -C {NPU_PERSIST}/models/Glyph-SDXL-v2 -cf - assets' "
        f"| tar -C {dst} -xf -"
    )
    rc = run(cmd, LOGROOT / "sync_glyph_assets.log")
    if rc == 0 and target.exists():
        mark(DONE, "glyph_assets_synced", {"path": str(target)})
    else:
        mark(FAILED, "glyph_assets_sync_failed", {"returncode": rc, "expected": str(target)})
        raise RuntimeError("glyph asset sync failed")


def run_text_caches():
    results = {}
    for i, name in enumerate(SETS):
        feat = PREP / name / "text_feature_cache/text_features.pt"
        manifest = SEL / name / f"{name}.cuda.jsonl"
        if feat.exists():
            results[name] = {"status": "skipped_existing", "path": str(feat)}
            continue
        cmd = [
            PY,
            str(REPO / "scripts/stage1/ocr_debug/cache_text_features.py"),
            "--config",
            str(TEXT_CONFIG),
            "--manifest-jsonl",
            str(manifest),
            "--output-dir",
            str(feat.parent),
            "--device-backend",
            "cuda",
            "--device-id",
            str(i % 5),
            "--batch-size",
            "4",
            "--dtype",
            "fp16",
            "--verify",
        ]
        rc = run(cmd, LOGROOT / f"{name}.text_cache.log", cwd=REPO)
        results[name] = {"status": "done" if rc == 0 and feat.exists() else "failed", "returncode": rc, "path": str(feat)}
        if results[name]["status"] == "failed":
            mark(NOTREADY, f"NOT_READY_{name}_text_cache", results[name])
    mark(DONE, "text_cache_retry_summary", results)
    state["text_cache_retry"] = results
    save()
    return results


def method_paths():
    return {
        "w001": (
            PERSIST / "outputs/ocr_w_sweep_setting_tune_5000/w001/checkpoints/0002500.pt",
            OUT / "method_configs/w001.cuda.yaml",
        ),
        "w001_ocrvis_only": (
            PERSIST / "outputs/ocr_two_new_ablation_readable50/w001_ocrvis_only/checkpoints/0002500.pt",
            OUT / "method_configs/w001_ocrvis_only.cuda.yaml",
        ),
        "w001_hr_only": (
            PERSIST / "outputs/ocr_two_new_ablation_readable50/w001_hr_only/checkpoints/0002500.pt",
            OUT / "method_configs/w001_hr_only.cuda.yaml",
        ),
        "w001_local_only": (
            PERSIST / "outputs/ocr_two_new_ablation_readable50/w001_local_only/checkpoints/0002500.pt",
            OUT / "method_configs/w001_local_only.cuda.yaml",
        ),
        "w001_hr_local": (
            PERSIST / "outputs/ocr_two_new_ablation_readable50/w001_hr_local/checkpoints/0002500.pt",
            OUT / "method_configs/w001_hr_local.cuda.yaml",
        ),
        "ocrvis_hr_local": (
            PERSIST / "outputs/ocr_two_new_ablation_readable50/ocrvis_hr_local/checkpoints/0002500.pt",
            OUT / "method_configs/ocrvis_hr_local.cuda.yaml",
        ),
        "ocr_box_gate_v1": (
            PERSIST / "outputs/ocr_box_gate/pipeline_runs/ocr_box_gate_w001_v1_v2_eval_2500/ocr_box_gate_v1/checkpoints/0002500.pt",
            OUT / "method_configs/ocr_box_gate_v1.cuda.yaml",
        ),
        "ocr_box_gate_v2_conf_proxy": (
            PERSIST / "outputs/ocr_box_gate/pipeline_runs/ocr_box_gate_w001_v1_v2_eval_2500/ocr_box_gate_v2_conf_proxy/checkpoints/0002500.pt",
            OUT / "method_configs/ocr_box_gate_v2_conf_proxy.cuda.yaml",
        ),
    }


def run_readable_eval():
    methods = {}
    for m, (ck, cfg) in method_paths().items():
        if ck.exists() and cfg.exists():
            methods[m] = (ck, cfg)
    if not methods:
        mark(NOTREADY, "NOT_READY_no_readable_methods", {})
        return
    for setname in READABLE:
        manifest = SEL / setname / f"{setname}.cuda.jsonl"
        bbox = PREP / setname / f"{setname}.with_proxy_conf.jsonl"
        feat = PREP / setname / "text_feature_cache/text_features.pt"
        if not (manifest.exists() and bbox.exists() and feat.exists()):
            mark(NOTREADY, f"NOT_READY_{setname}_readable_inputs_after_repair", {
                "manifest": manifest.exists(),
                "bbox": bbox.exists(),
                "text_feature": feat.exists(),
            })
            continue
        od = OUT / "readable50_eval" / setname
        od.mkdir(parents=True, exist_ok=True)
        cmd = [
            PY,
            str(REPO / "scripts/stage1/eval_unified_realworld40_5k.py"),
            "--manifest",
            str(manifest),
            "--bbox-jsonl",
            str(bbox),
            "--output-dir",
            str(od),
            "--mode",
            "holdout",
            "--max-images",
            "0",
            "--batch-size",
            "2",
            "--device-backend",
            "cuda",
            "--device-id",
            "0",
            "--mixed-precision",
            "fp16",
        ]
        for name, (ck, cfg) in methods.items():
            cmd += ["--run", f"{name}:{cfg}:{ck}:{feat}"]
        rc = run(cmd, LOGROOT / f"{setname}.readable_eval.log", cwd=REPO)
        if rc == 0:
            mark(DONE, f"{setname}_readable_eval_done", {"eval_dir": str(od), "methods": sorted(methods)})
        else:
            mark(FAILED, f"{setname}_readable_eval_failed", {"returncode": rc, "log": str(LOGROOT / f"{setname}.readable_eval.log")})


def main():
    lock()
    try:
        log("repair text cache + readable50 eval started")
        sync_glyph_assets()
        run_text_caches()
        run_readable_eval()
        mark(DONE, "repair_finished", {"logroot": str(LOGROOT)})
    finally:
        save()
        try:
            LOCK.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
