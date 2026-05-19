#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path("/public/sdc/yl_GigaTok_Loss")
REPO = Path("/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation")
PERSIST = ROOT / "gigatok_persist"
PY = "/data/duoduo_25/anaconda3/envs/RiemDiff_img/bin/python"
NPU = "root@139.9.157.110"
NPU_CONTAINER = "node1new"
NPU_ROOT = "/data2/duoduo_25/yl_GigaTok_Loss"
NPU_PERSIST = f"{NPU_ROOT}/gigatok_persist"
GPU_PERSIST = str(PERSIST)
STAMP = time.strftime("%Y%m%d_%H%M%S")
LOGROOT = ROOT / f"logs/companion_downstream_{STAMP}"
STATE = LOGROOT / "companion_state.json"
DONE = LOGROOT / "done"
FAILED = LOGROOT / "failed"
NOTREADY = LOGROOT / "not_ready"
OUT = PERSIST / "outputs/overnight_allin"
PREP = OUT / "prepared_inputs"
SEL = OUT / "selection"
LOCK = ROOT / "logs/run_all_in_companion_downstream_worker.lock"
PIDFILE = ROOT / "logs/run_all_in_companion_downstream_worker.pid"

SETS = [
    "holdout_v2_readable_balanced_200",
    "holdout_v2_readable_medium_200",
    "readable50_textscenes40_balanced",
    "readable50_textscenes40_medium",
    "readable50_textscenes40_dense_lite",
]
READABLE = [s for s in SETS if s.startswith("readable50_")]
DEVICES = [0, 1, 2, 3, 4]
DEEPSEEK = PERSIST / "models/DeepSeek-OCR"
TEXT_CONFIG = OUT / "configs/ocr_box_gate_v2_conf_proxy.realworld40_5k.cuda.yaml"

for d in [LOGROOT, DONE, FAILED, NOTREADY, OUT, PREP, SEL]:
    d.mkdir(parents=True, exist_ok=True)

status: Dict[str, Any] = {"stamp": STAMP, "logroot": str(LOGROOT), "events": []}


def save_state() -> None:
    STATE.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def log(msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    status.setdefault("events", []).append(line)
    save_state()


def mark(root: Path, name: str, data: Any) -> Path:
    p = root / f"{name}.json"
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def run(cmd: list[str] | str, log_path: Path | None = None, cwd: Path | None = None, timeout: int | None = None) -> int:
    shell = isinstance(cmd, str)
    if log_path is None:
        return subprocess.run(cmd, shell=shell, cwd=str(cwd) if cwd else None).returncode
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        display = cmd if shell else " ".join(map(str, cmd))
        f.write(f"\n==== CMD {time.strftime('%F %T')} ====\n{display}\n")
        f.flush()
        p = subprocess.Popen(
            cmd,
            shell=shell,
            cwd=str(cwd) if cwd else None,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        rc = p.wait(timeout=timeout)
        f.write(f"\n==== EXIT {rc} {time.strftime('%F %T')} ====\n")
        return rc


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def live_pid(pid: int) -> bool:
    return subprocess.run(["ps", "-p", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def acquire_lock() -> None:
    if LOCK.exists():
        try:
            old = int(LOCK.read_text().strip())
            if live_pid(old):
                print(f"companion already running pid={old}", flush=True)
                sys.exit(0)
        except Exception:
            pass
    LOCK.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    PIDFILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")


def npu_tar_pull(src_abs: str, dst_abs: Path) -> int:
    dst_abs.parent.mkdir(parents=True, exist_ok=True)
    parent = str(Path(src_abs).parent)
    base = Path(src_abs).name
    cmd = (
        f"ssh {NPU} 'docker exec {NPU_CONTAINER} tar -C {parent} -cf - {base}' "
        f"| tar -C {dst_abs.parent} -xf -"
    )
    return run(cmd, LOGROOT / "transfer.log")


def npu_tar_push(src_abs: Path, dst_abs: str) -> int:
    parent = str(src_abs.parent)
    base = src_abs.name
    dst_parent = str(Path(dst_abs).parent)
    cmd = (
        f"ssh {NPU} 'docker exec {NPU_CONTAINER} mkdir -p {dst_parent}' && "
        f"tar -C {parent} -cf - {base} "
        f"| ssh {NPU} 'docker exec -i {NPU_CONTAINER} tar -C {dst_parent} -xf -'"
    )
    return run(cmd, LOGROOT / "transfer.log")


def pull_required() -> None:
    log("sync scripts/config/selection from NPU container")
    files = [
        "repo/GigaTok_Loss_two_ablation/scripts/stage1/eval_unified_realworld40_5k.py",
        "repo/GigaTok_Loss_two_ablation/scripts/stage1/ocr_debug/generate_deepseek_ocr_bbox_jsonl_incremental.py",
        "repo/GigaTok_Loss_two_ablation/scripts/stage1/ocr_debug/probe_deepseek_ocr_token_confidence_incremental.py",
        "repo/GigaTok_Loss_two_ablation/scripts/stage1/ocr_debug/probe_deepseek_ocr_teacher_forced_confidence_incremental.py",
    ]
    for rel in files:
        src = f"{NPU_ROOT}/{rel}"
        dst = Path("/data/duoduo_25/yl_GigaTok_Loss") / rel
        if npu_tar_pull(src, dst) != 0:
            mark(NOTREADY, "script_sync_partial", {"failed": rel})

    npu_tar_pull(f"{NPU_PERSIST}/outputs/overnight_allin/selection", OUT / "selection")

    npu_cfg = (
        f"{NPU_PERSIST}/outputs/ocr_box_gate/realworld40_5k/configs/"
        "ocr_box_gate_v2_conf_proxy.realworld40_5k.newserver.yaml"
    )
    tmp = OUT / "configs/ocr_box_gate_v2_conf_proxy.realworld40_5k.newserver.yaml"
    npu_tar_pull(npu_cfg, tmp)
    if tmp.exists():
        txt = tmp.read_text(encoding="utf-8").replace(NPU_PERSIST, GPU_PERSIST).replace(NPU_ROOT, str(ROOT))
        TEXT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        TEXT_CONFIG.write_text(txt, encoding="utf-8")
    mark(DONE, "sync_required_done", {"selection": str(SEL), "text_config": str(TEXT_CONFIG)})


def rewrite_manifests() -> Dict[str, Any]:
    log("rewrite manifests to CUDA paths and validate images")
    result: Dict[str, Any] = {}
    for name in SETS:
        src = SEL / name / f"{name}.jsonl"
        dst = SEL / name / f"{name}.cuda.jsonl"
        rows = []
        missing = []
        for r in iter_jsonl(src):
            rr = dict(r)
            p = str(rr.get("image_path", ""))
            p = p.replace(NPU_PERSIST, GPU_PERSIST).replace("/home/ma-user/work/GigaTok_hr/gigatok_persist", GPU_PERSIST)
            rr["image_path"] = p
            if not Path(p).exists():
                missing.append(p)
            rows.append(rr)
        write_jsonl(dst, rows)
        result[name] = {
            "rows": len(rows),
            "cuda_manifest": str(dst),
            "missing_count": len(missing),
            "missing_examples": missing[:5],
        }
        if missing:
            mark(NOTREADY, f"NOT_READY_{name}_missing_images", result[name])
    status["manifests"] = result
    mark(DONE, "manifest_rewrite", result)
    return result


def sync_methods() -> Dict[str, Any]:
    log("sync 8 readable50 method ckpts/configs from NPU container")
    rels = {
        "w001": (
            "outputs/ocr_w_sweep_setting_tune_5000/w001/checkpoints/0002500.pt",
            "outputs/ocr_w_sweep_setting_tune_5000/configs/w001.yaml",
        ),
        "w001_ocrvis_only": (
            "outputs/ocr_two_new_ablation_readable50/w001_ocrvis_only/checkpoints/0002500.pt",
            "outputs/ocr_two_new_ablation_readable50/configs/w001_ocrvis_only.yaml",
        ),
        "w001_hr_only": (
            "outputs/ocr_two_new_ablation_readable50/w001_hr_only/checkpoints/0002500.pt",
            "outputs/ocr_two_new_ablation_readable50/configs/w001_hr_only.yaml",
        ),
        "w001_local_only": (
            "outputs/ocr_two_new_ablation_readable50/w001_local_only/checkpoints/0002500.pt",
            "outputs/ocr_two_new_ablation_readable50/configs/w001_local_only.yaml",
        ),
        "w001_hr_local": (
            "outputs/ocr_two_new_ablation_readable50/w001_hr_local/checkpoints/0002500.pt",
            "outputs/ocr_two_new_ablation_readable50/configs/w001_hr_local.yaml",
        ),
        "ocrvis_hr_local": (
            "outputs/ocr_two_new_ablation_readable50/ocrvis_hr_local/checkpoints/0002500.pt",
            "outputs/ocr_two_new_ablation_readable50/configs/ocrvis_hr_local.yaml",
        ),
        "ocr_box_gate_v1": (
            "outputs/ocr_box_gate/pipeline_runs/ocr_box_gate_w001_v1_v2_eval_2500/ocr_box_gate_v1/checkpoints/0002500.pt",
            "outputs/ocr_box_gate/pipeline_runs/ocr_box_gate_w001_v1_v2_eval_2500/configs/ocr_box_gate_v1.yaml",
        ),
        "ocr_box_gate_v2_conf_proxy": (
            "outputs/ocr_box_gate/pipeline_runs/ocr_box_gate_w001_v1_v2_eval_2500/ocr_box_gate_v2_conf_proxy/checkpoints/0002500.pt",
            "outputs/ocr_box_gate/pipeline_runs/ocr_box_gate_w001_v1_v2_eval_2500/configs/ocr_box_gate_v2_conf_proxy.yaml",
        ),
    }
    info: Dict[str, Any] = {}
    for m, (ckrel, cfgrel) in rels.items():
        ck = PERSIST / ckrel
        cfg_raw = PERSIST / cfgrel
        cfg = OUT / "method_configs" / f"{m}.cuda.yaml"
        if not ck.exists():
            npu_tar_pull(f"{NPU_PERSIST}/{ckrel}", ck)
        if not cfg_raw.exists():
            npu_tar_pull(f"{NPU_PERSIST}/{cfgrel}", cfg_raw)
        if cfg_raw.exists():
            txt = (
                cfg_raw.read_text(encoding="utf-8", errors="ignore")
                .replace(NPU_PERSIST, GPU_PERSIST)
                .replace("/home/ma-user/work/GigaTok_hr/gigatok_persist", GPU_PERSIST)
                .replace(NPU_ROOT, str(ROOT))
            )
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text(txt, encoding="utf-8")
        info[m] = {
            "checkpoint": str(ck),
            "checkpoint_exists": ck.exists(),
            "config": str(cfg),
            "config_exists": cfg.exists(),
        }
    mark(
        NOTREADY,
        "NOT_READY_gigatok_only",
        {
            "reason": (
                "same-readable50 gigatok_only checkpoint not synced/found in first-pass NPU source; "
                "8-method ablation will continue if eval stage runs"
            )
        },
    )
    mark(DONE, "method_sync_8", info)
    status["methods"] = info
    return info


def run_text_cache(name: str, manifest: Path, device: int) -> Dict[str, Any]:
    od = PREP / name / "text_feature_cache"
    feat = od / "text_features.pt"
    if feat.exists():
        return {"status": "skipped_existing", "path": str(feat)}
    cmd = [
        PY,
        str(REPO / "scripts/stage1/ocr_debug/cache_text_features.py"),
        "--config",
        str(TEXT_CONFIG),
        "--manifest-jsonl",
        str(manifest),
        "--output-dir",
        str(od),
        "--device-backend",
        "cuda",
        "--device-id",
        str(device),
        "--batch-size",
        "4",
        "--dtype",
        "fp16",
        "--verify",
    ]
    rc = run(cmd, PREP / name / "text_cache.log", cwd=REPO)
    return {"status": "done" if rc == 0 and feat.exists() else "failed", "returncode": rc, "path": str(feat)}


def run_bbox(name: str, manifest: Path, device: int) -> Dict[str, Any]:
    od = PREP / name
    bbox = od / f"{name}.ocr_bbox.jsonl"
    if bbox.exists():
        return {"status": "skipped_existing", "bbox": str(bbox)}
    cmd = [
        PY,
        str(REPO / "scripts/stage1/ocr_debug/generate_deepseek_ocr_bbox_jsonl_incremental.py"),
        "--manifest",
        str(manifest),
        "--model-path",
        str(DEEPSEEK),
        "--output-jsonl",
        str(bbox),
        "--output-dir",
        str(od / "bbox_raw"),
        "--state-path",
        str(od / f"{name}.bbox.state"),
        "--device-backend",
        "cuda",
        "--device-id",
        str(device),
        "--dtype",
        "fp16",
        "--attn-implementation",
        "eager",
        "--allow-empty",
    ]
    rc = run(cmd, od / "bbox.log", cwd=REPO)
    return {"status": "done" if rc == 0 and bbox.exists() else "failed", "returncode": rc, "bbox": str(bbox)}


def run_proxy_raw(name: str, manifest: Path, device: int) -> Dict[str, Any]:
    od = PREP / name
    bbox = od / f"{name}.ocr_bbox.jsonl"
    raw = od / f"{name}.proxy_raw.jsonl"
    if raw.exists():
        return {"status": "skipped_existing", "raw": str(raw)}
    token_script = REPO / "scripts/stage1/ocr_debug/probe_deepseek_ocr_token_confidence_incremental.py"
    teacher_script = REPO / "scripts/stage1/ocr_debug/probe_deepseek_ocr_teacher_forced_confidence_incremental.py"
    cmd = [
        PY,
        str(token_script),
        "--manifest",
        str(manifest),
        "--model-path",
        str(DEEPSEEK),
        "--input-bbox-jsonl",
        str(bbox),
        "--output-jsonl",
        str(raw),
        "--output-dir",
        str(od / "proxy_raw"),
        "--state-path",
        str(od / f"{name}.proxy.state"),
        "--device-backend",
        "cuda",
        "--device-id",
        str(device),
        "--dtype",
        "fp16",
        "--attn-implementation",
        "eager",
    ]
    rc = run(cmd, od / "proxy.log", cwd=REPO)
    mode = "selected_token_generate"
    if rc != 0 or not raw.exists():
        mode = "teacher_forced_token_logprob_fallback"
        cmd = [
            PY,
            str(teacher_script),
            "--manifest",
            str(manifest),
            "--model-path",
            str(DEEPSEEK),
            "--input-bbox-jsonl",
            str(bbox),
            "--output-jsonl",
            str(raw),
            "--output-dir",
            str(od / "proxy_raw_teacher_forced"),
            "--state-path",
            str(od / f"{name}.proxy_teacher_forced.state"),
            "--device-backend",
            "cuda",
            "--device-id",
            str(device),
            "--dtype",
            "fp16",
            "--attn-implementation",
            "eager",
        ]
        rc = run(cmd, od / "proxy_teacher_forced.log", cwd=REPO)
    return {
        "status": "done" if rc == 0 and raw.exists() else "failed",
        "returncode": rc,
        "raw": str(raw),
        "scoring_mode": mode,
    }


def normalize_all(names: List[str]) -> Dict[str, Any]:
    vals = []
    rows_by: Dict[str, Any] = {}
    for name in names:
        raw = PREP / name / f"{name}.proxy_raw.jsonl"
        if not raw.exists():
            continue
        rows = list(iter_jsonl(raw))
        rows_by[name] = rows
        for r in rows:
            for b in r.get("boxes", []):
                if b.get("proxy_conf_valid") and b.get("mean_logprob") is not None and b.get("token_span") is not None:
                    vals.append(float(b["mean_logprob"]))
    if not vals:
        mark(NOTREADY, "NOT_READY_global_proxy_no_valid_boxes", {"sets": names})
        return {}
    vals = sorted(vals)

    def pct(p: float) -> float:
        idx = (len(vals) - 1) * p / 100.0
        lo = int(idx)
        hi = min(len(vals) - 1, lo + 1)
        frac = idx - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac

    p10, p90 = pct(10), pct(90)
    summaries: Dict[str, Any] = {}
    for name, rows in rows_by.items():
        total = valid = failed = 0
        confs = []
        for r in rows:
            r["normalization"] = "robust_p10_p90_from_mean_logprob"
            r["normalization_scope"] = "global_all_valid_boxes_after_merge"
            for b in r.get("boxes", []):
                total += 1
                ok = b.get("proxy_conf_valid") and b.get("mean_logprob") is not None and b.get("token_span") is not None
                if ok:
                    c = max(0.0, min(1.0, (float(b["mean_logprob"]) - p10) / (p90 - p10 + 1e-8)))
                    valid += 1
                else:
                    c = 0.0
                    b["proxy_conf_valid"] = False
                    failed += 1
                b["confidence"] = c
                b["confidence_source"] = "deepseek_token_logprob_proxy"
                b["normalization"] = "robust_p10_p90_from_mean_logprob"
                b["normalization_scope"] = "global_all_valid_boxes_after_merge"
                confs.append(c)
        final = PREP / name / f"{name}.with_proxy_conf.jsonl"
        write_jsonl(final, rows)
        summ = {
            "rows": len(rows),
            "boxes_total": total,
            "boxes_with_proxy_conf": valid,
            "failed_alignment_count": failed,
            "proxy_conf_valid_ratio": valid / max(1, total),
            "logprob_p10": p10,
            "logprob_p90": p90,
            "confidence_min": min(confs) if confs else None,
            "confidence_mean": sum(confs) / len(confs) if confs else None,
            "confidence_max": max(confs) if confs else None,
            "confidence_unique_count": len(set(round(x, 6) for x in confs)),
            "normalization": "robust_p10_p90_from_mean_logprob",
            "normalization_scope": "global_all_valid_boxes_after_merge",
            "global_scope_sets": names,
            "no_minmax": True,
            "no_shard_local_confidence": True,
            "no_fake_default_confidence": True,
        }
        (PREP / name / f"{name}.proxy_summary.json").write_text(
            json.dumps(summ, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        summaries[name] = summ
        if summ["proxy_conf_valid_ratio"] < 0.90:
            mark(NOTREADY, f"NOT_READY_{name}_proxy_ratio", summ)
    mark(DONE, "global_proxy_normalization", summaries)
    status["proxy_summary"] = summaries
    return summaries


def prepare_inputs() -> Dict[str, Any]:
    manifests = {n: SEL / n / f"{n}.cuda.jsonl" for n in SETS}
    results: Dict[str, Any] = {}

    def prep_one(i_name):
        i, name = i_name
        dev = DEVICES[i % len(DEVICES)]
        manifest = manifests[name]
        r: Dict[str, Any] = {"manifest": str(manifest), "device": dev}
        r["text_cache"] = run_text_cache(name, manifest, dev)
        r["bbox"] = run_bbox(name, manifest, dev)
        if r["bbox"].get("status") != "failed":
            r["proxy_raw"] = run_proxy_raw(name, manifest, dev)
        return name, r

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(SETS), len(DEVICES))) as ex:
        for name, r in ex.map(prep_one, enumerate(SETS)):
            results[name] = r
            status["prepare_inputs"] = results
            save_state()
    normalize_all(SETS)
    for name in SETS:
        od = PREP / name
        if od.exists():
            npu_dst = f"{NPU_PERSIST}/outputs/overnight_allin/prepared_inputs/{name}"
            npu_tar_push(od, npu_dst)
    mark(DONE, "prepare_inputs_done", results)
    return results


def run_readable50_eval(methods: Dict[str, Any]) -> None:
    usable = {m: v for m, v in methods.items() if v.get("checkpoint_exists") and v.get("config_exists")}
    if not usable:
        mark(NOTREADY, "NOT_READY_readable50_no_methods", methods)
        return
    for setname in READABLE:
        manifest = SEL / setname / f"{setname}.cuda.jsonl"
        bbox = PREP / setname / f"{setname}.with_proxy_conf.jsonl"
        feat = PREP / setname / "text_feature_cache/text_features.pt"
        if not manifest.exists() or not bbox.exists() or not feat.exists():
            mark(
                NOTREADY,
                f"NOT_READY_{setname}_readable50_inputs",
                {"manifest": manifest.exists(), "bbox": bbox.exists(), "text_feature": feat.exists()},
            )
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
        for m, v in usable.items():
            cmd += ["--run", f"{m}:{v['config']}:{v['checkpoint']}:{feat}"]
        rc = run(cmd, od / "readable50_eval.log", cwd=REPO)
        if rc == 0:
            mark(DONE, f"{setname}_readable50_eval_done", {"eval_dir": str(od), "methods": list(usable)})
        else:
            mark(FAILED, f"{setname}_readable50_eval_failed", {"returncode": rc, "log": str(od / "readable50_eval.log")})


def main() -> None:
    acquire_lock()
    log("companion downstream worker started on 3090")
    try:
        pull_required()
        manifests = rewrite_manifests()
        methods = sync_methods()
        bad = [n for n, v in manifests.items() if v.get("missing_count")]
        if bad:
            mark(NOTREADY, "NOT_READY_missing_images_block_prepare", {"sets": bad})
            return
        prepare_inputs()
        run_readable50_eval(methods)
        mark(DONE, "companion_downstream_finished", {"logroot": str(LOGROOT), "prepared_inputs": str(PREP)})
    except Exception as e:
        mark(FAILED, "companion_exception", {"error": repr(e)})
        raise
    finally:
        save_state()
        try:
            LOCK.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
