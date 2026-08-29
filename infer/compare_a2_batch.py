#!/usr/bin/env python3
"""Load Wan-Animate-2 once, then generate a list of jobs. Run from infer/."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT / "infer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--config", default="./wan_animate_2_distillation.yaml")
    args = parser.parse_args()

    from core import build_object_from_config_file

    jobs = json.loads(Path(args.jobs).read_text())
    print(f"[a2_batch] loading pipeline from {args.config}, n_jobs={len(jobs)}", flush=True)
    t_load = time.time()
    pipeline = build_object_from_config_file(args.config)
    load_s = time.time() - t_load
    print(f"[a2_batch] pipeline loaded in {load_s:.1f}s", flush=True)

    metrics = []
    for i, job in enumerate(jobs):
        out_dir = Path(job["out_dir"])
        dest = out_dir / "03_wan_animate_2.mp4"
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"[a2_batch] skip existing {dest}", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt = job["prompt"]
        if isinstance(prompt, str):
            prompt = [prompt]
        prompt_ref = job.get("prompt_ref", "人物动作的参考视频")
        if isinstance(prompt_ref, str):
            prompt_ref = [prompt_ref]

        print(f"[a2_batch] ({i+1}/{len(jobs)}) {out_dir}", flush=True)
        import gc
        from wanxiang.ops.device import empty_cache
        gc.collect()
        empty_cache()
        t0 = time.time()
        pipeline(
            refer_img_path=job["refer"],
            tpl_video_path=job["video"],
            output_path=str(out_dir),
            width=int(job.get("width", 720)),
            height=int(job.get("height", 1280)),
            fps=int(job.get("fps", 16)),
            seed=int(job.get("seed", 42)),
            clip_len=int(job.get("clip_len", 81)),
            sample_guide_scale=float(job.get("sample_guide_scale", 1.0)),
            step=int(job.get("step", 10)),
            prompt=prompt,
            prompt_ref=prompt_ref,
        )
        gen_s = time.time() - t0
        src = out_dir / "results.mp4"
        if not src.exists():
            raise FileNotFoundError(f"missing {src}")
        shutil.move(str(src), str(dest))
        rec = {
            "out_dir": str(out_dir),
            "generate_s": round(gen_s, 2),
            "load_s": round(load_s, 2) if i == 0 else 0.0,
            "includes_load": i == 0,
            "width": int(job.get("width", 720)),
            "height": int(job.get("height", 1280)),
            "clip_len": int(job.get("clip_len", 81)),
            "step": int(job.get("step", 10)),
        }
        metrics.append(rec)
        (out_dir / "a2_timing.json").write_text(json.dumps(rec, indent=2))
        print(f"[a2_batch] done {dest} in {gen_s:.1f}s", flush=True)

    Path(args.jobs).with_suffix(".a2_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[a2_batch] all done", flush=True)


if __name__ == "__main__":
    main()
