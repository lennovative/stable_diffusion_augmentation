"""
SD 1.5 attention-guided image editing.

Usage:
    python main.py              # uses config.ini in current directory
    python main.py my_run.ini   # uses a specific config file
"""

import shutil
import sys
import configparser
from pathlib import Path
import torch
from diffusers import DDIMScheduler, DDIMInverseScheduler

from sd_editing import load_sd15_edit_pipe, run_batch_inversion_and_editing
from sd_editing.sam_mask import load_grounded_sam


def _parse_tuple_of_strings(value):
    return tuple(p.strip() for p in value.split(",") if p.strip())


def _parse_float_pair(value):
    a, b = value.split(",")
    return (float(a.strip()), float(b.strip()))


def load_config(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def next_run_dir(base_dir):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    next_idx = max(existing, default=-1) + 1
    run_dir = base / f"{next_idx:03d}"
    run_dir.mkdir()
    return run_dir


def load_concepts(path):
    concepts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            name, rest = line.split("=", 1)
            template, tokens_str = rest.split("|", 1)
            concepts[name.strip()] = (
                template.strip(),
                [t.strip() for t in tokens_str.split(",")],
            )
    return concepts


def load_prompts(path):
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.ini"
    cfg = load_config(config_path)

    g = cfg["generation"]
    p1 = cfg["pass1"]
    p2 = cfg["pass2"]

    pipe, _ = load_sd15_edit_pipe(
        model_name=cfg["model"]["name"],
        device=cfg["model"]["device"],
        dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    grounded_sam = None
    if g["base_mask_source"] == "sam":
        s = cfg["sam"]
        grounded_sam = load_grounded_sam(
            grounding_dino_model_id=s["grounding_dino_model"],
            sam_model_id=s["sam_model"],
            device=cfg["model"]["device"],
            box_threshold=s.getfloat("box_threshold"),
            text_threshold=s.getfloat("text_threshold"),
            edge_blur_radius=s.getint("edge_blur_radius"),
        )

    run_dir = next_run_dir(cfg["paths"]["output_dir"])
    shutil.copy(config_path, run_dir / "config.ini")
    print(f"Run directory: {run_dir}")

    results = run_batch_inversion_and_editing(
        pipe=pipe,
        base_dir=cfg["paths"]["base_dir"],
        concept_targets=load_concepts(cfg["paths"]["concepts"]),
        edit_prompts=load_prompts(cfg["paths"]["prompts"]),
        output_dir=run_dir,

        # generation
        num_inference_steps=g.getint("num_inference_steps"),
        input_size=g.getint("input_size"),
        attention_res=g.getint("attention_res"),
        allowed_places=_parse_tuple_of_strings(g["allowed_places"]),
        base_mask_step_range=_parse_float_pair(g["base_mask_step_range"]),
        invert_mask=g.getboolean("invert_mask"),
        multi_token_merge=g["multi_token_merge"],
        base_mask_source=g["base_mask_source"],
        grounded_sam=grounded_sam,
        save_debug_every=g.getint("save_debug_every"),
        save_debug_latents=g.getboolean("save_debug_latents"),
        save_inversion_pickle=g.getboolean("save_inversion_pickle"),

        # pass 1
        inversion_prompt_mode=p1["inversion_prompt_mode"],
        inversion_guidance_scale=p1.getfloat("inversion_guidance_scale"),
        edit_guidance_scale=p1.getfloat("guidance_scale"),
        eta=p1.getfloat("eta"),
        invert_frac=p1.getfloat("invert_frac"),
        use_inversion_attention_transmission=p1.getboolean("use_inversion_attention_transmission"),
        use_reconstruction_attention_transmission=p1.getboolean("use_reconstruction_attention_transmission"),
        transmission_alpha=p1.getfloat("transmission_alpha"),
        initial_noise_beta=p1.getfloat("initial_noise_beta"),
        base_mask_erode_radius=p1.getint("base_mask_erode_radius"),
        recon_dilate_radius=p1.getint("recon_dilate_radius"),
        transition_gap_radius=p1.getint("transition_gap_radius"),
        alpha_decay_start=p1.getfloat("alpha_decay_start"),
        recon_alpha_decay=p1.getboolean("recon_alpha_decay"),
        recon_attn_start_frac=p1.getfloat("recon_attn_start_frac"),

        # pass 2
        second_pass_polish=p2.getboolean("enabled"),
        polish_invert_frac=p2.getfloat("invert_frac"),
        polish_inversion_prompt_mode=p2["inversion_prompt_mode"],
        polish_inversion_guidance_scale=p2.getfloat("inversion_guidance_scale"),
        polish_reconstruction_prompt_mode=p2["reconstruction_prompt_mode"],
        polish_guidance_scale=p2.getfloat("guidance_scale"),
        polish_eta=p2.getfloat("eta"),
        polish_use_inversion_attention_transmission=p2.getboolean("use_inversion_attention_transmission"),
        polish_use_reconstruction_attention_transmission=p2.getboolean("use_reconstruction_attention_transmission"),
        polish_transmission_alpha=p2.getfloat("transmission_alpha"),
        polish_initial_noise_beta=p2.getfloat("initial_noise_beta"),
        polish_base_mask_erode_radius=p2.getint("base_mask_erode_radius"),
        polish_recon_dilate_radius=p2.getint("recon_dilate_radius"),
        polish_transition_gap_radius=p2.getint("transition_gap_radius"),
        polish_alpha_decay_start=p2.getfloat("alpha_decay_start"),
        polish_recon_alpha_decay=p2.getboolean("recon_alpha_decay"),
        polish_recon_attn_start_frac=p2.getfloat("recon_attn_start_frac"),
        save_pre_polish=p2.getboolean("save_pre_polish"),
    )

    print(f"\nDone. {len(results)} images edited → {run_dir}")


if __name__ == "__main__":
    main()
