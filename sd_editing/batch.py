from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from .inversion import ddim_invert_store
from .editing import reconstruct_ddim_with_attention_restoration
from .pipeline import load_image_rgb


def format_prompt_template(prompt, target_attribute=None):
    if prompt is None:
        return ""
    prompt = str(prompt)
    if "{}" in prompt and target_attribute is not None:
        return prompt.format(target_attribute)
    return prompt


def resolve_prompt_mode(mode, edit_prompt, custom_prompt="", target_attribute=None):
    """
    mode:
        "edit"   -> use the editing prompt from pass 1
        "custom" -> use custom_prompt, with optional {} formatting
        "empty"  -> use ""
    """
    mode = str(mode).lower().strip()
    if mode == "edit":
        return "" if edit_prompt is None else str(edit_prompt)
    if mode == "custom":
        return format_prompt_template(custom_prompt, target_attribute=target_attribute)
    if mode == "empty":
        return ""
    raise ValueError("Prompt mode must be one of: 'edit', 'custom', 'empty'.")


def run_batch_inversion_and_editing(
    pipe,
    base_dir: str,
    concept_targets: Dict[str, Tuple[str, List[str]]],
    edit_prompts: Iterable[str],
    output_dir: str,
    image_extensions=(".png", ".jpg", ".jpeg", ".webp", ".bmp"),

    # shared generation params
    num_inference_steps: int = 50,
    inversion_guidance_scale: float = 1.0,
    edit_guidance_scale: float = 7.5,
    input_size: int = 512,
    attention_res: int = 32,
    allowed_places=("up",),
    base_mask_step_range=(0.0, 0.5),
    invert_mask: bool = False,
    eta: float = 1.0,
    multi_token_merge: str = "average",
    base_mask_source: str = "attention",    # "attention" | "sam"
    grounded_sam=None,                       # from sam_mask.load_grounded_sam()
    save_debug_every: int = 5,
    save_debug_latents: bool = False,
    save_inversion_pickle: bool = False,

    # pass 1
    inversion_prompt_mode: str = "auto",    # "auto" (a photo of <subject>) | "empty"
    invert_frac: float = 0.9,
    use_inversion_attention_transmission: bool = True,
    use_reconstruction_attention_transmission: bool = True,
    transmission_alpha: float = 1.0,
    initial_noise_beta: float = 0.75,
    base_mask_erode_radius: int = 0,
    recon_dilate_radius: int = 2,
    transition_gap_radius: int = 0,
    alpha_decay_start: float = 0.5,

    # pass 2 / polish
    second_pass_polish: bool = True,
    polish_invert_frac: float = 0.7,

    polish_inversion_prompt_mode: str = "edit",
    polish_inversion_prompt: str = "",
    polish_inversion_guidance_scale: float = 1.0,

    polish_reconstruction_prompt_mode: str = "edit",
    polish_reconstruction_prompt: str = "",
    polish_guidance_scale: Optional[float] = None,
    polish_eta: float = 0.0,

    polish_use_inversion_attention_transmission: bool = True,
    polish_use_reconstruction_attention_transmission: bool = True,
    polish_transmission_alpha: float = 0.8,
    polish_initial_noise_beta: float = 0.0,
    polish_base_mask_erode_radius: int = 1,
    polish_recon_dilate_radius: int = 2,
    polish_transition_gap_radius: int = 0,
    polish_alpha_decay_start: float = 0.5,

    save_pre_polish: bool = True,
) -> List[dict]:
    """
    Process all images in base_dir/<concept>/ for every concept listed in
    concept_targets, applying each edit_prompt.

    concept_targets format:
        {
            "dog":     ("cute fluffy dog",  ["dog"]),
            "backpack": ("red backpack",    ["red", "backpack"]),
        }
        template is a plain-text description; every token must appear in it literally.

    edit_prompts format:
        [
            "a photo of a {} in the jungle",
            "a photo of a {} in a snowstorm",
        ]

    Returns a list of dicts with paths and metadata for each edited image.
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edit_prompts = list(edit_prompts)

    def _get_target_spec(folder_name):
        spec = concept_targets[folder_name]
        if not isinstance(spec, (tuple, list)) or len(spec) != 2:
            raise ValueError(f"concept_targets['{folder_name}'] must be (template, tokens)")
        template, tokens = str(spec[0]), spec[1]
        if isinstance(tokens, str):
            tokens = [tokens]
        tokens = [str(t) for t in tokens]
        missing = [t for t in tokens if t not in template]
        if missing:
            raise ValueError(
                f"concept_targets['{folder_name}']: tokens {missing} not found in template {template!r}"
            )
        return template, tokens, template

    def _list_images(folder):
        return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in image_extensions)

    def _safe_name(text, max_len=120):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in text)
        while "__" in safe:
            safe = safe.replace("__", "_")
        safe = safe.strip("._")
        return safe[:max_len]

    summary = []

    for subfolder in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        folder_name = subfolder.name
        if folder_name not in concept_targets:
            print(f"[SKIP] Folder '{folder_name}' not in concept_targets.")
            continue

        _, tokens, target_attribute = _get_target_spec(folder_name)
        image_paths = _list_images(subfolder)
        if not image_paths:
            print(f"[SKIP] No images in {subfolder}")
            continue

        print(f"\n=== {folder_name} | tokens={tokens} | target={target_attribute} | {len(image_paths)} images ===")

        concept_out = output_dir / folder_name
        concept_out.mkdir(parents=True, exist_ok=True)

        inv_prompt = "" if inversion_prompt_mode == "empty" else f"a photo of {target_attribute}"

        for image_path in image_paths:
            stem = _safe_name(image_path.stem)
            source_image = load_image_rgb(str(image_path), size=(input_size, input_size))
            print(f"[INV:PASS1] {image_path.name}  prompt={inv_prompt!r}")

            inv = ddim_invert_store(
                pipe=pipe,
                image=source_image,
                prompt=inv_prompt,
                tokens=tokens,
                num_inference_steps=num_inference_steps,
                invert_frac=invert_frac,
                guidance_scale=inversion_guidance_scale,
                input_size=input_size,
                attention_res=attention_res,
                allowed_places=allowed_places,
                capture_attention=True,
                allow_missing_token=False,
                multi_token_merge=multi_token_merge,
            )

            if save_inversion_pickle:
                torch.save(inv, concept_out / f"{stem}__inversion.pt")

            for prompt_idx, prompt_template in enumerate(edit_prompts):
                edit_prompt = prompt_template.format(target_attribute)
                prompt_name = _safe_name(f"{prompt_idx:02d}_{edit_prompt}")
                file_prefix = f"{stem}__{prompt_name}"

                debug_root = concept_out / f"{file_prefix}__debug"
                pass1_debug = debug_root / "pass1"
                pass2_debug = debug_root / "pass2_polish"

                print(f"[EDIT:PASS1] {image_path.name}  prompt={edit_prompt!r}")

                edited = reconstruct_ddim_with_attention_restoration(
                    pipe=pipe,
                    latents_all=inv["latents_all"],
                    attns_all=inv["attns_all"],
                    prompt=edit_prompt,
                    tokens=tokens,
                    guidance_scale=edit_guidance_scale,
                    num_inference_steps=num_inference_steps,
                    attention_res=attention_res,
                    allowed_places=allowed_places,
                    base_mask_step_range=base_mask_step_range,
                    invert_mask=invert_mask,
                    base_mask_erode_radius=base_mask_erode_radius,
                    eta=eta,
                    use_inversion_attention_transmission=use_inversion_attention_transmission,
                    use_reconstruction_attention_transmission=use_reconstruction_attention_transmission,
                    transmission_alpha=transmission_alpha,
                    initial_noise_beta=initial_noise_beta,
                    recon_dilate_radius=recon_dilate_radius,
                    transition_gap_radius=transition_gap_radius,
                    alpha_decay_start=alpha_decay_start,
                    multi_token_merge=multi_token_merge,
                    base_mask_source=base_mask_source,
                    source_image=source_image,
                    grounded_sam=grounded_sam,
                    debug_dir=str(pass1_debug),
                    save_debug_every=save_debug_every,
                    save_debug_latents=save_debug_latents,
                )

                pre_polish_path = None
                if save_pre_polish:
                    pre_polish_path = concept_out / f"{file_prefix}__pre_polish.png"
                    edited.save(pre_polish_path)

                if second_pass_polish:
                    p_inv_prompt = resolve_prompt_mode(
                        polish_inversion_prompt_mode,
                        edit_prompt=edit_prompt,
                        custom_prompt=polish_inversion_prompt,
                        target_attribute=target_attribute,
                    )
                    p_recon_prompt = resolve_prompt_mode(
                        polish_reconstruction_prompt_mode,
                        edit_prompt=edit_prompt,
                        custom_prompt=polish_reconstruction_prompt,
                        target_attribute=target_attribute,
                    )

                    print(
                        f"[INV:PASS2] {image_path.name}  "
                        f"frac={polish_invert_frac}  prompt={p_inv_prompt!r}"
                    )
                    polish_inv = ddim_invert_store(
                        pipe=pipe,
                        image=edited,
                        prompt=p_inv_prompt,
                        tokens=tokens,
                        num_inference_steps=num_inference_steps,
                        invert_frac=polish_invert_frac,
                        guidance_scale=polish_inversion_guidance_scale,
                        input_size=input_size,
                        attention_res=attention_res,
                        allowed_places=allowed_places,
                        capture_attention=polish_use_inversion_attention_transmission,
                        allow_missing_token=True,
                        multi_token_merge=multi_token_merge,
                    )

                    print(
                        f"[EDIT:PASS2] {image_path.name}  "
                        f"recon_prompt={p_recon_prompt!r}  alpha={polish_transmission_alpha}"
                    )
                    edited = reconstruct_ddim_with_attention_restoration(
                        pipe=pipe,
                        latents_all=polish_inv["latents_all"],
                        attns_all=inv["attns_all"],  # use original inversion attention maps
                        prompt=p_recon_prompt,
                        tokens=tokens,
                        guidance_scale=edit_guidance_scale if polish_guidance_scale is None else polish_guidance_scale,
                        num_inference_steps=num_inference_steps,
                        attention_res=attention_res,
                        allowed_places=allowed_places,
                        base_mask_step_range=base_mask_step_range,
                        invert_mask=invert_mask,
                        base_mask_erode_radius=polish_base_mask_erode_radius,
                        eta=polish_eta,
                        use_inversion_attention_transmission=polish_use_inversion_attention_transmission,
                        use_reconstruction_attention_transmission=polish_use_reconstruction_attention_transmission,
                        transmission_alpha=polish_transmission_alpha,
                        initial_noise_beta=polish_initial_noise_beta,
                        recon_dilate_radius=polish_recon_dilate_radius,
                        transition_gap_radius=polish_transition_gap_radius,
                        alpha_decay_start=polish_alpha_decay_start,
                        multi_token_merge=multi_token_merge,
                        base_mask_source=base_mask_source,
                        source_image=source_image,
                        grounded_sam=grounded_sam,
                        debug_dir=str(pass2_debug),
                        save_debug_every=save_debug_every,
                        save_debug_latents=save_debug_latents,
                    )

                edited_path = concept_out / f"{file_prefix}.png"
                edited.save(edited_path)

                # write metadata
                debug_root.mkdir(parents=True, exist_ok=True)
                with open(debug_root / "metadata.txt", "w", encoding="utf-8") as f:
                    f.write(f"folder_name: {folder_name}\n")
                    f.write(f"image_path: {image_path}\n")
                    f.write(f"tokens: {tokens}\n")
                    f.write(f"multi_token_merge: {multi_token_merge}\n")
                    f.write(f"target_attribute: {target_attribute}\n")
                    f.write(f"edit_prompt: {edit_prompt}\n\n")
                    f.write(f"[PASS1]\n")
                    f.write(f"inversion_prompt: {inv_prompt}\n")
                    f.write(f"num_inference_steps: {num_inference_steps}\n")
                    f.write(f"inversion_guidance_scale: {inversion_guidance_scale}\n")
                    f.write(f"edit_guidance_scale: {edit_guidance_scale}\n")
                    f.write(f"input_size: {input_size}\n")
                    f.write(f"attention_res: {attention_res}\n")
                    f.write(f"allowed_places: {allowed_places}\n")
                    f.write(f"eta: {eta}\n")
                    f.write(f"transmission_alpha: {transmission_alpha}\n")
                    f.write(f"initial_noise_beta: {initial_noise_beta}\n\n")
                    f.write(f"[PASS2_POLISH]\n")
                    f.write(f"enabled: {second_pass_polish}\n")
                    if second_pass_polish:
                        f.write(f"polish_invert_frac: {polish_invert_frac}\n")
                        f.write(f"polish_guidance_scale: {edit_guidance_scale if polish_guidance_scale is None else polish_guidance_scale}\n")
                        f.write(f"polish_transmission_alpha: {polish_transmission_alpha}\n")

                summary.append({
                    "folder_name": folder_name,
                    "image_path": str(image_path),
                    "tokens": tokens,
                    "target_attribute": target_attribute,
                    "edit_prompt": edit_prompt,
                    "edited_path": str(edited_path),
                    "pre_polish_path": str(pre_polish_path) if pre_polish_path else None,
                    "debug_root": str(debug_root),
                })

    return summary
