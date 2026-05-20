import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .pipeline import encode_prompt_cfg, decode_latents_to_pil
from .attention import (
    StepTokenAttentionRecorder,
    resolve_tokens_positions,
    register_attention_recorder,
    restore_attention_processors,
)
from .sam_mask import grounded_sam_mask
from .masks import (
    avg_pool_blur,
    otsu_threshold,
    preprocess_mask,
    dilate_mask,
    erode_mask,
    build_base_mask_from_inversion_attn,
    keep_largest_component,
    binary_from_mask,
)


def _replace_tokens_in_prompt(prompt, tokens, generic_token):
    result = prompt
    for token in tokens:
        result = re.sub(r"\b" + re.escape(token) + r"\b", generic_token, result)
    # collapse consecutive identical generic tokens into one
    pattern = r"\b(" + re.escape(generic_token) + r")(\s+\1)+"
    result = re.sub(pattern, r"\1", result)
    return result


def _has_any_attention_maps(attns_all):
    return any(a is not None for _, a in attns_all)


@torch.no_grad()
def reconstruct_ddim_with_attention_restoration(
    pipe,
    latents_all,
    attns_all,
    prompt,
    tokens,
    guidance_scale=7.5,
    num_inference_steps=50,
    start_from_step=None,
    input_size=512,
    attention_res=32,
    allowed_places=("mid",),
    base_mask_step_range=(0.0, 0.5),
    invert_mask=False,
    base_mask_erode_radius=0,
    eta=0.0,
    multi_token_merge="average",

    # external mask source
    base_mask_source="attention",   # "attention" | "sam"
    source_image=None,              # PIL image required when base_mask_source="sam"
    grounded_sam=None,              # dict from sam_mask.load_grounded_sam()

    # transmission controls
    use_inversion_attention_transmission=True,
    use_reconstruction_attention_transmission=True,
    transmission_alpha=1.0,
    transmission_alpha_end=0.0,
    initial_noise_beta=0.75,
    recon_dilate_radius=2,
    recon_blur_k=5,
    transition_gap_radius=0,
    alpha_decay_start=0.5,
    recon_alpha_decay=False,
    recon_attn_start_frac=0.0,
    token_replace_frac=0.0,
    token_replace_generic="subject",
    concept_template=None,   # if set, replace this whole phrase instead of individual tokens

    # asymmetric noise schedule (SDEdit-inspired)
    asymmetric_schedule=False,
    obj_start_frac=0.4,
    bg_start_frac=0.75,
    transmission_ramp_steps=3,
    z0=None,             # clean image latent tensor (required when asymmetric_schedule=True)
    recon_attn_end_frac=1.0,       # stop recon attention transmission after this fraction of denoising steps
    dual_recon_transmission=False, # run a second UNet pass with generic prompt; blend generic latents where its attention leaks
    transmission_source="inversion",  # ring source: "inversion" (lat_orig) | "noise" (q(z_t|z_0_bg), colour-neutral, active full run)
    init_latent="composed",           # denoising start latent: "composed" (SDEdit z_init) | "inversion" (lat at t_bg)
    z0_sdedit=None,                   # preprocessed z0 for SDEdit background (grayscale/blur); replaces z0 in SDEdit formula

    # debug
    debug_dir=None,
    save_debug_every=1,
    save_debug_latents=True,
):
    """
    Unified DDIM reconstruction / editing pass.

    Blends inversion latents back into the denoising trajectory using
    attention-derived masks (from inversion and/or reconstruction).

    Args:
        latents_all   : output of ddim_invert_store["latents_all"]
        attns_all     : output of ddim_invert_store["attns_all"]
        prompt        : editing prompt
        token         : subject token (e.g. "dog") for attention localisation
        transmission_alpha : global strength of the latent blending (0–1)
        initial_noise_beta : randomisation applied outside the subject mask
        debug_dir     : if set, saves per-step debug images here
    """

    def _ensure_odd(k):
        k = max(1, int(k))
        return k + 1 if k % 2 == 0 else k

    def _cast(x):
        return x.to(device=device, dtype=dtype)

    def _to_2d(x):
        if x is None:
            return None
        if x.ndim == 4:
            return x[0, 0]
        if x.ndim == 3:
            return x[0]
        return x

    def _save_map_png(x, path, size=None, normalize=False):
        if x is None:
            return
        x = _to_2d(x).detach().float()
        if size is not None:
            x = F.interpolate(x[None, None], size=size, mode="bilinear", align_corners=False)[0, 0]
        x = (x - x.min()) / (x.max() + 1e-8) if normalize else x.clamp(0, 1)
        arr = (x.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(path)

    def _save_latent_preview(latent, path):
        if not save_debug_latents:
            return
        try:
            img = decode_latents_to_pil(pipe, latent.to(device=device, dtype=dtype))
            img.save(path)
        except Exception as e:
            print(f"[WARN] Could not save latent preview {path}: {e}")

    def _binary_mask_otsu(mask_4d):
        thr = otsu_threshold(mask_4d[0, 0])
        return (mask_4d >= thr).float(), thr

    dtype = next(pipe.unet.parameters()).dtype
    device = pipe.device

    if isinstance(tokens, str):
        tokens = [tokens]

    prompt = "" if prompt is None else str(prompt)
    transmission_alpha = float(max(0.0, min(1.0, transmission_alpha)))
    initial_noise_beta = float(max(0.0, min(1.0, initial_noise_beta)))

    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = list(pipe.scheduler.timesteps)
    timestep_ints = [int(t.item()) for t in timesteps]

    latents_map = {int(t): l for (t, l) in latents_all}
    inv_attn_map = {int(t): a for (t, a) in attns_all}

    if start_from_step is None:
        t_start = int(latents_all[-1][0])
    else:
        start_from_step = max(0, min(int(start_from_step), len(latents_all) - 1))
        t_start = int(latents_all[start_from_step][0])

    if t_start not in timestep_ints:
        nearest_index = min(range(len(timestep_ints)), key=lambda i: abs(timestep_ints[i] - t_start))
        print(f"[WARN] Start timestep {t_start} not found; using nearest {timestep_ints[nearest_index]}")
        start_index = nearest_index
    else:
        start_index = timestep_ints.index(t_start)

    # ── asymmetric schedule: compute t_obj, t_bg, and override start_index ───
    do_asym = asymmetric_schedule and z0 is not None
    if asymmetric_schedule and z0 is None:
        print("[WARN] asymmetric_schedule=True but z0 not provided. Falling back to standard schedule.")

    t_obj_int = None     # timestep at which inversion stopped (object anchor)
    t_bg_int = None      # denoising start timestep (background SDEdit level)
    t_obj_denoising_idx = None  # index in timestep_ints for t_obj
    asym_start_index = None

    if do_asym:
        n_obj = max(1, min(int(round(obj_start_frac * num_inference_steps)), num_inference_steps - 1))
        n_bg  = max(n_obj + 1, min(int(round(bg_start_frac * num_inference_steps)), num_inference_steps))

        # In the denoising direction, timestep_ints goes from high→low.
        # The n-th inversion step corresponds to index (num_inference_steps - n) in denoising.
        t_obj_denoising_idx = num_inference_steps - n_obj   # index where t ≈ t_obj
        asym_start_index    = num_inference_steps - n_bg    # index where t ≈ t_bg (denoising start)

        # clamp to valid range
        t_obj_denoising_idx = max(0, min(t_obj_denoising_idx, num_inference_steps - 1))
        asym_start_index    = max(0, min(asym_start_index, t_obj_denoising_idx - 1))

        t_obj_int = timestep_ints[t_obj_denoising_idx]
        t_bg_int  = timestep_ints[asym_start_index]

        start_index = asym_start_index
        print(f"[ASYM] n_obj={n_obj}  n_bg={n_bg}  t_obj={t_obj_int}  t_bg={t_bg_int}  start_index={start_index}")

    latents = latents_all[-1][1].to(device=device, dtype=dtype).clone()
    latent_spatial = (latents.shape[-2], latents.shape[-1])

    # ── build inversion-derived mask ──────────────────────────────────────────
    inv_maps_only = [a for (_, a) in attns_all]
    have_inv_attention = _has_any_attention_maps(attns_all)

    _zeros = torch.zeros(1, 1, latent_spatial[0], latent_spatial[1])
    base_mask_raw_2d = torch.zeros(attention_res, attention_res)
    base_mask_soft = _zeros.clone()
    main_mask_bin = _zeros.clone()
    main_mask = _zeros.clone()
    inv_otsu_thr = None

    want_transmission = use_inversion_attention_transmission or use_reconstruction_attention_transmission
    have_sam = base_mask_source == "sam" and grounded_sam is not None and source_image is not None
    need_base_mask = want_transmission and (have_inv_attention or have_sam)

    if need_base_mask:
        if have_sam:
            sam_bin = grounded_sam_mask(source_image, tokens, grounded_sam, latent_spatial)
            main_mask = _cast(sam_bin)
            if base_mask_erode_radius > 0:
                main_mask = _cast(keep_largest_component(erode_mask(main_mask, radius=base_mask_erode_radius)))
            base_mask_soft = main_mask.clone()
            main_mask_bin = main_mask.clone()
        else:
            base_mask_raw_2d, base_mask_soft, main_mask = build_base_mask_from_inversion_attn(
                inv_maps_only,
                step_frac_range=base_mask_step_range,
                target_size=latent_spatial,
                invert=invert_mask,
                erode_radius=base_mask_erode_radius,
                final_dilate_radius=2 if not invert_mask else 0,
            )
            base_mask_soft = _cast(base_mask_soft)
            main_mask = _cast(main_mask)

            if base_mask_soft.shape[-2:] != latent_spatial:
                base_mask_soft = F.interpolate(base_mask_soft, size=latent_spatial, mode="bilinear", align_corners=False)
                base_mask_soft = _cast(base_mask_soft)

            if main_mask.shape[-2:] != latent_spatial:
                main_mask = F.interpolate(main_mask, size=latent_spatial, mode="nearest")
                main_mask = _cast(main_mask)

            main_mask_bin = main_mask.clone()
            main_mask = base_mask_soft * main_mask_bin
    else:
        base_mask_soft = _cast(base_mask_soft)
        main_mask_bin = _cast(main_mask_bin)
        main_mask = _cast(main_mask)

        if want_transmission and not have_inv_attention and not have_sam:
            print("[WARN] Transmission requested but no attention maps and no SAM predictor. Disabling.")

    # ── initial latent: asymmetric z_init or standard noise injection ─────────
    _z0_for_bg: torch.Tensor | None = None
    _eps_for_bg: torch.Tensor | None = None
    _alphas_cumprod_dev: torch.Tensor | None = None

    if do_asym:
        z_0_dev = _cast(z0)
        alphas_cumprod = _cast(pipe.scheduler.alphas_cumprod)
        alpha_bar_bg  = alphas_cumprod[t_bg_int]
        alpha_bar_obj = alphas_cumprod[t_obj_int]
        eps = torch.randn_like(z_0_dev)

        # Preprocessed z0 for the background SDEdit component (colour-neutral source).
        z_0_bg = _cast(z0_sdedit) if z0_sdedit is not None else z_0_dev

        if init_latent == "inversion":
            # Use the real DDIM inversion latent at t_bg directly.
            latents = latents_map[t_bg_int].to(device=device, dtype=dtype) \
                if t_bg_int in latents_map else latents
            print(f"[ASYM] init_latent=inversion  using lat at t_bg={t_bg_int}")
        else:
            # Compose z_init from two branches sharing the same noise sample.
            # Object region: forward-noise z_inv(t_obj) → t_bg via DDPM re-noising.
            # Background region: SDEdit forward-noise z_0_bg → t_bg.
            z_inv_obj = latents_map[t_obj_int].to(device=device, dtype=dtype) \
                if t_obj_int in latents_map else latents
            z_bg_noised = alpha_bar_bg.sqrt() * z_0_bg + (1.0 - alpha_bar_bg).sqrt() * eps
            ratio = (alpha_bar_bg / alpha_bar_obj.clamp(min=1e-8)).clamp(max=1.0)
            z_obj_noised = ratio.sqrt() * z_inv_obj + (1.0 - ratio).clamp(min=0).sqrt() * eps
            M_inv_bc = main_mask.expand(1, latents.shape[1], latent_spatial[0], latent_spatial[1]).clamp(0, 1)
            latents = M_inv_bc * z_obj_noised + (1.0 - M_inv_bc) * z_bg_noised

        # keep z_0_bg/eps/alphas alive for per-step q(z_t|z_0) when transmission_source="noise"
        if transmission_source == "noise":
            _z0_for_bg = z_0_bg
            _eps_for_bg = eps
            _alphas_cumprod_dev = alphas_cumprod

        if need_base_mask and initial_noise_beta > 0.0:
            bg_randn = torch.randn_like(latents)
            M_bg = (1.0 - main_mask).clamp(0, 1)
            scale = (1.0 - initial_noise_beta ** 2) ** 0.5
            latents = (1.0 - M_bg) * latents + M_bg * (scale * latents + initial_noise_beta * bg_randn)

        if debug_dir is not None:
            # always save z_init regardless of save_debug_latents — it's a one-time sanity check
            try:
                img = decode_latents_to_pil(pipe, latents.to(device=device, dtype=dtype))
                img.save(os.path.join(debug_dir, "z_init_asymmetric.png"))
            except Exception as e:
                print(f"[WARN] Could not save z_init preview: {e}")

    elif use_inversion_attention_transmission and need_base_mask and initial_noise_beta > 0.0:
        randn = torch.randn_like(latents)
        scale = (1.0 - initial_noise_beta ** 2) ** 0.5
        latents = main_mask * latents + (1.0 - main_mask) * (scale * latents + initial_noise_beta * randn)

    # ── reconstruction attention recorder ────────────────────────────────────
    recorder = None
    if use_reconstruction_attention_transmission and prompt != "":
        all_positions = resolve_tokens_positions(pipe, prompt, tokens)
        valid = [(t, pos) for t, pos in zip(tokens, all_positions) if pos]
        missing = [t for t, pos in zip(tokens, all_positions) if not pos]
        if missing:
            print(
                f"[WARN] Tokens {missing} not found in reconstruction prompt. Skipping missing tokens."
            )
        if not valid:
            print("[WARN] No valid tokens found in reconstruction prompt. Disabling reconstruction attention transmission.")
        else:
            recorder = StepTokenAttentionRecorder(
                tokens_positions=[pos for _, pos in valid],
                out_res=attention_res,
                keep_cond_only=(guidance_scale > 1.0),
                allowed_places=allowed_places,
                multi_token_merge=multi_token_merge,
            )
    elif use_reconstruction_attention_transmission and prompt == "":
        print("[WARN] Reconstruction attention transmission requested but prompt is empty. Disabling.")

    saved_attn_processors = None
    if recorder is not None:
        saved_attn_processors = register_attention_recorder(pipe, recorder, allowed_places=allowed_places)

    prompt_embeds = _cast(encode_prompt_cfg(pipe, prompt, guidance_scale=guidance_scale))

    prompt_embeds_generic = None
    if (token_replace_frac > 0.0 or dual_recon_transmission) and tokens and prompt:
        replace_targets = [concept_template] if concept_template else tokens
        replaced_prompt = _replace_tokens_in_prompt(prompt, replace_targets, token_replace_generic)
        if replaced_prompt != prompt:
            if token_replace_frac > 0.0:
                print(f"[TOKEN_REPLACE] frac={token_replace_frac}  prompt: {prompt!r} → {replaced_prompt!r}")
            if dual_recon_transmission:
                print(f"[DUAL_RECON] generic prompt: {replaced_prompt!r}")
            prompt_embeds_generic = _cast(encode_prompt_cfg(pipe, replaced_prompt, guidance_scale=guidance_scale))
        else:
            print(f"[TOKEN_REPLACE/DUAL_RECON] No concept template matched in {prompt!r}. Generic embedding disabled.")

    try:
        for step_index in range(start_index, num_inference_steps):
            progress = float(step_index) / float(num_inference_steps)
            t = timesteps[step_index]
            t_int = int(t.item())

            lat_orig = latents_map.get(t_int)
            if lat_orig is not None:
                lat_orig = lat_orig.to(device=device, dtype=dtype)

            # In the asymmetric SDEdit phase (t > t_obj), we don't use inversion latents.
            in_sdedit_phase = do_asym and (t_int > t_obj_int)

            # Object anchor always uses the real inversion latent.
            lat_obj_source = lat_orig

            # Ring source: colour-neutral noise formula for the entire run when
            # transmission_source="noise", so background positions are never
            # anchored to source colours regardless of phase.
            if transmission_source == "noise" and _z0_for_bg is not None:
                ab_t = _alphas_cumprod_dev[t_int]
                lat_ring_source = ab_t.sqrt() * _z0_for_bg + (1.0 - ab_t).sqrt() * _eps_for_bg
            else:
                lat_ring_source = lat_orig

            need_lat_orig = (not in_sdedit_phase) and transmission_alpha > 0.0 and (
                use_inversion_attention_transmission and need_base_mask
            )
            if need_lat_orig and lat_orig is None:
                raise KeyError(f"Missing inversion latent for timestep {t_int}")

            # ── stage 1: pre-UNet blend ───────────────────────────────────────
            mask = main_mask.clamp(0, 1) if need_base_mask else torch.zeros_like(main_mask)
            mask = _cast(mask)

            if progress <= alpha_decay_start:
                alpha_t = 1.0
            else:
                t_frac = max(0.0, 1.0 - (progress - alpha_decay_start) / max(1.0 - alpha_decay_start, 1e-8))
                alpha_t = transmission_alpha_end + (1.0 - transmission_alpha_end) * t_frac

            # Linear ramp at the t_obj boundary — kept separate from alpha_t so the
            # decay curve stays monotonic and alpha_t logs the pure decay value.
            ramp_factor = 1.0
            if do_asym and not in_sdedit_phase and transmission_ramp_steps > 0 and t_obj_denoising_idx is not None:
                steps_into_tx = step_index - t_obj_denoising_idx
                if steps_into_tx < transmission_ramp_steps:
                    ramp_factor = (steps_into_tx + 1) / float(transmission_ramp_steps)
            effective_alpha = alpha_t * ramp_factor

            use_stage1 = (not in_sdedit_phase) and use_inversion_attention_transmission and need_base_mask and transmission_alpha > 0.0

            if use_stage1:
                M = (transmission_alpha * effective_alpha * mask).clamp(0, 1)
                M = _cast(M)
                Mbc = M.expand(1, latents.shape[1], latent_spatial[0], latent_spatial[1])
                latents = _cast(latents)
                lat_for_unet = Mbc * lat_orig + (1.0 - Mbc) * latents
                std_ref = latents.detach().float().std(dim=(1, 2, 3), keepdim=True) + 1e-8
                std_blend = lat_for_unet.detach().float().std(dim=(1, 2, 3), keepdim=True) + 1e-8
                lat_for_unet = _cast(lat_for_unet * (std_ref / std_blend))
            else:
                M = torch.zeros_like(mask)
                lat_for_unet = _cast(latents)

            # ── UNet + CFG ────────────────────────────────────────────────────
            if recorder is not None:
                recorder.begin_step()

            latent_input = lat_for_unet if guidance_scale <= 1.0 else torch.cat([lat_for_unet, lat_for_unet], dim=0)
            latent_input = _cast(pipe.scheduler.scale_model_input(latent_input, t))

            # When dual_recon_transmission is on, normal pass always uses the full prompt;
            # token_replace_frac is only applied when running without the separate generic pass.
            if dual_recon_transmission:
                active_embeds = prompt_embeds
            else:
                active_embeds = (
                    prompt_embeds_generic
                    if prompt_embeds_generic is not None and progress < token_replace_frac
                    else prompt_embeds
                )
            noise_pred = _cast(pipe.unet(latent_input, t, encoder_hidden_states=active_embeds, return_dict=False)[0])

            if guidance_scale > 1.0:
                eps_u, eps_c = noise_pred.chunk(2)
                noise_pred = _cast(eps_u + guidance_scale * (eps_c - eps_u))

            # Save x_t before the scheduler step so both normal and generic passes
            # are stepped from the same starting latent.
            latents_x_t = latents

            latents = pipe.scheduler.step(noise_pred, t, latents_x_t, eta=eta, return_dict=True).prev_sample
            latents = latents.to(device=device, dtype=dtype)

            if recorder is not None:
                recorder.end_step()
                recon_map_raw = recorder.step_maps[-1]
            else:
                recon_map_raw = None

            # ── generic UNet pass (dual_recon_transmission) ───────────────────
            # Leakage is detected from the normal pass attention (recon_map_raw).
            # The generic pass only provides an alternative latent trajectory to
            # blend in where leakage is found; its own attention is not used.
            # The recorder begin/end are still called so the hook state stays clean.
            latents_generic_step = None
            if dual_recon_transmission and prompt_embeds_generic is not None:
                if recorder is not None:
                    recorder.begin_step()
                latent_input_g = lat_for_unet if guidance_scale <= 1.0 else torch.cat([lat_for_unet, lat_for_unet], dim=0)
                latent_input_g = _cast(pipe.scheduler.scale_model_input(latent_input_g, t))
                noise_pred_g = _cast(pipe.unet(latent_input_g, t, encoder_hidden_states=prompt_embeds_generic, return_dict=False)[0])
                if guidance_scale > 1.0:
                    eps_u_g, eps_c_g = noise_pred_g.chunk(2)
                    noise_pred_g = _cast(eps_u_g + guidance_scale * (eps_c_g - eps_u_g))
                latents_generic_step = pipe.scheduler.step(noise_pred_g, t, latents_x_t, eta=eta, return_dict=True).prev_sample
                latents_generic_step = latents_generic_step.to(device=device, dtype=dtype)
                if recorder is not None:
                    recorder.end_step()  # discard generic attention — not used

            # ── stage 2: post-step merge ──────────────────────────────────────
            recon_mask_base = recon_mask_bin = recon_mask_dilated = recon_ring = recon_otsu_thr = None

            if (
                use_reconstruction_attention_transmission
                and recorder is not None
                and recon_map_raw is not None
                and not invert_mask
            ):
                recon_mask_base = _cast(
                    F.interpolate(
                        preprocess_mask(recon_map_raw, target_size=latent_spatial, sharpness=10.0),
                        size=latent_spatial,
                        mode="bilinear",
                        align_corners=False,
                    )
                )
                recon_mask_bin, recon_otsu_thr = _binary_mask_otsu(recon_mask_base)
                recon_mask_dilated = _cast(dilate_mask(keep_largest_component(recon_mask_bin, n=3), radius=recon_dilate_radius))

                if recon_mask_dilated.shape[-2:] != latent_spatial:
                    recon_mask_dilated = _cast(F.interpolate(recon_mask_dilated, size=latent_spatial, mode="nearest"))

                if recon_blur_k and recon_blur_k > 1:
                    recon_mask_dilated = _cast(avg_pool_blur(recon_mask_dilated, k=_ensure_odd(recon_blur_k)))

            # Build the two stage-2 component masks separately so they can be
            # applied to different latent sources.
            mask2_inv   = torch.zeros_like(mask)  # object protection → lat_for_stage2
            mask2_recon = torch.zeros_like(mask)  # leakage ring → generic latent (or lat_for_stage2)

            if (not in_sdedit_phase) and use_inversion_attention_transmission and need_base_mask:
                mask2_inv = (mask * effective_alpha).clamp(0, 1)

            recon_tx_active = use_reconstruction_attention_transmission and progress < recon_attn_end_frac
            if recon_tx_active and recon_mask_dilated is not None and progress >= recon_attn_start_frac:
                if need_base_mask:
                    mask_gap = dilate_mask(main_mask_bin, radius=transition_gap_radius) if transition_gap_radius > 0 else main_mask_bin
                    recon_ring = (recon_mask_dilated - mask_gap).clamp(0, 1)
                else:
                    recon_ring = recon_mask_dilated.clamp(0, 1)
                mask2_recon = (recon_ring * (alpha_t if recon_alpha_decay else 1.0)).clamp(0, 1)

            mask2_inv   = _cast((transmission_alpha * mask2_inv).clamp(0, 1))
            mask2_recon = _cast((transmission_alpha * mask2_recon).clamp(0, 1))
            mask2 = torch.maximum(mask2_inv, mask2_recon)  # combined — for debug output only

            if mask2_inv.shape[-2:] != latent_spatial:
                mask2_inv = _cast(F.interpolate(mask2_inv, size=latent_spatial, mode="nearest"))
            if mask2_recon.shape[-2:] != latent_spatial:
                mask2_recon = _cast(F.interpolate(mask2_recon, size=latent_spatial, mode="nearest"))

            latents_before_merge = latents.clone()

            # Inversion component: protect the object region using the real inversion latent.
            if transmission_alpha > 0.0 and lat_obj_source is not None and float(mask2_inv.max().item()) > 0.0:
                Mbc2_inv = mask2_inv.expand(1, latents.shape[1], latent_spatial[0], latent_spatial[1])
                std_ref2 = latents.detach().float().std(dim=(1, 2, 3), keepdim=True) + 1e-8
                latents = Mbc2_inv * lat_obj_source + (1.0 - Mbc2_inv) * latents
                std_new = latents.detach().float().std(dim=(1, 2, 3), keepdim=True) + 1e-8
                latents = _cast(latents * (std_ref2 / std_new))

            # Recon ring: suppress leakage using the colour-neutral ring source.
            # When dual_recon_transmission is on, use generic-prompt latents instead.
            recon_ring_source = (
                latents_generic_step if (dual_recon_transmission and latents_generic_step is not None)
                else lat_ring_source
            )
            if transmission_alpha > 0.0 and recon_ring_source is not None and float(mask2_recon.max().item()) > 0.0:
                Mbc2_recon = mask2_recon.expand(1, latents.shape[1], latent_spatial[0], latent_spatial[1])
                std_ref2 = latents.detach().float().std(dim=(1, 2, 3), keepdim=True) + 1e-8
                latents = Mbc2_recon * recon_ring_source + (1.0 - Mbc2_recon) * latents
                std_new = latents.detach().float().std(dim=(1, 2, 3), keepdim=True) + 1e-8
                latents = _cast(latents * (std_ref2 / std_new))

            # ── debug saving ──────────────────────────────────────────────────
            if debug_dir is not None and ((step_index - start_index) % max(1, save_debug_every) == 0):
                step_dir = os.path.join(debug_dir, f"step_{step_index:03d}_t{t_int}")
                os.makedirs(step_dir, exist_ok=True)

                # base mask (inversion-derived or SAM, same every step)
                if base_mask_source != "sam":
                    _save_map_png(base_mask_raw_2d, os.path.join(step_dir, "00_base_attn_avg.png"), size=latent_spatial, normalize=True)
                _save_map_png(base_mask_soft, os.path.join(step_dir, "01_base_mask_soft.png"))
                _save_map_png(main_mask_bin, os.path.join(step_dir, "02_base_mask.png"))

                # per-step inversion attention
                inv_step_map = inv_attn_map.get(t_int)
                if inv_step_map is not None:
                    _save_map_png(inv_step_map, os.path.join(step_dir, "03_inv_step_attn.png"), size=latent_spatial, normalize=True)

                # per-step reconstruction attention pipeline
                if recon_map_raw is not None:
                    _save_map_png(recon_map_raw, os.path.join(step_dir, "04_recon_attn_raw.png"), size=latent_spatial, normalize=True)
                if recon_mask_base is not None:
                    _save_map_png(recon_mask_base, os.path.join(step_dir, "05_recon_mask_soft.png"))
                if recon_mask_bin is not None:
                    _save_map_png(recon_mask_bin, os.path.join(step_dir, "06_recon_mask_binary.png"))
                if recon_mask_dilated is not None:
                    _save_map_png(recon_mask_dilated, os.path.join(step_dir, "07_recon_mask_dilated.png"))
                if recon_ring is not None:
                    _save_map_png(recon_ring, os.path.join(step_dir, "08_recon_ring.png"))

                # stage 1 (pre-UNet blend)
                _save_map_png(mask, os.path.join(step_dir, "09_stage1_mask.png"))
                _save_map_png(M, os.path.join(step_dir, "10_stage1_M.png"))

                # stage 2 (post-step merge)
                _save_map_png(mask2, os.path.join(step_dir, "11_stage2_mask.png"))

                # latent previews
                _save_latent_preview(lat_for_unet, os.path.join(step_dir, "12_latent_pre_unet.png"))
                _save_latent_preview(latents_before_merge, os.path.join(step_dir, "13_latent_post_step.png"))
                _save_latent_preview(latents, os.path.join(step_dir, "14_latent_post_merge.png"))

                with open(os.path.join(step_dir, "thresholds.txt"), "w") as f:
                    f.write(f"progress={progress:.4f}\n")
                    f.write(f"alpha_t={alpha_t:.4f}\n")
                    f.write(f"ramp_factor={ramp_factor:.4f}\n")
                    f.write(f"effective_alpha={effective_alpha:.4f}\n")
                    f.write(f"alpha_decay_start={alpha_decay_start}\n")
                    f.write(f"transmission_alpha_end={transmission_alpha_end}\n")
                    f.write(f"inversion_otsu_threshold={inv_otsu_thr}\n")
                    f.write(f"reconstruction_otsu_threshold={recon_otsu_thr}\n")
                    f.write(f"use_inversion_attention_transmission={use_inversion_attention_transmission}\n")
                    f.write(f"use_reconstruction_attention_transmission={use_reconstruction_attention_transmission}\n")
                    f.write(f"transmission_alpha={transmission_alpha}\n")
                    f.write(f"initial_noise_beta={initial_noise_beta}\n")
                    f.write(f"recon_dilate_radius={recon_dilate_radius}\n")
                    f.write(f"transition_gap_radius={transition_gap_radius}\n")
                    f.write(f"asymmetric_schedule={do_asym}\n")
                    if do_asym:
                        f.write(f"in_sdedit_phase={in_sdedit_phase}\n")
                        f.write(f"t_obj={t_obj_int}  t_bg={t_bg_int}\n")

    finally:
        restore_attention_processors(pipe, saved_attn_processors)

    return decode_latents_to_pil(pipe, latents)
