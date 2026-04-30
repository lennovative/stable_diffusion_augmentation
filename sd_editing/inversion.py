import torch
from PIL import Image

from .pipeline import load_image_rgb, encode_image_to_latents, encode_prompt_cfg
from .attention import (
    StepTokenAttentionRecorder,
    resolve_tokens_positions,
    register_attention_recorder,
    restore_attention_processors,
)


@torch.no_grad()
def ddim_invert_store(
    pipe,
    image_path=None,
    image=None,
    prompt="",
    tokens=None,
    num_inference_steps=50,
    invert_frac=1.0,
    guidance_scale=1.0,
    input_size=512,
    attention_res=32,
    allowed_places=("mid",),
    capture_attention=True,
    allow_missing_token=True,
    multi_token_merge="average",
):
    """
    DDIM inversion with optional cross-attention capture for a given token.

    Supports partial inversion via invert_frac (0.0–1.0).
    Pass either image_path or a PIL image directly.

    Returns a dict with:
        latents_all   : list of (timestep_int, cpu_latent)
        attns_all     : list of (timestep_int, attention_map | None)
        prompt        : the prompt used
        invert_frac   : the fraction used
        num_inverse_steps
    """
    dtype = next(pipe.unet.parameters()).dtype
    device = pipe.device

    invert_frac = float(max(0.0, min(1.0, invert_frac)))
    n_inverse_steps = max(1, int(round(num_inference_steps * invert_frac)))
    n_inverse_steps = min(n_inverse_steps, num_inference_steps)

    prompt = "" if prompt is None else str(prompt)

    if image is None:
        if image_path is None:
            raise ValueError("Either image_path or image must be provided.")
        image = load_image_rgb(image_path, size=(input_size, input_size))
    else:
        image = image.convert("RGB")
        if image.size != (input_size, input_size):
            image = image.resize((input_size, input_size), Image.Resampling.LANCZOS)

    latents = encode_image_to_latents(pipe, image).to(device=device, dtype=dtype)
    prompt_embeds = encode_prompt_cfg(pipe, prompt, guidance_scale=guidance_scale).to(device=device, dtype=dtype)

    if isinstance(tokens, str):
        tokens = [tokens]

    recorder = None
    if capture_attention and tokens is not None and prompt != "":
        all_positions = resolve_tokens_positions(pipe, prompt, tokens)
        valid = [(t, pos) for t, pos in zip(tokens, all_positions) if pos]
        missing = [t for t, pos in zip(tokens, all_positions) if not pos]
        if missing:
            msg = f"Tokens {missing} not found in inversion prompt: {repr(prompt)}"
            if allow_missing_token:
                print(f"[WARN] {msg}. Skipping missing tokens.")
            else:
                raise ValueError(msg)
        if valid:
            recorder = StepTokenAttentionRecorder(
                tokens_positions=[pos for _, pos in valid],
                out_res=attention_res,
                keep_cond_only=(guidance_scale > 1.0),
                allowed_places=allowed_places,
                multi_token_merge=multi_token_merge,
            )

    saved_attn_processors = None
    if recorder is not None:
        saved_attn_processors = register_attention_recorder(pipe, recorder, allowed_places=allowed_places)

    pipe.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
    inverse_timesteps = list(pipe.inverse_scheduler.timesteps)

    latents_all = []
    attns_all = []

    try:
        for i in range(n_inverse_steps):
            t = inverse_timesteps[i]

            if recorder is not None:
                recorder.begin_step()

            latent_input = latents if guidance_scale <= 1.0 else torch.cat([latents, latents], dim=0)
            latent_input = pipe.inverse_scheduler.scale_model_input(latent_input, t)
            latent_input = latent_input.to(device=device, dtype=dtype)

            noise_pred = pipe.unet(
                latent_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                eps_u, eps_c = noise_pred.chunk(2)
                noise_pred = eps_u + guidance_scale * (eps_c - eps_u)

            latents = pipe.inverse_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents = latents.to(device=device, dtype=dtype)

            if recorder is not None:
                recorder.end_step()
                step_attn = recorder.step_maps[-1]
                step_attn = step_attn.clone() if step_attn is not None else None
            else:
                step_attn = None

            latents_all.append((int(t.item()), latents.detach().cpu().clone()))
            attns_all.append((int(t.item()), step_attn))

    finally:
        restore_attention_processors(pipe, saved_attn_processors)

    return {
        "latents_all": latents_all,
        "attns_all": attns_all,
        "prompt": prompt,
        "invert_frac": invert_frac,
        "num_inverse_steps": n_inverse_steps,
    }
