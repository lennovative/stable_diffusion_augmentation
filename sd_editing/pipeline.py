import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, DDIMScheduler, DDIMInverseScheduler


def load_sdxl_edit_pipe(
    model_name,
    device="cuda",
    dtype=torch.float16,
    custom_embed_path=None,
    use_xformers=True,
):
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    # SDXL VAE has force_upcast=True; it produces NaN/black images in fp16.
    # Keep it in fp32 throughout while the UNet stays in fp16.
    pipe.vae.to(torch.float32)

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if use_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[MEM] xFormers enabled.")
        except Exception as e:
            print(f"[MEM] xFormers not enabled: {e}")

    num_added_total = 0

    if custom_embed_path is not None and os.path.isfile(custom_embed_path):
        embed_data = torch.load(custom_embed_path, map_location="cpu")
        # SDXL has two tokenizers/text-encoders; add custom tokens to both.
        for tokenizer, text_encoder in [
            (pipe.tokenizer, pipe.text_encoder),
            (pipe.tokenizer_2, pipe.text_encoder_2),
        ]:
            for token_name, custom_embedding in embed_data.items():
                num_added = tokenizer.add_tokens([token_name])
                if num_added == 0:
                    print(f"Token '{token_name}' already exists in {tokenizer.__class__.__name__}.")
                else:
                    print(f"Added custom token '{token_name}' to {tokenizer.__class__.__name__}.")
                    text_encoder.resize_token_embeddings(len(tokenizer))
                    token_id = tokenizer.convert_tokens_to_ids(token_name)
                    emb = custom_embedding
                    # Resize embedding if the two encoders have different hidden dims.
                    target_dim = text_encoder.get_input_embeddings().weight.shape[1]
                    if emb.shape[-1] != target_dim:
                        emb = torch.nn.functional.interpolate(
                            emb.float().reshape(1, 1, -1),
                            size=target_dim,
                            mode="linear",
                            align_corners=False,
                        ).reshape(-1).to(emb.dtype)
                    text_encoder.get_input_embeddings().weight.data[token_id] = emb.to(
                        device=text_encoder.device,
                        dtype=text_encoder.dtype,
                    )
                    num_added_total += 1

    return pipe, num_added_total


def load_image_rgb(path, size=(1024, 1024)):
    return Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)


def preprocess_image_for_sdedit(image: Image.Image, mode: str, blur_radius: float = 3) -> Image.Image:
    """
    Neutralise source image colours before SDEdit ring-merge encoding.
    mode: "none" | "grayscale" | "grayscale_blur"
    """
    from PIL import ImageFilter, ImageOps
    if "blur" in mode:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if "grayscale" in mode:
        image = ImageOps.grayscale(image).convert("RGB")
    return image


@torch.no_grad()
def encode_image_to_latents(pipe, image):
    image = pipe.image_processor.preprocess(image).to(device=pipe.device, dtype=pipe.vae.dtype)
    latents = pipe.vae.encode(image).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    return latents


@torch.no_grad()
def decode_latents_to_pil(pipe, latents):
    # Cast to VAE dtype (float32) before decode — SDXL VAE is unstable in fp16.
    latents = latents.to(dtype=pipe.vae.dtype) / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image.float(), output_type="pil")[0]


@torch.no_grad()
def encode_prompt_cfg(pipe, prompt, guidance_scale=7.5, negative_prompt="", image_size=(1024, 1024)):
    """
    Encode a prompt for SDXL.

    Returns:
        prompt_embeds     : (2*B, seq, hidden) when do_cfg else (B, seq, hidden)
        added_cond_kwargs : dict with "text_embeds" and "time_ids" required by the SDXL UNet
    """
    do_cfg = guidance_scale > 1.0
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt if do_cfg else None,
        negative_prompt_2=None,
    )

    if do_cfg:
        prompt_embeds_out = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_out = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    else:
        prompt_embeds_out = prompt_embeds
        pooled_out = pooled_prompt_embeds

    h, w = image_size
    # time_ids encodes: original_height, original_width, crop_top, crop_left, target_height, target_width
    time_ids = torch.tensor([[h, w, 0, 0, h, w]], dtype=prompt_embeds.dtype, device=pipe.device)
    if do_cfg:
        time_ids = time_ids.repeat(2, 1)

    added_cond_kwargs = {"text_embeds": pooled_out, "time_ids": time_ids}
    return prompt_embeds_out, added_cond_kwargs
