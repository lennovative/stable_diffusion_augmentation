import os
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler, DDIMInverseScheduler


def load_sd15_edit_pipe(
    model_name,
    device="cuda",
    dtype=torch.float16,
    custom_embed_path=None,
    use_xformers=True,
):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_name,
        safety_checker=None,
        torch_dtype=dtype,
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

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
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder

        for token_name, custom_embedding in embed_data.items():
            num_added = tokenizer.add_tokens([token_name])
            if num_added == 0:
                print(f"Token '{token_name}' already exists.")
            else:
                print(f"Added custom token '{token_name}'.")
                text_encoder.resize_token_embeddings(len(tokenizer))
                token_id = tokenizer.convert_tokens_to_ids(token_name)
                text_encoder.get_input_embeddings().weight.data[token_id] = custom_embedding.to(
                    device=text_encoder.device,
                    dtype=text_encoder.dtype,
                )
                num_added_total += 1

    return pipe, num_added_total


def load_image_rgb(path, size=(512, 512)):
    return Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)


@torch.no_grad()
def encode_image_to_latents(pipe, image):
    image = pipe.image_processor.preprocess(image).to(device=pipe.device, dtype=pipe.vae.dtype)
    latents = pipe.vae.encode(image).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    return latents


@torch.no_grad()
def decode_latents_to_pil(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


@torch.no_grad()
def encode_prompt_cfg(pipe, prompt, guidance_scale=7.5, negative_prompt=""):
    do_cfg = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=pipe.device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
    )
    if do_cfg:
        return torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    return prompt_embeds
