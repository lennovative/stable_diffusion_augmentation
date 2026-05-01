import torch
import torch.nn.functional as F


def _gaussian_blur(mask, radius):
    """Blur a [1,1,H,W] float mask with a box approximation of a Gaussian."""
    if radius <= 0:
        return mask
    k = 2 * int(radius) + 1
    mask = F.avg_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
    mask = F.avg_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
    return mask.clamp(0.0, 1.0)


def load_grounded_sam(grounding_dino_model_id, sam_model_id, device, box_threshold=0.3, text_threshold=0.25, edge_blur_radius=2):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

    print(f"[SAM] Loading GroundingDINO: {grounding_dino_model_id}")
    gd_processor = AutoProcessor.from_pretrained(grounding_dino_model_id)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_dino_model_id).to(device)
    gd_model.eval()

    print(f"[SAM] Loading SAM: {sam_model_id}")
    sam_model = SamModel.from_pretrained(sam_model_id).to(device)
    sam_processor = SamProcessor.from_pretrained(sam_model_id)
    sam_model.eval()

    return {
        "gd_model": gd_model,
        "gd_processor": gd_processor,
        "sam_model": sam_model,
        "sam_processor": sam_processor,
        "device": device,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "edge_blur_radius": edge_blur_radius,
    }


def grounded_sam_mask(image_pil, tokens, grounded_sam, target_size):
    """
    Returns a float mask tensor [1,1,H,W] at target_size using
    GroundingDINO (text→boxes) + SAM (boxes→mask).
    """
    device = grounded_sam["device"]
    gd_model = grounded_sam["gd_model"]
    gd_processor = grounded_sam["gd_processor"]
    sam_model = grounded_sam["sam_model"]
    sam_processor = grounded_sam["sam_processor"]
    box_threshold = grounded_sam["box_threshold"]
    text_threshold = grounded_sam["text_threshold"]
    edge_blur_radius = grounded_sam["edge_blur_radius"]

    # GroundingDINO expects ". "-separated labels with a trailing period
    text = " . ".join(tokens) + " ."

    inputs = gd_processor(images=image_pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gd_model(**inputs)

    results = gd_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]],  # (H, W)
    )[0]

    boxes = results["boxes"].cpu()  # [N, 4] xyxy absolute

    if len(boxes) == 0:
        print(f"[WARN][SAM] No boxes detected for tokens {tokens}. Returning empty mask.")
        return torch.zeros(1, 1, *target_size)

    # SAM: boxes → masks; input_boxes is [batch, N, 4]
    inputs_sam = sam_processor(
        images=image_pil,
        input_boxes=[[b.tolist() for b in boxes]],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        sam_out = sam_model(**inputs_sam)

    # pred_masks: [1, N, 3, H/4, W/4];  iou_scores: [1, N, 3]
    masks_list = sam_processor.post_process_masks(
        sam_out.pred_masks.cpu(),
        inputs_sam["original_sizes"].cpu(),
        inputs_sam["reshaped_input_sizes"].cpu(),
    )
    masks = masks_list[0]           # [N, 3, H, W]  bool
    iou = sam_out.iou_scores[0].cpu()  # [N, 3]
    best_idx = iou.argmax(dim=-1)   # [N]

    best_masks = torch.stack(
        [masks[i, best_idx[i]] for i in range(len(boxes))], dim=0
    ).float()                       # [N, H, W]

    combined = best_masks.any(dim=0).float()[None, None]  # [1, 1, H, W]
    combined = F.interpolate(combined, size=target_size, mode="nearest")
    combined = _gaussian_blur(combined, edge_blur_radius)
    return combined.cpu()
