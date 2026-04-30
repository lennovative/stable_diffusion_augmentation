import numpy as np
import torch
import torch.nn.functional as F


def avg_pool_blur(x, k=5):
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return x
    y = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    if y.shape[-2:] != x.shape[-2:]:
        y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return y


def otsu_threshold(mask_2d: torch.Tensor, num_bins: int = 256) -> float:
    x = mask_2d.detach().float()
    x = x - x.min()
    x = x / (x.max() + 1e-8)

    hist = torch.histc(x.flatten(), bins=num_bins, min=0.0, max=1.0)
    hist = hist / (hist.sum() + 1e-8)

    bin_centers = torch.linspace(0.0, 1.0, steps=num_bins, device=x.device)
    omega = torch.cumsum(hist, dim=0)
    mu = torch.cumsum(hist * bin_centers, dim=0)
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu).pow(2) / (omega * (1.0 - omega) + 1e-8)
    idx = torch.argmax(sigma_b2).item()
    return float(bin_centers[idx].item())


def preprocess_mask(mask_2d, target_size, sharpness=10.0, threshold=None, invert=False):
    x = mask_2d[None, None].float()
    x = x - x.min()
    x = x / (x.max() + 1e-8)

    if threshold is None:
        threshold = otsu_threshold(x[0, 0])

    x = torch.sigmoid((x - threshold) * sharpness)

    if invert:
        x = 1.0 - x

    x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
    x = x.clamp(0, 1)
    return x


def binary_from_mask(mask, thresh=0.1):
    return (mask >= thresh).float()


def dilate_mask(mask, radius=3):
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


def erode_mask(mask, radius=3):
    if radius <= 0:
        return mask
    return 1.0 - dilate_mask(1.0 - mask, radius=radius)


def keep_largest_component(mask, n=1):
    n = max(1, int(n))
    m = (mask[0, 0] > 0.5).detach().cpu().numpy().astype(np.uint8)
    h, w = m.shape
    visited = np.zeros_like(m, dtype=np.uint8)
    components = []

    for y in range(h):
        for x in range(w):
            if m[y, x] == 0 or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = 1
            pts = []
            while stack:
                cx, cy = stack.pop()
                pts.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < w and 0 <= ny < h and m[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = 1
                        stack.append((nx, ny))
            components.append(pts)

    components.sort(key=len, reverse=True)

    out = np.zeros_like(m, dtype=np.float32)
    for component in components[:n]:
        for x, y in component:
            out[y, x] = 1.0

    return torch.from_numpy(out)[None, None].to(mask.device, dtype=mask.dtype)


def build_base_mask_from_inversion_attn(
    inv_step_maps,
    step_frac_range=(0.0, 0.5),
    target_size=(64, 64),
    invert=False,
    erode_radius=0,
    final_dilate_radius=None,
):
    valid = [m for m in inv_step_maps if m is not None]
    if len(valid) == 0:
        raise RuntimeError("No inversion attention maps available.")

    n = len(inv_step_maps)
    s0 = int(step_frac_range[0] * n)
    s1 = max(s0 + 1, int(step_frac_range[1] * n))

    maps = [m for m in inv_step_maps[s0:s1] if m is not None]
    if len(maps) == 0:
        maps = valid

    base_raw = torch.stack(maps, dim=0).mean(dim=0)
    base_raw = base_raw - base_raw.min()
    base_raw = base_raw / (base_raw.max() + 1e-8)

    base_soft = preprocess_mask(base_raw, target_size=target_size, sharpness=12.0, invert=False)
    base_soft = avg_pool_blur(base_soft, k=5)
    base_soft = base_soft.clamp(0, 1)

    main = keep_largest_component(binary_from_mask(base_soft, 0.5))

    if erode_radius > 0:
        main = erode_mask(main, radius=erode_radius)
        main = keep_largest_component(main)

    if final_dilate_radius is None:
        final_dilate_radius = 2 if not invert else 0

    if final_dilate_radius > 0:
        main = dilate_mask(main, radius=final_dilate_radius)

    if invert:
        main = 1.0 - main

    return base_raw, base_soft, main
