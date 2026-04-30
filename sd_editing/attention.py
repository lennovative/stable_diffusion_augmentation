import math
import torch
import torch.nn.functional as F


def merge_token_maps(maps, mode):
    """Merge per-token attention maps. maps: list of [H, W] tensors."""
    stacked = torch.stack(maps, dim=0)  # [N, H, W]
    if mode == "average":
        return stacked.mean(dim=0)
    if mode == "maximum":
        return stacked.max(dim=0).values
    if mode == "intersection":
        return stacked.min(dim=0).values
    raise ValueError(f"Unknown multi_token_merge: {mode!r}. Choose average | maximum | intersection")


def find_subsequence_positions(sequence, subsequence):
    out = []
    n, m = len(sequence), len(subsequence)
    if m == 0 or m > n:
        return out
    for i in range(n - m + 1):
        if sequence[i : i + m] == subsequence:
            out.extend(range(i, i + m))
    return out


def resolve_token_positions(pipe, prompt, token):
    tokenizer = pipe.tokenizer
    prompt_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids[0].tolist()

    token_ids = tokenizer(token, add_special_tokens=False).input_ids
    return find_subsequence_positions(prompt_ids, token_ids)


def resolve_tokens_positions(pipe, prompt, tokens):
    """Resolve positions for each token in tokens. Returns list of lists."""
    return [resolve_token_positions(pipe, prompt, t) for t in tokens]


class StepTokenAttentionRecorder:
    def __init__(
        self,
        tokens_positions,
        out_res=32,
        keep_cond_only=True,
        allowed_places=("mid",),
        multi_token_merge="average",
    ):
        # backward compat: flat list[int] → single-token list-of-lists
        if tokens_positions and isinstance(tokens_positions[0], int):
            tokens_positions = [tokens_positions]
        self.tokens_positions = [list(tp) for tp in tokens_positions]
        self.multi_token_merge = multi_token_merge
        self.out_res = out_res
        self.keep_cond_only = keep_cond_only
        self.allowed_places = set(allowed_places)
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self._current_maps = []
        self.step_maps = []

    def begin_step(self):
        self._current_maps = []

    def end_step(self):
        if len(self._current_maps) == 0:
            self.step_maps.append(None)
            return
        hm = torch.stack(self._current_maps, dim=0).mean(dim=0)
        hm = hm - hm.min()
        hm = hm / (hm.max() + 1e-8)
        self.step_maps.append(hm.cpu())

    def __call__(self, attn_probs, is_cross, place_in_unet):
        if is_cross and place_in_unet in self.allowed_places:
            if self.keep_cond_only and attn_probs.shape[0] >= 2:
                attn_probs = attn_probs[attn_probs.shape[0] // 2 :]

            q_len = attn_probs.shape[1]
            spatial_res = int(math.sqrt(q_len))
            if spatial_res * spatial_res == q_len:
                per_token_maps = []
                for positions in self.tokens_positions:
                    valid_positions = [p for p in positions if p < attn_probs.shape[2]]
                    if valid_positions:
                        tm = attn_probs[:, :, valid_positions].mean(dim=-1).mean(dim=0)
                        tm = tm.reshape(1, 1, spatial_res, spatial_res)
                        tm = F.interpolate(
                            tm,
                            size=(self.out_res, self.out_res),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]
                        per_token_maps.append(tm.detach().float().cpu())
                if per_token_maps:
                    merged = merge_token_maps(per_token_maps, self.multi_token_merge)
                    self._current_maps.append(merged)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0

        return attn_probs


class CrossAttnCaptureProcessor:
    def __init__(self, controller, place_in_unet):
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_probs = attn.get_attention_scores(query, key, attention_mask)
        self.controller(attn_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attn_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def register_attention_recorder(pipe, recorder, allowed_places=("mid",)):
    saved_attn_processors = dict(pipe.unet.attn_processors)

    if len(saved_attn_processors) == 0:
        raise RuntimeError(
            "pipe.unet.attn_processors is empty. Reload the pipeline or reset the UNet processors."
        )

    attn_procs = {}
    for name, old_proc in saved_attn_processors.items():
        if name.startswith("mid_block"):
            place = "mid"
        elif name.startswith("up_blocks"):
            place = "up"
        elif name.startswith("down_blocks"):
            place = "down"
        else:
            attn_procs[name] = old_proc
            continue

        if place in allowed_places:
            attn_procs[name] = CrossAttnCaptureProcessor(recorder, place)
        else:
            attn_procs[name] = old_proc

    if len(attn_procs) != len(saved_attn_processors):
        raise RuntimeError(
            f"Built {len(attn_procs)} attention processors, "
            f"but expected {len(saved_attn_processors)}."
        )

    pipe.unet.set_attn_processor(attn_procs)
    recorder.num_att_layers = len(pipe.unet.attn_processors)

    return saved_attn_processors


def restore_attention_processors(pipe, saved_attn_processors):
    if saved_attn_processors is not None:
        pipe.unet.set_attn_processor(saved_attn_processors)
