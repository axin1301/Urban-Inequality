import torch


def apply():
    from trl.trainer import grpo_trainer as G

    if getattr(G.GRPOTrainer, "_multiimage_patched", False):
        return

    _split, _unsplit = G.split_pixel_values_by_grid, G.unsplit_pixel_values_by_grid
    _PER_IMAGE = ("pixel_values", "image_sizes", "pixel_attention_mask")

    def split(batch):
        if "image_grid_thw" in batch or batch.get("num_images") is None:
            return _split(batch)
        n = list(batch["num_images"])
        out = dict(batch)
        for k in _PER_IMAGE:
            v = batch.get(k)
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.size(0) == sum(n):
                out[k] = list(torch.split(v, n, dim=0))
        return out

    def unsplit(batch):
        batch = _unsplit(batch)
        out = dict(batch)
        for k in _PER_IMAGE[1:]:
            if isinstance(batch.get(k), list):
                out[k] = torch.cat(batch[k], dim=0)
        return out

    G.split_pixel_values_by_grid = split
    G.unsplit_pixel_values_by_grid = unsplit
    orig = G.GRPOTrainer._get_per_token_logps_and_entropies
    orig = getattr(orig, "__wrapped__", orig)

    def patched(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None,
                compute_entropy=False, pixel_values=None, image_grid_thw=None, num_images=None,
                pixel_attention_mask=None, image_sizes=None, token_type_ids=None):
        if (pixel_values is None or image_grid_thw is not None or num_images is None
                or all(n == 1 for n in num_images)):
            return orig(self, model, input_ids, attention_mask, logits_to_keep, batch_size=batch_size,
                        compute_entropy=compute_entropy, pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw, num_images=num_images,
                        pixel_attention_mask=pixel_attention_mask, image_sizes=image_sizes,
                        token_type_ids=token_type_ids)
        if pixel_values.size(0) != sum(num_images):
            raise RuntimeError(
                f"[multiimage patch] pixel_values 有 {pixel_values.size(0)} 行，但图像数之和为 {sum(num_images)}："
                "每张图应只占 1 行（InternVL 需 crop_to_patches=False）")
        cum = [0]
        for n in num_images:
            cum.append(cum[-1] + n)
        batch_size = batch_size or input_ids.size(0)
        all_logps, all_ent = [], []
        for start in range(0, input_ids.size(0), batch_size):
            end = min(start + batch_size, input_ids.size(0))
            ids = input_ids[start:end]
            mi = {"input_ids": ids, "attention_mask": attention_mask[start:end],
                  "pixel_values": pixel_values[cum[start]:cum[end]]}
            if image_sizes is not None:
                mi["image_sizes"] = image_sizes[cum[start]:cum[end]]
            if pixel_attention_mask is not None:
                mi["pixel_attention_mask"] = pixel_attention_mask[cum[start]:cum[end]]
            if token_type_ids is not None:
                mi["token_type_ids"] = token_type_ids[start:end]
            if "logits_to_keep" in self.model_kwarg_keys:
                mi["logits_to_keep"] = logits_to_keep + 1
            mi["use_cache"] = False
            logits = model(**mi).logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :] / self.temperature
            all_logps.append(G.selective_log_softmax(logits, ids[:, -logits_to_keep:]))
            if compute_entropy:
                with torch.no_grad():
                    all_ent.append(G.entropy_from_logits(logits))
        return torch.cat(all_logps, 0), (torch.cat(all_ent, 0) if compute_entropy else None)

    G.GRPOTrainer._get_per_token_logps_and_entropies = patched
    G.GRPOTrainer._multiimage_patched = True
    print("[trl patch] multi-image slicing for non-Qwen VLMs enabled", flush=True)
