import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    # ---- [SEG] -> visual-token attention options
    parser.add_argument(
        "--disable_seg_attn",
        action="store_true",
        default=False,
        help="do not compute / save the [SEG]->visual-token attention",
    )
    parser.add_argument(
        "--attn_layers",
        default="all",
        type=str,
        help='decoder layers used for the aggregated map: "all", "last", '
        'a range "16-31" or a list "0,8,31" (0-based). The raw file always '
        "contains every layer.",
    )
    parser.add_argument(
        "--attn_query",
        default="seg",
        type=str,
        choices=["seg", "lisa"],
        help='"seg": attention row of the [SEG] token itself; "lisa": the row one '
        "position earlier, i.e. the hidden state LISA feeds to SAM.",
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


# --------------------------------------------------------------------------
# [SEG] -> visual-token attention helpers
# --------------------------------------------------------------------------
def parse_layer_spec(spec, n_layers):
    """'all' | 'last' | '16-31' | '0,8,31'  ->  sorted list of 0-based layer ids."""
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(n_layers))
    if spec == "last":
        return [n_layers - 1]
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    layers = sorted(set(layers))
    assert all(0 <= l < n_layers for l in layers), (
        "attn_layers out of range, the model has {} layers".format(n_layers)
    )
    return layers


def aggregate_seg_attention(attn, layers):
    """
    attn: (L, H, S, N) tensor from LISAForCausalLM.get_seg_visual_attention.
    Returns (S, N): mean over the selected layers and over all heads.
    """
    return attn[layers].mean(dim=(0, 1))


def clip_view_box(h, w, clip_image_processor):
    """
    CLIPImageProcessor resizes the shortest edge to 224 and CENTER-CROPS a square.
    So the 16x16 patch grid only covers this square region of the original image.
    Returns (top, left, crop_h, crop_w) in original-image pixels.
    """
    if getattr(clip_image_processor, "do_center_crop", True):
        side = min(h, w)
        return (h - side) // 2, (w - side) // 2, side, side
    return 0, 0, h, w


def make_heatmap(grid_map, h, w, crop_box):
    """
    grid_map: (g, g) tensor (one [SEG] token). Min-max normalised to [0, 1], then
    resized with BILINEAR interpolation to the region of the original image that
    CLIP saw, and placed on an (h, w) canvas.
    Returns heat (h, w) float32 in [0, 1] and valid (h, w) bool (inside CLIP view).
    """
    top, left, ch, cw = crop_box
    g = grid_map.float()
    g = (g - g.min()) / (g.max() - g.min() + 1e-12)
    up = F.interpolate(
        g[None, None], size=(ch, cw), mode="bilinear", align_corners=False
    )[0, 0]
    up = up.clamp(0, 1).numpy()

    heat = np.zeros((h, w), dtype=np.float32)
    valid = np.zeros((h, w), dtype=bool)
    heat[top : top + ch, left : left + cw] = up
    valid[top : top + ch, left : left + cw] = True
    return heat, valid


def save_heatmap_images(heat, valid, image_rgb, heat_path, overlay_path, alpha=0.5):
    """Pure heat map + overlay on the original image (outside CLIP's view = darkened)."""
    heat_u8 = np.uint8(np.clip(heat, 0, 1) * 255)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)  # BGR
    color[~valid] = 0
    cv2.imwrite(heat_path, color)

    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    overlay = img_bgr.copy()
    overlay[valid] = (1 - alpha) * img_bgr[valid] + alpha * color[valid].astype(
        np.float32
    )
    overlay[~valid] *= 0.3
    cv2.imwrite(overlay_path, overlay.astype(np.uint8))


def save_seg_attention(seg_attn, image_np, base_name, args, clip_image_processor):
    """Save raw / aggregated attention matrices and heat maps for every [SEG] token."""
    if seg_attn is None:
        print("No [SEG] token was generated -> no attention to save.")
        return

    attn = seg_attn["attn"]  # (L, H, S, N)
    n_layers, n_heads, n_seg, n_vis = attn.shape
    g = seg_attn["grid_size"]
    layers = parse_layer_spec(args.attn_layers, n_layers)
    agg = aggregate_seg_attention(attn, layers)  # (S, N)
    grid_maps = agg.reshape(n_seg, g, g)  # row-major patch order

    h, w = image_np.shape[:2]
    crop_box = clip_view_box(h, w, clip_image_processor)

    prefix = "{}/{}".format(args.vis_save_path, base_name)

    raw_path = "{}_seg_attn_raw.npy".format(prefix)
    np.save(raw_path, attn.numpy())  # (L, H, S, N)
    agg_path = "{}_seg_attn_grid.npy".format(prefix)
    np.save(agg_path, grid_maps.numpy())  # (S, g, g)
    print(
        "[SEG] attention (query={}, layers={}, {} seg token(s)):".format(
            seg_attn["query"], args.attn_layers, n_seg
        )
    )
    print("{} has been saved. shape (layers, heads, seg, visual) = {}".format(
        raw_path, tuple(attn.shape)))
    print("{} has been saved. shape (seg, {}, {})".format(agg_path, g, g))

    for j in range(n_seg):
        gm = grid_maps[j]
        mass = float(gm.sum())
        peak = int(gm.reshape(-1).argmax())
        print(
            "  seg {}: attention mass on visual tokens = {:.4f}, peak patch (row, col) = ({}, {})".format(
                j, mass, peak // g, peak % g
            )
        )

        csv_path = "{}_seg{}_attn_grid.csv".format(prefix, j)
        np.savetxt(csv_path, gm.numpy(), delimiter=",", fmt="%.6e")

        heat, valid = make_heatmap(gm, h, w, crop_box)
        heat_path = "{}_seg{}_attn_heatmap.png".format(prefix, j)
        overlay_path = "{}_seg{}_attn_overlay.png".format(prefix, j)
        save_heatmap_images(heat, valid, image_np, heat_path, overlay_path)
        for p in (csv_path, heat_path, overlay_path):
            print("{} has been saved.".format(p))


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    use_seg_attn = not args.disable_seg_attn

    while True:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        prompt = input("Please input your prompt: ")
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_path = input("Please input the image path: ")
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # evaluate() keeps its old 2-value return unless return_seg_attention=True
        result = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
            return_seg_attention=use_seg_attn,
            attn_query=args.attn_query,
        )
        if use_seg_attn:
            output_ids, pred_masks, seg_attn = result
        else:
            output_ids, pred_masks = result
            seg_attn = None
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)

        base_name = image_path.split("/")[-1].split(".")[0]

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask_all = pred_mask.detach().cpu().numpy()
            # j-th mask <-> j-th [SEG] token (same order as seg_attn). j == 0 keeps
            # the original file names; further [SEG] tokens get a "_seg{j}" suffix.
            for j in range(pred_mask_all.shape[0]):
                pred_mask = pred_mask_all[j]
                pred_mask = pred_mask > 0
                suffix = "" if j == 0 else "_seg{}".format(j)

                save_path = "{}/{}_mask_{}{}.jpg".format(
                    args.vis_save_path, base_name, i, suffix
                )
                cv2.imwrite(save_path, pred_mask * 100)
                print("{} has been saved.".format(save_path))

                save_path = "{}/{}_masked_img_{}{}.jpg".format(
                    args.vis_save_path, base_name, i, suffix
                )
                save_img = image_np.copy()
                save_img[pred_mask] = (
                    image_np * 0.5
                    + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                )[pred_mask]
                save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, save_img)
                print("{} has been saved.".format(save_path))

        if use_seg_attn:
            save_seg_attention(seg_attn, image_np, base_name, args, clip_image_processor)


if __name__ == "__main__":
    main(sys.argv[1:])