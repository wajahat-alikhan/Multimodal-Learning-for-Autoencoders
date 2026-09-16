import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from transformers import (
    AutoTokenizer,
    CLIPModel,
    CLIPTokenizer,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
)


CLASS_DESCRIPTIONS = (
    "a sleek flying vehicle with large wings",
    "a shiny four-wheeled motor vehicle for transportation",
    "a vibrant feathered flying animal with colorful plumage",
    "a fluffy small domesticated feline pet",
    "a graceful wild animal with large antlers",
    "a playful domesticated canine pet with a shiny coat",
    "a small green amphibious jumping animal with smooth skin",
    "a majestic large four-legged animal with a flowing mane",
    "a large sturdy boat traveling on water",
    "a heavy-duty motor vehicle for transporting goods",
)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class MultimodalAutoencoder(nn.Module):
    """Decode image-only, text-only, or fused CLIP features at 32×32."""

    def __init__(self, clip_dim: int, mode: str) -> None:
        super().__init__()
        self.mode = mode
        fusion_input_dim = clip_dim * 2 if mode == "both" else clip_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, clip_dim),
            nn.GELU(),
            nn.Linear(clip_dim, clip_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(clip_dim, 256 * 4 * 4),
            nn.GELU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8×8
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 16×16
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 32×32
            nn.GELU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "both":
            if text_features is None:
                raise ValueError("both mode requires text features")
            latent = torch.cat((image_features, text_features), dim=1)

        elif self.mode == "image_only":
            latent = image_features

        else:  # text_only
            if text_features is None:
                raise ValueError("text_only mode requires text features")
            latent = text_features

        return self.decoder(self.fusion(latent))


class IndexedDataset(Dataset):
    """Returns the source index so captions/features can be cached."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        image, label = self.dataset[index]
        return image, label, index


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=-1, eps=1e-6)


def prepare_clip_images(images: torch.Tensor) -> torch.Tensor:
    """Resize [0, 1] RGB images and apply official CLIP normalization."""
    images = F.interpolate(
        images,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)

    return (images - mean) / std


def clip_image_features(clip: CLIPModel, images: torch.Tensor) -> torch.Tensor:
    """
    Do not wrap this in torch.no_grad() for generated decoder images.
    CLIP weights stay frozen, but gradient must reach generated pixels.
    """
    return clip.get_image_features(pixel_values=prepare_clip_images(images))


def encode_texts(
    clip: CLIPModel,
    tokenizer: CLIPTokenizer,
    texts: List[str],
    device: torch.device,
) -> torch.Tensor:
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    tokens = {name: value.to(device) for name, value in tokens.items()}

    with torch.no_grad():
        return clip.get_text_features(**tokens)


class CaptionCache:
    """Persistent per-image captions, kept outside the training loop."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path

        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as cache_file:
                self.captions: Dict[str, str] = json.load(cache_file)
        else:
            self.captions = {}

    def get(self, indices: torch.Tensor) -> List[str]:
        indices_list = indices.detach().cpu().tolist()
        missing = [
            str(index)
            for index in indices_list
            if str(index) not in self.captions
        ]

        if missing:
            raise RuntimeError(
                "Caption cache is incomplete. Run --prepare-captions before "
                "training with --text-source generated_caption."
            )

        return [self.captions[str(index)] for index in indices_list]

    def require_complete(self, dataset_size: int) -> None:
        missing_count = sum(
            str(index) not in self.captions
            for index in range(dataset_size)
        )

        if missing_count:
            raise RuntimeError(
                f"Caption cache is incomplete ({missing_count} examples missing). "
                "Run --prepare-captions without --caption-limit before training."
            )

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.cache_path.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as cache_file:
            json.dump(
                self.captions,
                cache_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

        temporary_path.replace(self.cache_path)


@torch.no_grad()
def populate_caption_cache(
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cache: CaptionCache,
    device: torch.device,
    save_every: int,
    preview_count: int,
    limit: int,
) -> None:
    """Generate captions once, saving the cache periodically."""

    image_processor = ViTImageProcessor.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    )
    captioner = VisionEncoderDecoderModel.from_pretrained(
        "nlpconnect/vit-gpt2-image-captioning"
    ).to(device)

    captioner.requires_grad_(False)
    captioner.eval()

    generated_count = 0
    printed_count = 0

    for batch_number, (images, _labels, indices) in enumerate(loader, start=1):
        indices_list = indices.tolist()

        missing_positions = [
            position
            for position, index in enumerate(indices_list)
            if str(index) not in cache.captions
        ]

        if not missing_positions:
            continue

        if limit > 0:
            remaining = limit - generated_count

            if remaining <= 0:
                break

            missing_positions = missing_positions[:remaining]

        pil_images = [
            transforms.ToPILImage()(images[position])
            for position in missing_positions
        ]

        pixel_values = image_processor(
            images=pil_images,
            return_tensors="pt",
        ).pixel_values

        generated_ids = captioner.generate(
            pixel_values.to(device),
            max_new_tokens=32,
        )

        captions = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        for position, caption in zip(missing_positions, captions):
            caption = caption.strip()
            cache.captions[str(indices_list[position])] = caption
            generated_count += 1

            if printed_count < preview_count:
                print(f"caption[{indices_list[position]}]: {caption}")
                printed_count += 1

        if batch_number % save_every == 0:
            cache.save()
            print(f"Saved {len(cache.captions)} cached captions.")

    cache.save()

    del captioner

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Caption preparation complete: {len(cache.captions)} cached captions.")


def assert_semantic_gradient_path(
    model: MultimodalAutoencoder,
    clip: CLIPModel,
    class_text_features: torch.Tensor,
    device: torch.device,
) -> None:
    """Fail if semantic loss cannot reach generated image pixels."""

    model.eval()

    probe_images = torch.rand(2, 3, 32, 32, device=device)
    probe_labels = torch.tensor([0, 1], device=device)

    with torch.no_grad():
        source_features = clip_image_features(clip, probe_images)
        text_features = class_text_features[probe_labels]

    generated_images = model(source_features, text_features)
    generated_features = clip_image_features(clip, generated_images)

    semantic_target = normalize_features(text_features)

    semantic_loss = 1.0 - (
        normalize_features(generated_features) * semantic_target
    ).sum(dim=1).mean()

    pixel_gradient = torch.autograd.grad(
        semantic_loss,
        generated_images,
        allow_unused=True,
    )[0]

    if (
        pixel_gradient is None
        or not torch.isfinite(pixel_gradient).all()
        or pixel_gradient.abs().sum() == 0
    ):
        raise RuntimeError(
            "Semantic loss has no usable gradient path to generated images."
        )

    print("Gradient-path assertion passed.")


def make_loaders(
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = IndexedDataset(
        datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
    )

    train_size = int(0.9 * len(dataset))

    train_set, validation_set = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(args.split_seed),
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.workers > 0,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.init_seed),
        **loader_kwargs,
    )

    validation_loader = DataLoader(
        validation_set,
        shuffle=False,
        **loader_kwargs,
    )

    source_loader = DataLoader(
        dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, validation_loader, source_loader


@torch.no_grad()
def precompute_source_features(
    clip: CLIPModel,
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    dataset_size: int,
) -> torch.Tensor:
    """Compute fixed source-image CLIP features once for all epochs."""

    feature_cache = torch.empty(
        dataset_size,
        clip.config.projection_dim,
        device=device,
    )

    for images, _labels, indices in loader:
        feature_cache[indices.to(device)] = clip_image_features(
            clip,
            images.to(device, non_blocking=True),
        )

    return feature_cache


def run_epoch(
    model: MultimodalAutoencoder,
    clip: CLIPModel,
    clip_tokenizer: CLIPTokenizer,
    class_text_features: torch.Tensor,
    caption_cache: Optional[CaptionCache],
    source_feature_cache: Optional[torch.Tensor],
    loader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    optimizer: Optional[AdamW],
    semantic_weight: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "semantic": 0.0,
        "psnr": 0.0,
    }

    examples = 0

    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]

        # Every mode shares identical text supervision. image_only does not
        # receive text at the decoder input, making it a conservative baseline.
        with torch.no_grad():
            if source_feature_cache is None:
                source_image_features = clip_image_features(clip, images)
            else:
                source_image_features = source_feature_cache[indices.to(device)]

            if caption_cache is None:
                text_features = class_text_features[labels]
            else:
                captions = caption_cache.get(indices)
                text_features = encode_texts(
                    clip,
                    clip_tokenizer,
                    captions,
                    device,
                )

            decoder_text_features = (
                None if model.mode == "image_only" else text_features
            )

        if training:
            optimizer.zero_grad(set_to_none=True)

            generated_images = model(
                source_image_features,
                decoder_text_features,
            )

            # semantic_weight=0 skips backward through CLIP.
            if semantic_weight == 0:
                with torch.no_grad():
                    generated_image_features = clip_image_features(
                        clip,
                        generated_images,
                    )
            else:
                # No torch.no_grad here: semantic gradient reaches decoder pixels.
                generated_image_features = clip_image_features(
                    clip,
                    generated_images,
                )

        else:
            with torch.no_grad():
                generated_images = model(
                    source_image_features,
                    decoder_text_features,
                )

                generated_image_features = clip_image_features(
                    clip,
                    generated_images,
                )

        squared_error_per_image = (
            (generated_images - images).square().flatten(1).mean(dim=1)
        )

        reconstruction_loss = squared_error_per_image.mean()
        semantic_target = normalize_features(text_features)

        semantic_loss = 1.0 - (
            normalize_features(generated_image_features) * semantic_target
        ).sum(dim=1).mean()

        loss = reconstruction_loss + semantic_weight * semantic_loss

        if training:
            loss.backward()
            optimizer.step()

        totals["loss"] += loss.detach().item() * batch_size
        totals["reconstruction"] += reconstruction_loss.detach().item() * batch_size
        totals["semantic"] += semantic_loss.detach().item() * batch_size

        totals["psnr"] += (
            10.0
            * torch.log10(1.0 / squared_error_per_image.clamp_min(1e-10))
        ).sum().detach().item()

        examples += batch_size

    return {name: value / examples for name, value in totals.items()}


@torch.no_grad()
def save_reconstruction_grid(
    model: MultimodalAutoencoder,
    clip: CLIPModel,
    clip_tokenizer: CLIPTokenizer,
    class_text_features: torch.Tensor,
    caption_cache: Optional[CaptionCache],
    source_feature_cache: Optional[torch.Tensor],
    validation_loader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    output_path: Path,
) -> None:
    """Save originals above their corresponding validation reconstructions."""

    model.eval()

    images, labels, indices = next(iter(validation_loader))

    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    if source_feature_cache is None:
        source_features = clip_image_features(clip, images)
    else:
        source_features = source_feature_cache[indices.to(device)]

    if caption_cache is None:
        text_features = class_text_features[labels]
    else:
        captions = caption_cache.get(indices)
        text_features = encode_texts(
            clip,
            clip_tokenizer,
            captions,
            device,
        )

    decoder_text_features = (
        None if model.mode == "image_only" else text_features
    )

    generated_images = model(
        source_features,
        decoder_text_features,
    )

    count = min(8, images.shape[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(
        torch.cat(
            (images[:count].cpu(), generated_images[:count].cpu()),
            dim=0,
        ),
        output_path,
        nrow=count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=Path, default=Path("data"))

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/multimodal_autoencoder"),
    )

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--semantic-weight", type=float, default=0.3)

    parser.add_argument(
        "--mode",
        choices=("both", "image_only", "text_only"),
        default="both",
    )

    parser.add_argument(
        "--text-source",
        choices=("class_template", "generated_caption"),
        default="class_template",
    )

    parser.add_argument(
        "--caption-cache",
        type=Path,
        default=Path("data/cifar10_vit_gpt2_captions.json"),
    )

    parser.add_argument(
        "--skip-gradient-check",
        action="store_true",
    )

    parser.add_argument(
        "--prepare-captions",
        action="store_true",
    )

    parser.add_argument(
        "--caption-preview-count",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--caption-limit",
        type=int,
        default=0,
        help="Generate at most this many missing captions; 0 means all.",
    )

    parser.add_argument(
        "--caption-save-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--no-precompute-source-features",
        action="store_true",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--init-seed",
        type=int,
        default=7,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.init_seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader, source_loader = make_loaders(args)

    clip = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device)

    clip.requires_grad_(False)
    clip.eval()

    clip_tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    class_text_features = encode_texts(
        clip,
        clip_tokenizer,
        list(CLASS_DESCRIPTIONS),
        device,
    )

    caption_cache = None

    if args.text_source == "generated_caption":
        caption_cache = CaptionCache(args.caption_cache)

        if args.prepare_captions:
            populate_caption_cache(
                source_loader,
                caption_cache,
                device,
                args.caption_save_every,
                args.caption_preview_count,
                args.caption_limit,
            )
            return

        caption_cache.require_complete(len(source_loader.dataset))

    model = MultimodalAutoencoder(
        clip.config.projection_dim,
        args.mode,
    ).to(device)

    if not args.skip_gradient_check:
        assert_semantic_gradient_path(
            model,
            clip,
            class_text_features,
            device,
        )

    source_feature_cache = None

    if not args.no_precompute_source_features:
        print("Precomputing source-image CLIP features.")

        source_feature_cache = precompute_source_features(
            clip,
            source_loader,
            device,
            len(source_loader.dataset),
        )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    best_validation_mse = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            clip,
            clip_tokenizer,
            class_text_features,
            caption_cache,
            source_feature_cache,
            train_loader,
            device,
            optimizer,
            args.semantic_weight,
        )

        validation_metrics = run_epoch(
            model,
            clip,
            clip_tokenizer,
            class_text_features,
            caption_cache,
            source_feature_cache,
            validation_loader,
            device,
            None,
            args.semantic_weight,
        )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_mse={validation_metrics['reconstruction']:.4f} "
            f"val_psnr={validation_metrics['psnr']:.2f} "
            f"val_semantic={validation_metrics['semantic']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )

        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        if validation_metrics["reconstruction"] < best_validation_mse:
            best_validation_mse = validation_metrics["reconstruction"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_metrics": validation_metrics,
                    "args": vars(args),
                },
                args.output_dir / "best_checkpoint.pt",
            )

    save_reconstruction_grid(
        model,
        clip,
        clip_tokenizer,
        class_text_features,
        caption_cache,
        source_feature_cache,
        validation_loader,
        device,
        args.output_dir / "samples.png",
    )

    torch.save(
        model.state_dict(),
        args.output_dir / "final_model.pt",
    )


if __name__ == "__main__":
    main()
