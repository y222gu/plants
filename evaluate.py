"""Unified evaluation entry point.

Run any trained model on a test set and produce metrics CSV/JSON.

Usage:
    python evaluate.py --model yolo --checkpoint output/runs/yolo/run/weights/best.pt
    python evaluate.py --model maskrcnn --checkpoint output/runs/maskrcnn/run/model_final.pth
    python evaluate.py --model unet --checkpoint output/runs/unet/run/checkpoints/best.ckpt
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.annotation_utils import load_sample_annotations
from src.config import DEFAULT_IMG_SIZE, NUM_CLASSES, OUTPUT_DIR, TARGET_CLASSES
from src.dataset import SampleRegistry
from src.evaluation import (
    PredictionResult,
    convert_detectron2_instances,
    convert_semantic_to_instances,
    evaluate_samples,
)
from src.preprocessing import load_sample_normalized, to_uint8
from src.splits import get_split, print_split_summary


def predict_yolo(checkpoint: str, samples, img_size: int) -> dict:
    """Run YOLO inference and return predictions."""
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    predictions = {}

    for sample in samples:
        img = load_sample_normalized(sample)
        img_uint8 = to_uint8(img)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        results = model(img_bgr, imgsz=img_size, verbose=False)[0]

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy().astype(np.uint8)
            labels = results.boxes.cls.cpu().numpy().astype(np.int32)
            scores = results.boxes.conf.cpu().numpy().astype(np.float32)

            # Resize masks to original image size
            h, w = img.shape[:2]
            resized = np.zeros((len(masks), h, w), dtype=np.uint8)
            for i in range(len(masks)):
                resized[i] = cv2.resize(masks[i], (w, h),
                                        interpolation=cv2.INTER_NEAREST)
            masks = resized
        else:
            h, w = img.shape[:2]
            masks = np.zeros((0, h, w), dtype=np.uint8)
            labels = np.zeros(0, dtype=np.int32)
            scores = np.zeros(0, dtype=np.float32)

        predictions[sample.uid] = PredictionResult(
            masks=masks, labels=labels, scores=scores,
        )

    return predictions


def predict_unet(checkpoint: str, samples, img_size: int) -> dict:
    """Run U-Net inference and return predictions."""
    from train_unet import SegmentationModule

    model = SegmentationModule.load_from_checkpoint(checkpoint)
    model.eval()
    model.cuda()

    predictions = {}
    for sample in samples:
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float().cuda()

        with torch.no_grad():
            logits = model(tensor)
        sem_mask = logits.argmax(dim=1).squeeze().cpu().numpy()

        # Resize back
        sem_mask = cv2.resize(sem_mask.astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(np.int32)

        pred = convert_semantic_to_instances(sem_mask)
        predictions[sample.uid] = pred

    return predictions


def predict_maskrcnn(checkpoint: str, samples, img_size: int) -> dict:
    """Run Mask R-CNN inference."""
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = checkpoint
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.INPUT.MIN_SIZE_TEST = img_size
    cfg.INPUT.MAX_SIZE_TEST = img_size

    predictor = DefaultPredictor(cfg)
    predictions = {}

    for sample in samples:
        img = load_sample_normalized(sample)
        h, w = img.shape[:2]
        img_uint8 = to_uint8(img)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        outputs = predictor(img_bgr)
        instances = outputs["instances"].to("cpu")
        pred = convert_detectron2_instances(instances, h, w)
        predictions[sample.uid] = pred

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Unified model evaluation")
    parser.add_argument("--model", required=True,
                        choices=["yolo", "maskrcnn", "unet"],
                        help="Model type")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--strategy", default="strategy1",
                        choices=["strategy1", "strategy2", "strategy3"])
    parser.add_argument("--species", default=None)
    parser.add_argument("--subset", default="test",
                        help="Which split to evaluate on")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for metrics")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup
    registry = SampleRegistry()
    split = get_split(args.strategy, registry, seed=args.seed, species=args.species)
    samples = split[args.subset]
    print(f"Evaluating {args.model} on {len(samples)} {args.subset} samples")

    # Run inference
    predict_fn = {
        "yolo": predict_yolo,
        "maskrcnn": predict_maskrcnn,
        "unet": predict_unet,
    }[args.model]

    print("Running inference...")
    predictions = predict_fn(args.checkpoint, samples, args.img_size)

    # Evaluate
    print("Computing metrics...")
    metrics = evaluate_samples(predictions, samples)
    metrics.print_summary()

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = OUTPUT_DIR / "evaluation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.model}_{args.strategy}_{args.subset}.json"

    metrics.save(out_path)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
