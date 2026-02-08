"""Mask R-CNN training with Detectron2.

Usage:
    python train_maskrcnn.py --strategy strategy1
    python train_maskrcnn.py --strategy strategy2 --lr 0.0005 --epochs 50
"""

import argparse
import os
from pathlib import Path

import torch

from src.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMG_SIZE,
    DEFAULT_PATIENCE,
    NUM_CLASSES,
    OUTPUT_DIR,
    TARGET_CLASSES,
)
from src.dataset import SampleRegistry
from src.formats.coco_format import export_coco_dataset
from src.splits import get_split, print_split_summary

# Detectron2 imports
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator


class EarlyStoppingHook(HookBase):
    """Early stopping based on validation mAP."""

    def __init__(self, patience: int = 15, eval_period: int = 500):
        self.patience = patience
        self.eval_period = eval_period
        self.best_map = 0.0
        self.wait = 0

    def after_step(self):
        if (self.trainer.iter + 1) % self.eval_period == 0:
            # Check if validation metrics improved
            storage = self.trainer.storage
            try:
                current_map = storage.latest().get("segm/AP", (0,))[0]
                if current_map > self.best_map:
                    self.best_map = current_map
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        print(f"\nEarly stopping at iter {self.trainer.iter}. "
                              f"Best mAP: {self.best_map:.4f}")
                        raise StopIteration
            except (KeyError, TypeError):
                pass


class Trainer(DefaultTrainer):
    """Custom trainer with COCO evaluator."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_dir = os.path.join(cfg.OUTPUT_DIR, "eval")
        os.makedirs(output_dir, exist_ok=True)
        return COCOEvaluator(dataset_name, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN training (Detectron2)")
    parser.add_argument("--strategy", default="strategy1",
                        choices=["strategy1", "strategy2", "strategy3"])
    parser.add_argument("--species", default=None)
    parser.add_argument("--backbone", default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                        help="Detectron2 model zoo config")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    # Setup
    registry = SampleRegistry()
    split = get_split(args.strategy, registry, seed=args.seed, species=args.species)
    print_split_summary(split)

    # Export to COCO format
    run_name = f"maskrcnn_{args.strategy}"
    if args.species:
        run_name += f"_{args.species}"
    export_dir = OUTPUT_DIR / "coco_dataset" / run_name
    json_paths = export_coco_dataset(split, export_dir, img_size=args.img_size)

    if args.export_only:
        return

    # Register datasets
    for subset in ["train", "val", "test"]:
        ds_name = f"plants_{subset}"
        if ds_name in DatasetCatalog.list():
            DatasetCatalog.remove(ds_name)
        register_coco_instances(
            ds_name,
            {},
            str(json_paths[subset]),
            str(export_dir / subset),
        )

    # Estimate iterations
    n_train = len(split["train"])
    iters_per_epoch = max(1, n_train // args.batch_size)
    max_iter = iters_per_epoch * args.epochs
    eval_period = iters_per_epoch * 2  # eval every 2 epochs

    # Configure
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.backbone))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.backbone)

    cfg.DATASETS.TRAIN = ("plants_train",)
    cfg.DATASETS.TEST = ("plants_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))
    cfg.SOLVER.WARMUP_ITERS = min(1000, iters_per_epoch * 3)
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.CHECKPOINT_PERIOD = eval_period

    cfg.INPUT.MIN_SIZE_TRAIN = (args.img_size,)
    cfg.INPUT.MAX_SIZE_TRAIN = args.img_size
    cfg.INPUT.MIN_SIZE_TEST = args.img_size
    cfg.INPUT.MAX_SIZE_TEST = args.img_size

    cfg.TEST.EVAL_PERIOD = eval_period

    cfg.OUTPUT_DIR = str(OUTPUT_DIR / "runs" / "maskrcnn" / run_name)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.SEED = args.seed

    # Train
    trainer = Trainer(cfg)
    trainer.register_hooks([
        EarlyStoppingHook(patience=args.patience, eval_period=eval_period),
    ])
    trainer.resume_or_load(resume=False)

    try:
        trainer.train()
    except StopIteration:
        pass  # early stopping

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    cfg.DATASETS.TEST = ("plants_test",)
    evaluator = COCOEvaluator("plants_test", output_dir=cfg.OUTPUT_DIR)
    results = Trainer.test(cfg, trainer.model, evaluators=[evaluator])
    print(f"\nTest results: {results}")


if __name__ == "__main__":
    main()
