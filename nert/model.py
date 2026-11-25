from contextlib import contextmanager
from copy import deepcopy
from copy import copy
from pathlib import Path

import ultralytics
from ultralytics import YOLO
from ultralytics.nn import tasks
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import yaml_load

from .modules import C3k2DMO, C2LS, NFEH


@contextmanager
def registered(cfg):
    if ultralytics.__version__ != "8.3.0":
        raise RuntimeError("This integration targets ultralytics==8.3.0; install requirements.txt")
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)
    names = {layer[2] for layer in cfg["backbone"] + cfg["head"]}
    replacements = {}
    for name, original, cls in [("C3k2DMO", "C3k2", C3k2DMO), ("C2LS", "C2PSA", C2LS),
                                ("NFEH", "v10Detect", NFEH)]:
        if name in names:
            replacements[name] = cls
            replacements[original] = cls
    saved = {name: getattr(tasks, name, None) for name in replacements}
    try:
        for name, cls in replacements.items():
            setattr(tasks, name, cls)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                delattr(tasks, name)
            else:
                setattr(tasks, name, value)


class NERTTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        with registered(cfg):
            return super().get_model(cfg, weights, verbose)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return NERTValidator(self.test_loader, save_dir=self.save_dir,
                             args=copy(self.args), _callbacks=self.callbacks)


class NERTValidator(DetectionValidator):
    def __call__(self, trainer=None, model=None):
        # During training the validator also computes the dual-branch loss.
        head = None
        if trainer is not None:
            target = trainer.ema.ema if trainer.ema else trainer.model
            candidate = target.model[-1]
            if isinstance(candidate, NFEH):
                head = candidate
                head.validation_loss = True
        try:
            return super().__call__(trainer=trainer, model=model)
        finally:
            if head is not None:
                head.validation_loss = False


class NERTYOLO(YOLO):
    def __init__(self, model="configs/nert.yaml", **kwargs):
        if str(model).endswith((".yaml", ".yml")):
            with registered(model):
                super().__init__(model=model, task="detect", **kwargs)
        else:
            super().__init__(model=model, task="detect", **kwargs)
        # Upstream 8.3.0 save() expects a checkpoint mapping even for YAML-built models.
        if self.ckpt is None:
            self.ckpt = {}

    @property
    def task_map(self):
        mapping = deepcopy(super().task_map)
        mapping["detect"]["trainer"] = NERTTrainer
        mapping["detect"]["validator"] = NERTValidator
        return mapping
