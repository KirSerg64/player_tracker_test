import argparse
import os
from pathlib import Path

import rich.logging
import torch
import hydra
import warnings
import logging

from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import monkeypatch_hydra, progress, wandb
from tracklab.engine.video import VideoOnlineTrackingEngine

from hydra.utils import instantiate
from omegaconf import OmegaConf


os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="pkg://sn_gamestate.configs", config_name=None)
def main(cfg):
    # with initialize(version_base=None, config_path="pkg://sn_gamestate.configs"б):
        # cfg = compose(config_name=config_name)
        # HydraConfig.instance().set_config(cfg)
    device = init_environment(cfg)

    log.info("Start processing")
    # Instantiate all modules
    tracking_dataset = instantiate(cfg.dataset)
    # evaluator = instantiate(cfg.eval, tracking_dataset=tracking_dataset)

    # log.info(f"{tracking_dataset}")

    modules = []
    if cfg.pipeline is not None:
        for name in cfg.pipeline:
            log.info(f"Loading module: {name}")
            module = cfg.modules[name]
            inst_module = instantiate(module, device=device, tracking_dataset=tracking_dataset)
            modules.append(inst_module)

    pipeline = Pipeline(models=modules)

    tracking_set = tracking_dataset.sets[cfg.dataset.eval_set]
    tracker_state = TrackerState(tracking_set, pipeline=pipeline, **cfg.state)
    tracking_engine = instantiate(
        cfg.engine,
        modules=pipeline,
        tracker_state=tracker_state,
    )

    tracking_engine.track_dataset()

    # # Train tracking modules
    # for module in modules:
    #     if module.training_enabled:
    #         module.train(tracking_dataset, pipeline, evaluator, OmegaConf.to_container(cfg.dataset, resolve=True))

    close_environment()

    return 0


def set_sharing_strategy():
    torch.multiprocessing.set_sharing_strategy(
        "file_system"
    )


def init_environment(cfg):
    # For Hydra and Slurm compatibility
    progress.use_rich = cfg.use_rich
    set_sharing_strategy()  # Do not touch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if cfg.use_rich:
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.ERROR)
        log.root.addHandler(rich.logging.RichHandler(level=logging.INFO))
    else:
        # TODO : Fix for mmcv fix. This should be done in a nicer way
        for handler in log.root.handlers:
            if type(handler) is logging.StreamHandler:
                handler.setLevel(logging.INFO)
    # wandb.init(cfg)
    log.info(f"Run directory: {Path().absolute()}")
    log.info(f"Using device: '{device}'.")

    if cfg.print_config:
        log.info(OmegaConf.to_yaml(cfg))
    return device


def close_environment():
    # wandb.finish()
    return


def evaluate(cfg, evaluator, tracker_state):
    if cfg.get("eval_tracking", True):  # and cfg.dataset.nframes == -1:
        log.info("Starting evaluation.")
        evaluator.run(tracker_state)
    elif not cfg.get("eval_tracking", True):
        log.warning("Skipping evaluation because 'eval_tracking' was set to False.")
    else:
        log.warning(
            "Skipping evaluation because only part of video was tracked (i.e. 'cfg.dataset.nframes' was not set "
            "to -1)"
        )


if __name__ == "__main__":
    main()

