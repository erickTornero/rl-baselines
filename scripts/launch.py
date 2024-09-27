import sys
from os.path import relpath
import os
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint
from rl_baselines.data.dummy_data import DummyRLDataModule
from rl_baselines.policy_gradient.reinforce_discrete import ReinforceDiscreteSystem

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--train", dest="train", action='store_true', default=False)
    parser.add_argument("--test", dest="test", action='store_true', default=False)
    parser.add_argument("--resume", dest="resume", type=str, default=None, required=False)
    parser.add_argument("--save-video", dest="save_video", action='store_true', default=False)


    args, extras = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, OmegaConf.from_cli(extras))

    # TODO: improve this
    system_class = ReinforceDiscreteSystem if config.system.type == 'reinforce-discrete' else None
    data_class = DummyRLDataModule if config.system.type == 'reinforce-discrete' else None

    model = system_class.from_config(config)
    dm = data_class(config)
    callbacks = []
    if args.train:
        model.allocate_saving_folders()
        logger = [
            TensorBoardLogger(model.working_dir, "tb_logs"),
            CSVLogger(
                save_dir=model.working_dir,
                name="csv_logs"
            )
        ]
        ckpt_callback = ModelCheckpoint(
                dirpath=model.get_absolute_path("ckpts"),
                filename="{epoch:02d}-{Episode Reward:.2f}",
                mode='max',
                **config.checkpoint
            )
        callbacks += [
            ckpt_callback,
        ]
        rank_zero_only(
            lambda: model.save_cmd(
                "cmd.txt",
                "python " + " ".join(sys.argv) + "\n" + str(args)
            )
        )()
    else:
        model = model.eval()
        logger=None

    if args.resume:
        model.load(args.resume)
    elif args.test and 'best_ckpt' in config and os.path.exists(config.best_ckpt):
        model.load(config.best_ckpt)

    trainer = pl.Trainer(
        max_epochs=config.trainer.max_episodes,
        callbacks=callbacks,
        logger=logger
    )
    if args.train:
        trainer.fit(model, dm)
        best_ckpt = relpath(ckpt_callback.best_model_path)
        config.best_ckpt = best_ckpt
        model.save_cfg("parsed.yaml")
    elif args.test:
        model.test_rollout(save_video=args.save_video)
    else:
        raise NotImplementedError("")


