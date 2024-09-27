from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from rl_baselines.data.dummy_data import DummyRLDataModule
from rl_baselines.policy_gradient.reinforce_discrete import ReinforceDiscreteSystem

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", type=str, required=True)
    parser.add_argument("--train", dest="train", action='store_true', default=False)
    parser.add_argument("--test", dest="test", action='store_true', default=False)
    parser.add_argument("--resume", dest="resume", type=str, default=None, required=False)

    args, extras = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, OmegaConf.from_cli(extras))

    # TODO: improve this
    system_class = ReinforceDiscreteSystem if config.system.type == 'reinforce-discrete' else None
    data_class = DummyRLDataModule if config.system.type == 'reinforce-discrete' else None

    model = system_class.from_config(config)
    dm = data_class(config)

    if args.train:
        model.allocate_saving_folders()
        logger = TensorBoardLogger(model.cfg.project_folder, "logs")
    else:
        model = model.eval()
        logger=None

    if args.resume:
        model.load(args.resume)
    max_epochs = config.trainer.pop('max_episodes')
    config.trainer.max_epochs = max_epochs

    trainer = pl.Trainer(**config.trainer, logger=logger)
    if args.train:
        trainer.fit(model, dm)
    elif args.test:
        model.test_rollout()
    else:
        raise NotImplementedError("")


