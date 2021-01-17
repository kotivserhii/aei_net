import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from aei_net import AEINet

def main(args):
    model = AEINet()
    save_path = os.path.join('/aeinet_files/chkp', args.name)
    os.makedirs(save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=save_path,
        monitor='val_loss',
        verbose=True,
        save_top_k=-1
    )

    trainer = Trainer(
        logger=pl_loggers.TensorBoardLogger('/aeinet_files/log_dir'),
        early_stop_callback=None,
        checkpoint_callback=checkpoint_callback,
        weights_save_path=save_path,
        gpus=-1,
        num_sanity_val_steps=1,
        resume_from_checkpoint=args.checkpoint_path,
        gradient_clip_val=0,
        progress_bar_refresh_rate=1,
        max_epochs=10000,
    )
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="Name of the run.")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint for resuming")

    args = parser.parse_args()

    main(args)