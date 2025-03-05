# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser
import inspect
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import ArgoverseV1DataModule
from models.hivt import HiVT

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=5)
    parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min')
    
    # Gets a list of valid arguments for the Trainer constructor
    sig = inspect.signature(pl.Trainer)
    valid_args = list(sig.parameters.keys())
    # Filter parameters that match the Trainer in the args
    trainer_kwargs = {k: v for k, v in vars(args).items() if k in valid_args}
    trainer = pl.Trainer(**trainer_kwargs, callbacks=[model_checkpoint])
    
    model = HiVT(**vars(args))
    
    #Gets the list of valid arguments for the DataModule constructor
    valid_args = inspect.getfullargspec(ArgoverseV1DataModule.__init__).args
    # Filter invalid parameters in args 
    datamodule_kwargs = {k: v for k, v in vars(args).items() if k in valid_args}
    #Instantiate the DataModule
    datamodule = ArgoverseV1DataModule(**datamodule_kwargs)
    
    trainer.fit(model, datamodule)
