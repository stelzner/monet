from collections import namedtuple

config_options = [
    # Training config
    'vis_every',  # Visualize progress every X iterations
    'batch_size',
    'num_epochs',
    'load_parameters',  # Load parameters from checkpoint
    'checkpoint_file',  # File for loading/storing checkpoints
    'data_dir',  # Directory for the training data
    # Model config
    'num_slots',  # Number of slots k,
    'num_blocks',  # Number of blochs in attention U-Net 
    'channel_base',  # Number of channels used for the first U-Net conv layer
    'bg_sigma',  # Sigma of the decoder distributions for the first slot
    'fg_sigma',  # Sigma of the decoder distributions for all other slots
]

MonetConfig = namedtuple('MonetConfig', config_options)

sprite_config = MonetConfig(vis_every=50,
                            batch_size=64,
                            num_epochs=20,
                            load_parameters=True,
                            checkpoint_file='./checkpoints/sprites.ckpt',
                            data_dir='./data/',
                            num_slots=4,
                            num_blocks=5,
                            channel_base=64,
                            bg_sigma=0.09,
                            fg_sigma=0.11,
                           )

clevr_config = MonetConfig(vis_every=50,
                           batch_size=20,
                           num_epochs=20,
                           load_parameters=True,
                           checkpoint_file='./checkpoints/clevr.ckpt',
                           data_dir='~/data/CLEVR_v1.0/images/train/',
                           num_slots=11,
                           num_blocks=6,
                           channel_base=16,
                           bg_sigma=0.09,
                           fg_sigma=0.11,
                          )



