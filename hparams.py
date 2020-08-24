class hparams:
    batch_size = 16
    clip_grad = 2.5
    lr = 3e-4

    training_epochs = 1000
    log_interval = 100
    checkpoint_save_interval = 5

    seed = 1111
    cuda = True

    mode = 'train'

    restore_path = None
