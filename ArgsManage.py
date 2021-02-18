from ImportModule import *


def args_manage():
    # ====== Random Seed Initialization ====== #
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()
    args = parser.parse_args("")
    args.exp_name = "exp1_lr_model_code"

    # ====== Model ====== #
    args.model_code = 'VGG11'
    args.in_channels = 3
    args.out_dim = 10
    args.act = 'relu'

    # ====== Regularization ======= #
    args.l2 = 0.00001
    args.use_bn = True
    args.use_cuda = True

    # ====== Optimizer & Training ====== #
    args.optim = 'RMSprop' #'RMSprop' #SGD, RMSprop, ADAM...
    args.lr = 0.0001
    args.epoch = 10

    args.train_batch_size = 512
    args.test_batch_size = 1024

    # ====== Experiment Variable ====== #
    name_var1 = 'lr'
    name_var2 = 'model_code'
    list_var1 = [0.0001, 0.00001]
    list_var2 = ['VGG11', 'VGG13']

    return args
