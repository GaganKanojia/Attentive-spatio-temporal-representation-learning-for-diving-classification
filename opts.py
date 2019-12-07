import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--root_path',
        default='/media/parker/Zone_A/diving_cvprw',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--result_path',
        default='models_lstm_64_augumented_test',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--video_path',
        default='/media/parker/Zone_A/Resound_action_final',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--resume_epoch',
        default=-1,
        type=int,
        help='resume from the saved model')
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)

    parser.add_argument(
        '--repre_learning', default=0, type=int, help='set 1 to perform representation learning else 0')
    parser.add_argument(
        '--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument(
        '--sample_duration',
        default=64,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--n_threads',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--n_epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')

    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument(
        '--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', default=1e-3, type=float, help='Weight Decay')

    parser.add_argument(
        '--input_size_encoder_attn', default=1024, type=int, help='input size of the encoder and attention network')
    parser.add_argument(
        '--hidden_size', default=512, type=int, help='size of hidden states of encoder')
    parser.add_argument(
        '--hidden_size_decoder', default=256, type=int, help='size of hidden states of decoder')
    parser.add_argument(
        '--num_layers', default=3, type=int, help='number of layers for encoder, attention network and decoder')

    args = parser.parse_args()

    return args
