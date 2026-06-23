import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, we currently support only long_term_forecast, options:[long_term_forecast]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='temp', help='model id')
    parser.add_argument('--model', type=str, default='RAFT',
                        help='model name, options: [RAFT]')
    parser.add_argument('-Phase', '--Phase', '--phase', action='store_true', dest='phase',
                        help='activate phase-aware RAFT retrieval and phase-domain fusion; '
                             'defaults to retrieval_variant C')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument(
        '--output_attention', action='store_true',
        help='whether to output attention in ecoder'
    )
    parser.add_argument(
        '--n_period', type=int, default=3,
        help='Number of Periods'
    )
    parser.add_argument(
        '--topm', type=int, default=20,
        help='shape retrieval top-k candidate count'
    )
    parser.add_argument(
        '--retrieval_variant', type=str, default='A', choices=['A', 'B', 'C'],
        help='A: RAFT top-k; B: shape top-k then phase hard top-m; '
             'C: shape top-k then shape+phase soft re-rank top-m'
    )
    parser.add_argument(
        '--phase_top_m', '--top_m', type=int, default=5, dest='phase_top_m',
        help='final top-m candidates kept after phase-aware selection'
    )
    parser.add_argument(
        '--phase_lambda', type=float, default=0.1,
        help='lambda for C: score_final = score_shape + lambda * score_phase'
    )
    parser.add_argument(
        '--phase_tau', type=float, default=2.0,
        help='temperature for phase score exp(-d_phase / tau)'
    )
    parser.add_argument(
        '--phase_period', type=int, default=None,
        help='phase period P; defaults to --period_len'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.1,
        help='retrieval softmax temperature'
    )
    parser.add_argument(
        '--period_len', type=int, default=24,
        help='default phase period; used when --phase_period is unset'
    )
    parser.add_argument(
        '--cycle', type=int, default=None,
        help='compatibility alias for --period_len'
    )
    parser.add_argument('--phase_fusion', '--phase-fusion', action='store_true', default=False,
                        help='enable phase-domain fusion after RAFT retrieval')
    parser.add_argument('--phase_fusion_mode', choices=['residual', 'backbone'], default='residual',
                        help='residual: add a gated phase residual to RAFT; '
                             'backbone: use the phase-domain predictor as the main output head')
    parser.add_argument('--phase_fusion_scale', type=float, default=0.1,
                        help='max scale of phase-domain residual fusion')
    parser.add_argument('--no-phase-fusion', '--no_phase_fusion',
                        '--no-phase-routing', '--no_phase_routing',
                        action='store_true', dest='disable_phase_fusion', default=False,
                        help='disable phase-domain fusion even when -Phase is used')
    parser.add_argument('-phase_block', '--phase_block', '--phase-block',
                        action='store_true', dest='phase_block', default=False,
                        help='enable phase-conditioned residual corrector at test time')
    parser.add_argument('--phase_block_periods', type=str, default=None,
                        help='comma-separated PRC periods; if unset, periods are estimated by FFT')
    parser.add_argument('--phase_block_max_periods', type=int, default=2,
                        help='maximum number of FFT-estimated periods used by PRC')
    parser.add_argument('--phase_block_min_strength', type=float, default=0.05,
                        help='minimum FFT power ratio required to keep an estimated PRC period')
    parser.add_argument('--phase_block_patch', type=int, default=3,
                        help='local micro-patch width around each phase point')
    parser.add_argument('--phase_block_cycles', type=int, default=4,
                        help='number of previous phase-aligned cycles in each PRC key')
    parser.add_argument('--phase_block_topk', type=int, default=5,
                        help='number of residual neighbors retrieved by PRC')
    parser.add_argument('--phase_block_temperature', type=float, default=0.1,
                        help='softmax temperature for PRC residual aggregation')
    parser.add_argument('--phase_block_alpha', type=float, default=0.2,
                        help='maximum PRC residual injection scale')
    parser.add_argument('--phase_block_sim_threshold', type=float, default=0.3,
                        help='nearest-neighbor similarity threshold below which PRC is gated off')
    parser.add_argument('--phase_block_residual_var_scale', type=float, default=1.0,
                        help='penalty scale for disagreement among retrieved residuals')
    parser.add_argument('--phase_block_stride', type=int, default=1,
                        help='stride for storing residuals into the PRC datastore')
    parser.add_argument('--phase_block_bank_mode', choices=['full', 'single'], default='full',
                        help='full: store one PhaseBlock residual per forecast step; '
                             'single: legacy one-key-per-window residual bank')
    parser.add_argument('--phase_block_query_chunk', type=int, default=256,
                        help='query chunk size for full PhaseBlock retrieval')
    parser.add_argument('--phase_block_memory_source', choices=['val', 'train_roll_val'], default='val',
                        help='PRC datastore source: val is strictest; train_roll_val also uses the later train windows')
    parser.add_argument('--phase_block_train_init_ratio', type=float, default=0.7,
                        help='when using train_roll_val, train windows after this ratio enter the rolling memory')
    parser.add_argument('--use_revin', action='store_true', default=False,
                        help='use RevIN (mean/std instance norm) instead of subtract-last offset')
    parser.add_argument('--diversity_aware', action='store_true', default=False,
                        help='SARAF-style diversity-aware retrieval: time-NMS selection + '
                             'reliability features (top-k sim strength & neighbor disagreement)')
    parser.add_argument('--retrieval_pool', type=int, default=100,
                        help='candidate pool size (top-N) before diversity selection')
    parser.add_argument('--nms_gap', type=int, default=None,
                        help='min index gap between selected neighbors in time-NMS '
                             '(defaults to pred_len)')
    parser.add_argument('--rel_topk', type=int, default=5,
                        help='number of top neighbors used to estimate similarity strength')
    parser.add_argument('--period_list', type=str, default=None,
                        help='comma-separated phase periods for multi-period routing, '
                             'e.g. "24,168". Defaults to --period_len if unset.')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='latent dim of cross-phase routing branch')
    parser.add_argument('--phase_layers', type=int, default=1,
                        help='number of cross-phase routing layers')
    parser.add_argument('--phase_num_routers', type=int, default=8,
                        help='number of routers in cross-phase routing')
    parser.add_argument('--phase_heads', type=int, default=4,
                        help='attention heads in cross-phase routing')
    parser.add_argument('--phase_attn_dropout', type=float, default=0.1,
                        help='dropout in cross-phase routing branch')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=0, help="Randomization seed")
    parser.add_argument('--random_seed', type=int, default=None, help="Compatibility alias for --seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    args = parser.parse_args()
    if args.phase:
        args.model = 'RAFT'
        if args.retrieval_variant == 'A':
            args.retrieval_variant = 'C'
        if not args.disable_phase_fusion:
            args.phase_fusion = True
    if args.disable_phase_fusion:
        args.phase_fusion = False
    if args.phase_fusion_mode == 'backbone' and not args.disable_phase_fusion:
        args.phase_fusion = True
    if args.cycle is not None:
        args.period_len = args.cycle
    if args.phase_period is None:
        args.phase_period = args.period_len
    if args.random_seed is not None:
        args.seed = args.random_seed
    if args.topm <= 0:
        raise ValueError('--topm must be positive')
    if args.phase_top_m <= 0:
        raise ValueError('--phase_top_m must be positive')
    if args.phase_fusion_scale < 0:
        raise ValueError('--phase_fusion_scale must be non-negative')
    if args.phase_block_max_periods <= 0:
        raise ValueError('--phase_block_max_periods must be positive')
    if args.phase_block_patch <= 0:
        raise ValueError('--phase_block_patch must be positive')
    if args.phase_block_cycles <= 0:
        raise ValueError('--phase_block_cycles must be positive')
    if args.phase_block_topk <= 0:
        raise ValueError('--phase_block_topk must be positive')
    if args.phase_block_temperature <= 0:
        raise ValueError('--phase_block_temperature must be positive')
    if args.phase_block_alpha < 0:
        raise ValueError('--phase_block_alpha must be non-negative')
    if args.phase_block_stride <= 0:
        raise ValueError('--phase_block_stride must be positive')
    if args.phase_block_query_chunk <= 0:
        raise ValueError('--phase_block_query_chunk must be positive')
    if not 0 <= args.phase_block_train_init_ratio < 1:
        raise ValueError('--phase_block_train_init_ratio must be in [0, 1)')
    if args.retrieval_variant != 'A' and args.phase_tau <= 0:
        raise ValueError('--phase_tau must be positive for phase-aware retrieval')
    if args.retrieval_variant != 'A' and args.phase_period <= 0:
        raise ValueError('--phase_period must be positive for phase-aware retrieval')
    if args.retrieval_variant != 'A':
        args.des = '{}_ret{}_m{}_lam{}_tau{}'.format(
            args.des,
            args.retrieval_variant,
            args.phase_top_m,
            args.phase_lambda,
            args.phase_tau,
        )
    if args.phase_fusion:
        args.des = '{}_pf{}_{}'.format(args.des, args.period_list or args.period_len, args.phase_fusion_mode)
    if args.phase_block:
        args.des = '{}_pb{}_k{}_a{}'.format(
            args.des,
            args.phase_block_periods or 'fft',
            args.phase_block_topk,
            args.phase_block_alpha,
        )

    fix_seed = args.seed
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    else:
        # We support only long term forecasting.
        assert(0)

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
