'''
Created on Oct 10, 2019
Tensorflow Implementation of Disentangled Graph Collaborative Filtering (DGCF) model in:
Wang Xiang et al. Disentangled Graph Collaborative Filtering. In SIGIR 2020.
Note that: This implementation is based on the codes of NGCF.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run DGCF.")
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')
    
    parser.add_argument('--pick', type=int, default=0,
                        help='O for no pick, 1 for pick')
    parser.add_argument('--pick_scale', type=float, default=1e10,
                        help='Scale')
    parser.add_argument('--dataset', nargs='?', default='gowalla',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1:Use stored models.')
    parser.add_argument('--embed_name', nargs='?', default='',
                        help='Name for pretrained model.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')


    parser.add_argument('--epoch', type=int, default=3000,
                        help='Number of epochs')      
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--cor_flag', type=int, default=1,
                        help='Correlation matrix flag')
    parser.add_argument('--corDecay', type=float, default=0.0,
                        help='Distance Correlation Weight')
    parser.add_argument('--regs', nargs='?', default='[1e-3,1e-4,1e-4]',
                        help='Regularizations.')
        
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Layer numbers.')
    parser.add_argument('--n_factors', type=int, default=4,
                        help='Number of factors to disentangle the original embed-size representation.')
    parser.add_argument('--n_iterations', type=int, default=2,
                        help='Number of iterations to perform the routing mechanism.')
    
    
    parser.add_argument('--show_step', type=int, default=15,
                        help='Test every show_step epochs.')
    parser.add_argument('--early', type=int, default=40,
                        help='Step for stopping')           
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Metrics scale')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Save Better Model')
    parser.add_argument('--save_name', nargs='?', default='best_model',
                        help='Save_name.')
    
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')


    return parser.parse_args()
