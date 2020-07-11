 #!/usr/local/bin/bash
'''
Created on Oct 10, 2019
Tensorflow Implementation of Disentangled Graph Collaborative Filtering (DGCF) model in:
Wang Xiang et al. Disentangled Graph Collaborative Filtering. In SIGIR 2020.
Note that: This implementation is based on the codes of NGCF.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''

import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import os
import sys
import random as rd
import pickle
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utility.helper import *
from utility.batch_test import *


class GDCF(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings

        self.pretrain_data = pretrain_data
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 1
        self.norm_adj = data_config['norm_adj']
        self.all_h_list = data_config['all_h_list']
        self.all_t_list = data_config['all_t_list']
        self.A_in_shape = self.norm_adj.tocoo().shape

        self.n_nonzero_elems = self.norm_adj.count_nonzero()
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.n_factors = args.n_factors
        self.n_iterations = args.n_iterations
        self.n_layers = args.n_layers
        self.pick_level = args.pick_scale
        self.cor_flag = args.cor_flag
        if args.pick == 1:
            self.is_pick = True
        else:
            self.is_pick = False

        self.batch_size = args.batch_size
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.users = tfv1.placeholder(tf.int32, shape=(None,))
        self.pos_items = tfv1.placeholder(tf.int32, shape=(None,))
        self.neg_items = tfv1.placeholder(tf.int32, shape=(None,))

        # additional placeholders for the distance correlation
        self.cor_users = tfv1.placeholder(tf.int32, shape=(None,))
        self.cor_items = tfv1.placeholder(tf.int32, shape=(None,))

        # assign different values with different factors (channels).
        self.A_values = tfv1.placeholder(tf.float32, shape=[self.n_factors, len(self.all_h_list)], name='A_values')

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        # create models
        self.ua_embeddings, self.ia_embeddings, self.f_weight, self.ua_embeddings_t, self.ia_embeddings_t = self._create_star_routing_embed_with_P(pick_=self.is_pick)

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.u_g_embeddings_t = tf.nn.embedding_lookup(self.ua_embeddings_t, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.pos_i_g_embeddings_t = tf.nn.embedding_lookup(self.ia_embeddings_t, self.pos_items)


        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
        self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        self.cor_u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.cor_users)
        self.cor_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.cor_items)


        #Inference for the testing phase.
        self.batch_ratings = tf.matmul(self.u_g_embeddings_t, self.pos_i_g_embeddings_t, transpose_a=False, transpose_b=True)
    
        #Generate Predictions & Optimize via BPR loss.
        self.mf_loss, self.emb_loss = self.create_bpr_loss(self.u_g_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings)

        # whether user distance correlation
        if args.corDecay < 1e-9:
            self.cor_loss = tf.constant(0.0)
        else:
            self.cor_loss = args.corDecay * self.create_cor_loss(self.cor_u_g_embeddings, self.cor_i_g_embeddings)                   

        # self.loss = self.mf_loss + self.emb_loss + self.reg_loss
        self.loss = self.mf_loss + self.emb_loss + self.cor_loss
        self.opt = tfv1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding')
            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embed'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        return all_weights

    def _create_star_routing_embed_with_P(self, pick_ = False):
        '''
        pick_ : True, the model would narrow the weight of the least important factor down to 1/args.pick_scale.
        pick_ : False, do nothing.
        '''
        p_test = False
        p_train = False

        A_values = tf.ones(shape=[self.n_factors, len(self.all_h_list)])
        # get a (n_factors)-length list of [n_users+n_items, n_users+n_items]

        # load the initial all-one adjacency values
        # .... A_values is a all-ones dense tensor with the size of [n_factors, all_h_list].
        

        # get the ID embeddings of users and items
        # .... ego_embeddings is a dense tensor with the size of [n_users+n_items, embed_size];
        # .... all_embeddings stores a (n_layers)-len list of outputs derived from different layers.
        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)
        all_embeddings = [ego_embeddings]
        all_embeddings_t = [ego_embeddings]

        output_factors_distribution = []
        
        factor_num = [self.n_factors, self.n_factors, self.n_factors]
        iter_num = [self.n_iterations, self.n_iterations, self.n_iterations]
        for k in range(0, self.n_layers):
            # prepare the output embedding list
            # .... layer_embeddings stores a (n_factors)-len list of outputs derived from the last routing iterations.
            n_factors_l = factor_num[k]
            n_iterations_l = iter_num[k]
            layer_embeddings = []
            layer_embeddings_t = []
            
            # split the input embedding table
            # .... ego_layer_embeddings is a (n_factors)-leng list of embeddings [n_users+n_items, embed_size/n_factors]
            ego_layer_embeddings = tf.split(ego_embeddings, n_factors_l, 1)
            ego_layer_embeddings_t = tf.split(ego_embeddings, n_factors_l, 1) 

            # perform routing mechanism
            for t in range(0, n_iterations_l):
                iter_embeddings = []
                iter_embeddings_t = []
                A_iter_values = []

                # split the adjacency values & get three lists of [n_users+n_items, n_users+n_items] sparse tensors
                # .... A_factors is a (n_factors)-len list, each of which is an adjacency matrix
                # .... D_col_factors is a (n_factors)-len list, each of which is a degree matrix w.r.t. columns
                # .... D_row_factors is a (n_factors)-len list, each of which is a degree matrix w.r.t. rows
                if t == n_iterations_l - 1:
                    p_test = pick_
                    p_train = False

                A_factors, D_col_factors, D_row_factors = self._convert_A_values_to_A_factors_with_P(n_factors_l, A_values, pick= p_train)
                A_factors_t, D_col_factors_t, D_row_factors_t = self._convert_A_values_to_A_factors_with_P(n_factors_l, A_values, pick= p_test)
                for i in range(0, n_factors_l):
                    # update the embeddings via simplified graph convolution layer
                    # .... D_col_factors[i] * A_factors[i] * D_col_factors[i] is Laplacian matrix w.r.t. the i-th factor
                    # .... factor_embeddings is a dense tensor with the size of [n_users+n_items, embed_size/n_factors]
                    factor_embeddings = tf.sparse.sparse_dense_matmul(D_col_factors[i], ego_layer_embeddings[i])
                    factor_embeddings_t = tf.sparse.sparse_dense_matmul(D_col_factors_t[i], ego_layer_embeddings_t[i])

                    factor_embeddings_t = tf.sparse.sparse_dense_matmul(A_factors_t[i], factor_embeddings_t)
                    factor_embeddings = tf.sparse.sparse_dense_matmul(A_factors[i], factor_embeddings)

                    factor_embeddings = tf.sparse.sparse_dense_matmul(D_col_factors[i], factor_embeddings)
                    factor_embeddings_t = tf.sparse.sparse_dense_matmul(D_col_factors_t[i], factor_embeddings_t)

                    iter_embeddings.append(factor_embeddings)
                    iter_embeddings_t.append(factor_embeddings_t)
                    
                    if t == n_iterations_l - 1:
                        layer_embeddings = iter_embeddings
                        layer_embeddings_t = iter_embeddings_t

                    # get the factor-wise embeddings
                    # .... head_factor_embeddings is a dense tensor with the size of [all_h_list, embed_size/n_factors]
                    # .... analogous to tail_factor_embeddings
                    head_factor_embedings = tf.nn.embedding_lookup(factor_embeddings, self.all_h_list)
                    tail_factor_embedings = tf.nn.embedding_lookup(ego_layer_embeddings[i], self.all_t_list)

                    # .... constrain the vector length
                    # .... make the following attentive weights within the range of (0,1)
                    head_factor_embedings = tf.math.l2_normalize(head_factor_embedings, axis=1)
                    tail_factor_embedings = tf.math.l2_normalize(tail_factor_embedings, axis=1)

                    # get the attentive weights
                    # .... A_factor_values is a dense tensor with the size of [all_h_list,1]
                    A_factor_values = tf.reduce_sum(tf.multiply(head_factor_embedings, tf.tanh(tail_factor_embedings)), axis=1)

                    # update the attentive weights
                    A_iter_values.append(A_factor_values)

                # pack (n_factors) adjacency values into one [n_factors, all_h_list] tensor
                A_iter_values = tf.stack(A_iter_values, 0)
                # add all layer-wise attentive weights up.
                A_values += A_iter_values
                
                if t == n_iterations_l - 1:
                    #layer_embeddings = iter_embeddings
                    output_factors_distribution.append(A_factors)

            # sum messages of neighbors, [n_users+n_items, embed_size]
            side_embeddings = tf.concat(layer_embeddings, 1)
            side_embeddings_t = tf.concat(layer_embeddings_t, 1)
            
            ego_embeddings = side_embeddings
            ego_embeddings_t = side_embeddings_t
            # concatenate outputs of all layers
            all_embeddings_t += [ego_embeddings_t]
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)

        all_embeddings_t = tf.stack(all_embeddings_t, 1)
        all_embeddings_t = tf.reduce_mean(all_embeddings_t, axis=1, keep_dims=False)

        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        u_g_embeddings_t, i_g_embeddings_t = tf.split(all_embeddings_t, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings, output_factors_distribution, u_g_embeddings_t, i_g_embeddings_t

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(self.u_g_embeddings_pre) + tf.nn.l2_loss(
            self.pos_i_g_embeddings_pre) + tf.nn.l2_loss(self.neg_i_g_embeddings_pre)
        regularizer = regularizer / self.batch_size

        # In the first version, we implement the bpr loss via the following codes:
        # We report the performance in our paper using this implementation.
        #         maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        #         mf_loss = tf.negative(tf.reduce_mean(maxi))

        ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
        ## However, it will change the training performance and training performance.
        ## Please retrain the model and do a grid search for the best experimental setting.

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        emb_loss = self.decay * regularizer

        return mf_loss, emb_loss

    def create_cor_loss(self, cor_u_embeddings, cor_i_embeddings):
        cor_loss = tf.constant(0.0, tf.float32)

        if self.cor_flag == 0:
            return  cor_loss

        ui_embeddings = tf.concat([cor_u_embeddings, cor_i_embeddings], axis=0)
        ui_factor_embeddings = tf.split(ui_embeddings, self.n_factors, 1)

        for i in range(0, self.n_factors-1):
            x = ui_factor_embeddings[i]
            y = ui_factor_embeddings[i+1]
            cor_loss += self._create_distance_correlation(x, y)

        cor_loss /= ((self.n_factors + 1.0) * self.n_factors/2)

        return cor_loss

    def model_save(self, path, dataset, ses, savename='best_model'):
        save_pretrain_path = '%spretrain/%s/%s' % (path, dataset, savename)        
        np.savez(save_pretrain_path,user_embed=np.array(self.weights['user_embedding'].eval(session=ses)),
                                    item_embed=np.array(model.weights['item_embedding'].eval(session=ses))) 

    def _create_distance_correlation(self, X1, X2):

        def _create_centered_distance(X):
            '''
                Used to calculate the distance matrix of N samples.
                (However how could tf store a HUGE matrix with the shape like 70000*70000*4 Bytes????)
            '''
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            # X = tf.math.l2_normalize(XX, axis=1)

            r = tf.reduce_sum(tf.square(X), 1, keepdims=True)
            D = tf.sqrt(tf.maximum(r - 2 * tf.matmul(a=X, b=X, transpose_b=True) + tf.transpose(r), 0.0) + 1e-8)
 
            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - tf.reduce_mean(D, axis=0, keepdims=True) - tf.reduce_mean(D, axis=1, keepdims=True) \
                + tf.reduce_mean(D)
            return D

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = tf.dtypes.cast(tf.shape(D1)[0], tf.float32)
            dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2) / (n_samples * n_samples), 0.0) + 1e-8)
            # dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2)) / n_samples
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)

        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        dcor = dcov_12 / (tf.sqrt(tf.maximum(dcov_11 * dcov_22, 0.0)) + 1e-10)
        # return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor

    def _convert_A_values_to_A_factors_with_P(self, f_num, A_factor_values, pick=True):

        A_factors = []
        D_col_factors = []
        D_row_factors = []
        # get the indices of adjacency matrix.
        A_indices = np.mat([self.all_h_list, self.all_t_list]).transpose()
        D_indices = np.mat([list(range(self.n_users+self.n_items)), list(range(self.n_users+self.n_items))]).transpose()

        # apply factor-aware softmax function over the values of adjacency matrix
        # .... A_factor_values is [n_factors, all_h_list]
        if pick:
            A_factor_scores = tf.nn.softmax(A_factor_values, 0)
            min_A = tf.reduce_min(A_factor_scores, 0)
            index = A_factor_scores > (min_A + 0.0000001)
            index = tf.cast(index, tf.float32)*(self.pick_level-1.0) + 1.0  # adjust the weight of the minimum factor to 1/self.pick_level

            A_factor_scores = A_factor_scores * index
            A_factor_scores = A_factor_scores / tf.reduce_sum(A_factor_scores, 0)
        else:
            A_factor_scores = tf.nn.softmax(A_factor_values, 0)

        for i in range(0, f_num):
            # in the i-th factor, couple the adjacency values with the adjacency indices
            # .... A_i_tensor is a sparse tensor with size of [n_users+n_items, n_users+n_items]
            A_i_scores = A_factor_scores[i]
            A_i_tensor = tf.SparseTensor(A_indices, A_i_scores, self.A_in_shape)

            # get the degree values of A_i_tensor
            # .... D_i_scores_col is [n_users+n_items, 1]
            # .... D_i_scores_row is [1, n_users+n_items]
            D_i_col_scores = 1/tf.math.sqrt(tf.sparse_reduce_sum(A_i_tensor, axis=1))
            D_i_row_scores = 1/tf.math.sqrt(tf.sparse_reduce_sum(A_i_tensor, axis=0))

            # couple the laplacian values with the adjacency indices
            # .... A_i_tensor is a sparse tensor with size of [n_users+n_items, n_users+n_items]
            D_i_col_tensor = tf.SparseTensor(D_indices, D_i_col_scores, self.A_in_shape)
            D_i_row_tensor = tf.SparseTensor(D_indices, D_i_row_scores, self.A_in_shape)

            A_factors.append(A_i_tensor)
            D_col_factors.append(D_i_col_tensor)
            D_row_factors.append(D_i_row_tensor)

        # return a (n_factors)-length list of laplacian matrix
        return A_factors, D_col_factors, D_row_factors


def load_best(name="best_model"):
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, name)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the best model:', name)
    except Exception:
        pretrain_data = None
    return pretrain_data

def load_adjacency_list_data(adj_mat):
    tmp = adj_mat.tocoo()
    all_h_list = list(tmp.row)
    all_t_list = list(tmp.col)
    all_v_list = list(tmp.data)

    return all_h_list, all_t_list, all_v_list

def create_initial_A_values(n_factors, all_v_list):
    return np.array([all_v_list] * n_factors)

def sample_cor_samples(n_users, n_items, cor_batch_size):
    '''
        We have to sample some embedded representations out of all nodes.
        Becasue we have no way to store cor-distance for each pair.
    '''
    cor_users = rd.sample(list(range(n_users)), cor_batch_size)
    cor_items = rd.sample(list(range(n_items)), cor_batch_size)

    return cor_users, cor_items

if __name__ == '__main__':
    whether_test_batch = True 
    
    print("************************* Run with following settings 🏃 ***************************")
    print(args)
    print("************************************************************************************")

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    plain_adj, norm_adj, mean_adj, pre_adj = data_generator.get_adj_mat()

    all_h_list, all_t_list, all_v_list = load_adjacency_list_data(plain_adj)

    A_values_init = create_initial_A_values(args.n_factors, all_v_list)

    config['norm_adj'] = plain_adj
    config['all_h_list'] = all_h_list
    config['all_t_list'] = all_t_list


    t0 = time()
    """
    *********************************************************
    pretrain = 1: load embeddings with name such as embedding_xxx(.npz), l2_best_model(.npz)
    pretrain = 0: default value, no pretrained embeddings.
    """
    if args.pretrain == 1:
        print("Try to load pretain: ", args.embed_name)
        pretrain_data = load_best(name=args.embed_name)
        if pretrain_data == None:
             print("Load pretrained model(%s)fail!!!!!!!!!!!!!!!"%(args.embed_name))
    else:
        pretrain_data = None

    model = GDCF(data_config=config, pretrain_data=pretrain_data)


    tf_config = tfv1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tfv1.Session(config=tf_config)


    sess.run(tfv1.global_variables_initializer())
    cur_best_pre_0 = 0.


    """
    *********************************************************
    Train
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss, cor_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1
        cor_batch_size = int(max(data_generator.n_users/n_batch, data_generator.n_items/n_batch))

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            cor_users, cor_items = sample_cor_samples(data_generator.n_users, data_generator.n_items, cor_batch_size)
            _, batch_loss, batch_mf_loss, batch_emb_loss, batch_cor_loss = sess.run([model.opt, model.loss, 
                                                                                    model.mf_loss, model.emb_loss, 
                                                                                    model.cor_loss],
                                                                                    feed_dict={model.users: users, 
                                                                                            model.pos_items: pos_items,
                                                                                            model.neg_items: neg_items,
                                                                                            model.cor_users: cor_users,
                                                                                            model.cor_items: cor_items})
            loss += batch_loss / n_batch
            mf_loss += batch_mf_loss / n_batch
            emb_loss += batch_emb_loss / n_batch
            cor_loss += batch_cor_loss / n_batch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            print(mf_loss, emb_loss)
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1)  % args.show_step != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (epoch, time() - t1, loss, mf_loss, emb_loss, cor_loss)
                print(perf_str)
            # Skip testing
            continue

        # Begin test at this epoch. 
        loss_test, mf_loss_test, emb_loss_test, cor_loss_test = 0., 0., 0., 0.
        for idx in range(n_batch):
            cor_users, cor_items = sample_cor_samples(data_generator.n_users, data_generator.n_items, cor_batch_size)
            users, pos_items, neg_items = data_generator.sample_test()
            batch_loss_test, batch_mf_loss_test, batch_emb_loss_test, batch_cor_loss_test = sess.run(
                [model.loss, model.mf_loss, model.emb_loss, model.cor_loss],
                feed_dict={model.users: users, 
                        model.pos_items: pos_items,
                        model.neg_items: neg_items,
                        model.A_values: A_values_init,
                        model.cor_users: cor_users,
                        model.cor_items: cor_items})
            loss_test += batch_loss_test / n_batch
            mf_loss_test += batch_mf_loss_test / n_batch
            emb_loss_test += batch_emb_loss_test / n_batch
            cor_loss_test += batch_cor_loss_test / n_batch

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True, batch_test_flag=whether_test_batch)


        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: test==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss_test, mf_loss_test, emb_loss_test, cor_loss_test, ret['recall'][0],
                       ret['recall'][-1],
                       ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                       ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)
            
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=args.early)

        # early stopping when cur_best_pre_0 is decreasing for given steps. 
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1 :
            model.model_save(args.proj_path, args.dataset, sess, savename=args.save_name)
            print('save the model with performance: ', cur_best_pre_0)

        
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)


