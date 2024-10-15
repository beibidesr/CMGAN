# -*- coding: utf-8 -*-
# @Time   : 2023/9/15
# @Author : Jianfang liu
# @Email  : jianfangliu@mails.ccnu.edu.cn

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class Aggregator(nn.Module):
    """ GNN Aggregator layer
    """

    def __init__(self, input_dim, output_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if self.aggregator_type == 'gcn':
            self.W = nn.Linear(self.input_dim, self.output_dim)
        elif self.aggregator_type == 'graphsage':
            self.W = nn.Linear(self.input_dim * 2, self.output_dim)
        elif self.aggregator_type == 'bi':
            self.W1 = nn.Linear(self.input_dim, self.output_dim)
            self.W2 = nn.Linear(self.input_dim, self.output_dim)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings, norm_matrix_r, rel_embeddings):
       
        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings) + 0.1 * torch.sparse.mm(norm_matrix_r, rel_embeddings)
        

        if self.aggregator_type == 'gcn':
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == 'graphsage':
            ego_embeddings = self.activation(self.W(torch.cat([ego_embeddings, side_embeddings], dim=1)))
        elif self.aggregator_type == 'bi':
            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)

        return ego_embeddings


class CMGAN(KnowledgeRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(CMGAN, self).__init__(config, dataset)

        # load dataset info
        self.ckg = dataset.ckg_graph(form='dgl', value_field='relation_id')
        self.all_hs = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').row).to(self.device)
        self.all_ts = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').col).to(self.device)
        self.all_rs = torch.LongTensor(dataset.ckg_graph(form='coo', value_field='relation_id').data).to(self.device)
        self.matrix_size = torch.Size([self.n_users + self.n_entities, self.n_users + self.n_entities])

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.kg_embedding_size = config['kg_embedding_size']
        self.layers = [self.embedding_size] + config['layers']
        self.aggregator_type = config['aggregator_type']
        self.mess_dropout = config['mess_dropout']
        self.reg_weight = config['reg_weight']

        # generate intermediate data
        self.A_in = self.init_graph()  # init the attention matrix by the structure of ckg
        indices_r = torch.tensor([[0], [0]])
        kg_score = torch.tensor([0.0])
        self.A_in_r = torch.sparse.FloatTensor(indices_r, kg_score, [self.n_users + self.n_entities, self.n_relations]).cuda()

        # 矩阵的浅拷贝
        self.A_in_1 = self.A_in
        self.A_in_2 = self.A_in

       
        affine = True
        self.projection_head = torch.nn.ModuleList()
        inner_size = self.layers[-1] * 3
        print("inner size:", inner_size)
        self.projection_head.append(torch.nn.Linear(inner_size, inner_size * 4, bias=False))
        self.projection_head.append(torch.nn.BatchNorm1d(inner_size * 4, eps=1e-12, affine=affine))
        self.projection_head.append(torch.nn.Linear(inner_size * 4, inner_size, bias=False))
        self.projection_head.append(torch.nn.BatchNorm1d(inner_size, eps=1e-12, affine=affine))
        self.mode = 0

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(self.n_relations, self.embedding_size * self.kg_embedding_size)
        self.aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.aggregator_layers.append(Aggregator(input_dim, output_dim, self.mess_dropout, self.aggregator_type))
        self.tanh = nn.Tanh()
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.ce_loss = nn.CrossEntropyLoss() 
        self.restore_user_e = None
        self.restore_entity_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_entity_e']

        self.W_a = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.W_b = nn.Linear(self.embedding_size, 1)
        self.act = nn.LeakyReLU()

        self.interest_embedding = torch.nn.parameter.Parameter(torch.nn.init.normal_(torch.zeros(24, 3 * self.embedding_size), mean=0.0, std=0.2), requires_grad=True)

        user_item_id = self.A_in.coalesce().indices()

        max_len = 0
        row2col = {}
        for row, col in zip(user_item_id[0].tolist(), user_item_id[1].tolist()):
            if row >= self.n_users:
                break
            if row not in row2col:
                row2col[row] = [col]
            else:
                row2col[row].append(col)
                if len(row2col[row]) > max_len:
                    max_len = len(row2col[row])

        self.user_item_id = []
        self.user_count = []

        for idx in range(self.n_users):
            if idx in row2col:
                tup = copy.deepcopy(row2col[idx])
                if len(tup) > 0:
                    self.user_count.append(len(tup))
                else:
                    self.user_count.append(1)
                while (len(tup)) < max_len:
                    tup.append(self.A_in.coalesce().size()[1])
                self.user_item_id.append(tup)
            else:
                self.user_item_id.append([self.A_in.coalesce().size()[1]] * max_len)
                self.user_count.append(1)
        self.user_item_id = torch.tensor(self.user_item_id)

        self.margin = 0.01

    def init_graph(self):
        import dgl

        adj_list = []
        for rel_type in range(1, self.n_relations, 1):
            edge_idxs = self.ckg.filter_edges(lambda edge: edge.data['relation_id'] == rel_type)
            sub_graph = dgl.edge_subgraph(self.ckg, edge_idxs, preserve_nodes=True). \
                adjacency_matrix(transpose=False, scipy_fmt='coo').astype('float')
            rowsum = np.array(sub_graph.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.LongTensor([final_adj_matrix.row, final_adj_matrix.col])
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(self.device)

    def _get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)
        return ego_embeddings

    def _get_rel_embeddings(self):
        rel_embeddings = self.relation_embedding.weight
        return rel_embeddings

    def forward_multi_interest(self, norm_matrix, ego_embeddings):
        self.ego_embeddings_x = torch.cat([ego_embeddings, torch.zeros(1, ego_embeddings.shape[1], dtype=torch.float).cuda()], dim=0)
        user_item_embedding = self.ego_embeddings_x[self.user_item_id]

        assign_matrix = torch.einsum("ijk,hk->ihj", user_item_embedding, self.interest_embedding)

        values, indices = torch.max(assign_matrix, dim=1)
        y = torch.zeros_like(assign_matrix)
        y.scatter_(dim=1, index=indices.unsqueeze(1), value=1)
        z = torch.zeros_like(y)
        mask = torch.where(y==1, z, -9999 * torch.ones_like(y))
        result = torch.any(y!=0, dim=2)

        interest_mask = result.float().unsqueeze(-1)

        assign_matrix = torch.softmax(assign_matrix.cuda() + mask.cuda(), dim=2)

        user_interest_embedding = torch.einsum("ijk,ihj->ihk", user_item_embedding, assign_matrix)

        last_hidden_states = user_interest_embedding[torch.arange(user_interest_embedding.size(0)) != 0]

        return last_hidden_states, torch.sum(user_interest_embedding * interest_mask, dim=1) / torch.sum(interest_mask, dim=1), user_interest_embedding, interest_mask #动态多兴趣


    def forward_multi_interest_1(self, norm_matrix, ego_embeddings, user_embedding):
        self.ego_embeddings_x = torch.cat(
            [ego_embeddings, torch.zeros(1, ego_embeddings.shape[1], dtype=torch.float).cuda()], dim=0)
        user_item_embedding = self.ego_embeddings_x[self.user_item_id]

        assign_matrix = torch.einsum("ijk,hk->ihj", user_item_embedding, self.interest_embedding)

        values, indices = torch.max(assign_matrix, dim=1)
        y = torch.zeros_like(assign_matrix)
        y.scatter_(dim=1, index=indices.unsqueeze(1), value=1)
        z = torch.zeros_like(y)
        mask = torch.where(y == 1, z, -9999 * torch.ones_like(y))
        result = torch.any(y != 0, dim=2)

        interest_mask = result.float().unsqueeze(-1)

        assign_matrix = torch.softmax(assign_matrix.cuda() + mask.cuda(), dim=2)

        user_interest_embedding = torch.einsum("ijk,ihj->ihk", user_item_embedding, assign_matrix)

        last_hidden_states = user_interest_embedding[torch.arange(user_interest_embedding.size(0)) != 0]

        return last_hidden_states, 0.9 * user_embedding + 0.1 * torch.sum(user_interest_embedding, dim=1)/ user_interest_embedding.shape[1], user_interest_embedding, interest_mask

    def forward(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        rel_embedding = self._get_rel_embeddings()

        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in, ego_embeddings, self.A_in_r, rel_embedding)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])

        last_hidden_states, multi_interest, user_interest, interest_mask = self.forward_multi_interest_1(self.A_in, kgat_all_embeddings, user_all_embeddings)

        return last_hidden_states, multi_interest, user_interest, interest_mask, entity_all_embeddings

    def forward_1(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        rel_embedding = self._get_rel_embeddings()

        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_1, ego_embeddings, self.A_in_r, rel_embedding)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])
        return user_all_embeddings, entity_all_embeddings

    def forward_2(self):
        ego_embeddings = self._get_ego_embeddings()
        embeddings_list = [ego_embeddings]
        rel_embeddings = self._get_rel_embeddings()

        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in_2, ego_embeddings, self.A_in_r, rel_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(kgat_all_embeddings, [self.n_users, self.n_entities])
        return user_all_embeddings, entity_all_embeddings

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def _get_kg_embedding(self, h, r, pos_t, neg_t):
        h_e = self.entity_embedding(h).unsqueeze(1)
        pos_t_e = self.entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = self.entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(r.size(0), self.embedding_size, self.kg_embedding_size)

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        return h_e, r_e, pos_t_e, neg_t_e

    def cts_loss(self, z_i, z_j, temp, batch_size):

        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        mask = self.mask_correlated_samples(batch_size)

        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.ce_loss(logits, labels)
        return loss

    def projection_head_map(self, state, mode):
        for i, l in enumerate(self.projection_head):
            if i % 2 != 0:
                if mode == 0:
                    l.train()
                else:
                    l.eval()
            state = l(state)
            if i % 2 != 0:
                state = F.relu(state)
        return state

    def calculate_loss(self, interaction):
        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training rs
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        last_hidden_states, user_all_embeddings, user_interest_embedding, interest_mask, entity_all_embeddings = self.forward()

        kgat_all_embeddings = torch.cat((user_all_embeddings, entity_all_embeddings), 0)

        user_all_embeddings_1, entity_all_embeddings_1 = self.forward_1()
        user_all_embeddings_2, entity_all_embeddings_2 = self.forward_2()

        user_rand_samples = self.rand_sample(user_all_embeddings_1.shape[0], size=user.shape[0] // 16, replace=False)
        entity_rand_samples = self.rand_sample(entity_all_embeddings_1.shape[0], size=user.shape[0], replace=False)

        cts_embedding_1 = user_all_embeddings_1[torch.tensor(user_rand_samples).type(torch.long)]
        cts_embedding_2 = user_all_embeddings_2[torch.tensor(user_rand_samples).type(torch.long)]

        e_cts_embedding_1 = entity_all_embeddings_1[torch.tensor(entity_rand_samples).type(torch.long)]
        e_cts_embedding_2 = entity_all_embeddings_2[torch.tensor(entity_rand_samples).type(torch.long)]

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        cts_embedding_1 = self.projection_head_map(cts_embedding_1, self.mode)
        cts_embedding_2 = self.projection_head_map(cts_embedding_2, 1 - self.mode)
        e_cts_embedding_1 = self.projection_head_map(e_cts_embedding_1, self.mode)
        e_cts_embedding_2 = self.projection_head_map(e_cts_embedding_2, 1 - self.mode)

        u_embeddings = self.projection_head_map(u_embeddings, self.mode)
        pos_embeddings = self.projection_head_map(pos_embeddings, 1 - self.mode)

        self.mode = 1 - self.mode

        cts_loss = self.cts_loss(cts_embedding_1, cts_embedding_2, temp=1.0, batch_size=cts_embedding_1.shape[0])

        e_cts_loss = self.cts_loss(e_cts_embedding_1, e_cts_embedding_2, temp=1.0, batch_size=e_cts_embedding_1.shape[0])

        ui_cts_loss = self.cts_loss(u_embeddings, pos_embeddings, temp=1.0, batch_size=u_embeddings.shape[0])

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = entity_all_embeddings[pos_item]
        neg_embeddings = entity_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, neg_scores)
        reg_loss = self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)

        intest_emb = self.interest_embedding
        all_item_embeddings = entity_all_embeddings[:self.n_items]
        l_int = self.reg_loss(all_item_embeddings.detach() - torch.matmul(all_item_embeddings.detach(), torch.matmul(intest_emb.T, intest_emb)))  #Octopus公式4
        l_reg = self.reg_loss(torch.matmul(intest_emb, intest_emb.T) - torch.eye(intest_emb.shape[0]).cuda())

        user_interest_embedding = user_interest_embedding[torch.tensor(user_rand_samples).type(torch.long)]
        interest_mask = interest_mask[torch.tensor(user_rand_samples).type(torch.long)]
        interest_mask_1 = interest_mask.squeeze_(2)

        input_ids = torch.ones(torch.tensor(user_rand_samples).type(torch.long).shape[0], 16)

        norm_rep = user_interest_embedding / (user_interest_embedding.norm(dim=2, keepdim=True) + 0.1)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1, 2))

        cl_loss = self.contrastive_loss(cosine_scores, interest_mask_1)

        loss = mf_loss + self.reg_weight * reg_loss + 0.01 * ui_cts_loss + 0.0001 * (l_int + l_reg) + 0.001 * cl_loss

        return loss

    def build_mask_matrix(self, seqlen, valid_len_list):
        res_list = []
        base_mask = torch.ones(seqlen, seqlen) - torch.eye(seqlen, seqlen)
        base_mask = base_mask.type(torch.FloatTensor)
        bsz = len(valid_len_list)
        for i in range(bsz):
            one_base_mask = base_mask.clone()
            one_valid_len = valid_len_list[i]
            one_base_mask[:, one_valid_len:] = 0.
            one_base_mask[one_valid_len:, :] = 0.
            res_list.append(one_base_mask)
        res_mask = torch.stack(res_list, dim=0)

        assert res_mask.size() == torch.Size([bsz, seqlen, seqlen])
        return res_mask

    def contrastive_loss(self, score_matrix, input_ids):
        bsz, seqlen, _ = score_matrix.size()

        gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2)
        gold_score = torch.unsqueeze(gold_score, -1)
        assert gold_score.size() == torch.Size([bsz, seqlen, 1])
        difference_matrix = gold_score - score_matrix
        assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
        loss_matrix = self.margin - difference_matrix
        loss_matrix = torch.nn.functional.relu(loss_matrix)

        input_mask = torch.ones_like(input_ids).type(torch.FloatTensor)

        if loss_matrix.is_cuda:
            input_mask = input_mask.cuda(loss_matrix.get_device())
        input_mask = input_mask.masked_fill(input_ids.eq(0.0).cuda(), 0.0)

        if loss_matrix.is_cuda:
            input_mask = input_mask.cuda(loss_matrix.get_device())

        valid_len_list = torch.sum(input_mask, dim=-1).tolist()
        loss_mask = self.build_mask_matrix(seqlen, [int(item) for item in valid_len_list])
        if score_matrix.is_cuda:
            loss_mask = loss_mask.cuda(score_matrix.get_device())

        masked_loss_matrix = loss_matrix * loss_mask

        loss_matrix = torch.sum(masked_loss_matrix, dim=-1)
        assert loss_matrix.size() == input_ids.size()

        loss_matrix = loss_matrix * input_mask

        cl_loss = torch.sum(loss_matrix) / torch.sum(loss_mask)

        return cl_loss

    def calculate_kg_loss(self, interaction):

        if self.restore_user_e is not None or self.restore_entity_e is not None:
            self.restore_user_e, self.restore_entity_e = None, None

        # get loss for training kg
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t)
        pos_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        kg_loss = F.softplus(pos_tail_score - neg_tail_score).mean()
        kg_reg_loss = self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)
        loss = kg_loss + self.reg_weight * kg_reg_loss

        return loss

    def generate_transE_score(self, hs, ts, r):
        all_embeddings = self._get_ego_embeddings()
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding.weight[r]
        r_trans_w = self.trans_w.weight[r].view(self.embedding_size, self.kg_embedding_size)

        h_e = torch.matmul(h_e, r_trans_w)
        t_e = torch.matmul(t_e, r_trans_w)

        kg_score = self.W_b(self.act(self.W_a(torch.cat([t_e, h_e+r_e], dim=1)))).squeeze()
        return kg_score

    def rand_sample(self, high, size=None, replace=False):

        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample

    def update_attentive_A(self):
        kg_score_list, row_list, rel_list, col_list = [], [], [], []
        for rel_idx in range(1, self.n_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            kg_score = self.generate_transE_score(self.all_hs[triple_index], self.all_ts[triple_index], rel_idx).view(-1)
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])
            rel_list.append(self.all_rs[triple_index])
            kg_score_list.append(kg_score)
        kg_score = torch.cat(kg_score_list, dim=0)

        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        rel = torch.cat(rel_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1)
        indices_r = torch.cat([row, rel], dim=0).view(2, -1)

        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)
        self.A_in = A_in

        A_in_r = torch.sparse.FloatTensor(indices_r, kg_score, [A_in.shape[0], self.relation_embedding.weight.shape[0]]).cpu()
        A_in_r = torch.sparse.softmax(A_in_r, dim=1).to(self.device)
        self.A_in_r = A_in_r

        drop_edge_1 = self.rand_sample(indices.shape[1], size=int(indices.shape[1] * 0.1), replace=False)
        indices_1 = indices.view(-1, 2)[torch.tensor(drop_edge_1).type(torch.long)].view(2, -1)
        kg_score_1 = kg_score[torch.tensor(drop_edge_1).type(torch.long)]
        A_in_1 = torch.sparse.FloatTensor(indices_1, kg_score_1, self.matrix_size).cpu()
        A_in_1 = torch.sparse.softmax(A_in_1, dim=1).to(self.device)

        drop_edge_2 = self.rand_sample(indices.shape[1], size=int(indices.shape[1] * 0.1), replace=False)
        indices_2 = indices.view(-1, 2)[torch.tensor(drop_edge_2).type(torch.long)].view(2, -1)
        kg_score_2 = kg_score[torch.tensor(drop_edge_2).type(torch.long)]
        A_in_2 = torch.sparse.FloatTensor(indices_2, kg_score_2, self.matrix_size).cpu()
        A_in_2 = torch.sparse.softmax(A_in_2, dim=1).to(self.device)

        self.A_in_1 = A_in_1
        self.A_in_2 = A_in_2

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_interest_embedding, user_all_embeddings, user_interest, interest_mask, entity_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = entity_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_entity_e is None:
           user_interest_embedding, self.restore_user_e, user_interest, interest_mask, self.restore_entity_e = self.forward()
        u_embeddings = self.restore_user_e[user]
        i_embeddings = self.restore_entity_e[:self.n_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))

        return scores.view(-1)
