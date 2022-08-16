import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
import os
from torch import optim
from . import network
import torch.nn.init as init
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

from collections import OrderedDict

class TypeGraphConvolution(nn.Module):
    """
    Type GCN layer
    """
    def __init__(self, in_features, out_features, bias=True):
        super(TypeGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, text, adj, dep_embed):
        batch_size, max_len, feat_dim = text.shape
        val_us = text.unsqueeze(dim=2) #B*M*1*F
        val_us = val_us.repeat(1, 1, max_len, 1)     #B*M*M*F
        val_sum = val_us + dep_embed    #B*M*M*F
        adj_us = adj.unsqueeze(dim=-1)   #B*M*M*1
        adj_us = adj_us.repeat(1, 1, 1, feat_dim) #B*M*M*F
        hidden = torch.matmul(val_sum.float(), self.weight.float())  #B*M*M*F
        output = hidden.transpose(1,2) * adj_us.float()   #B*M*M*F
        output = torch.sum(output, dim=2)   #B*M*F

        if self.bias is not None:
            output = output + self.bias

        return F.relu(output.type_as(text))

class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False, backend_model=None):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        # self.tanh=nn.Sigmoid()
        # self.dropout=torch.nn.Dropout(0.05)
        self.hidden_dim=768
        self.mode="pos_token" #"start" #"max_pooling" pos_token
        self.bert_fc=torch.nn.Linear(768,self.hidden_dim)
        self.dep_type=["ROOT", "det", "nsubj", "mark", "acl", "advmod", "nmod:poss", "amod", "dobj", "case", "nmod", "compound", "punct", "nsubjpass", "auxpass", "cc", "conj", "advcl", "cop", "acl:relcl", "ccomp", "aux", "csubjpass", "nummod", "dep", "xcomp", "appos", "nmod:npmod", "compound:prt", "root", "nmod:tmod", "neg", "mwe", "parataxis", "det:predet", "expl", "iobj", "cc:preconj", "csubj", "discourse"]
        # self.dep_type_embedding = nn.Embedding((len(self.dep_type)+2), self.hidden_dim, padding_idx=0)
        self.dep_type_embedding = nn.Embedding((len(self.dep_type)+1)*2, self.hidden_dim, padding_idx=0)
        gcn_layer = TypeGraphConvolution(self.hidden_dim, self.hidden_dim)
        self.gcn_layer = nn.ModuleList([copy.deepcopy(gcn_layer) for _ in range(1)])

        ####TODO
        if backend_model == 'cp':
            ckpt = torch.load("./CP_model/CP")
            #import pdb
            #pdb.set_trace()
            temp = OrderedDict()
            ori_dict = self.bert.state_dict()
            for name, parameter in ckpt["bert-base"].items():
                if name in ori_dict:
                    temp[name] = parameter

            ori_dict.update(temp)
            self.bert.load_state_dict(ori_dict)

        #self.bert.load_state_dict(ckpt["bert-base"])

        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        ##
        '''
        self.linear_t = nn.Linear(768*2, 768)
        self.linear_h = nn.Linear(768*2, 768)
        '''

    def global_atten2(self, h_state, t_state, sequence_outputs):
        #the best model now, 2021/10/22, 86.12%
        t_temp0 = t_state.view(t_state.shape[0], 1, -1)
        #import pdb
        #pdb.set_trace()
        t_temp = torch.softmax(torch.matmul(sequence_outputs, t_temp0.permute(0,2,1)), 1)#.squeeze() ##[20, 128, 1]
        t_temp = t_temp.expand(sequence_outputs.shape[0], sequence_outputs.shape[1], sequence_outputs.shape[2])
        t_global_feature = torch.mean(t_temp * sequence_outputs, 1)
        t_state = torch.cat((t_state, t_global_feature), -1)

        h_temp0 = h_state.view(h_state.shape[0], 1, -1)
        #import pdb
        #pdb.set_trace()
        h_temp = torch.softmax(torch.matmul(sequence_outputs, h_temp0.permute(0,2,1)), 1)#.squeeze() ##[20, 128, 1]
        h_temp = h_temp.expand(sequence_outputs.shape[0], sequence_outputs.shape[1], sequence_outputs.shape[2])
        h_global_feature = torch.mean(h_temp * sequence_outputs, 1)
        h_state = torch.cat((h_state, h_global_feature), -1)
        return h_state, t_state

    def entity_atten(self, h_state, t_state, sequence_outputs, inputs):
        batch, dim = h_state.shape

        h_final = torch.zeros([batch, dim], dtype=torch.float32, device='cuda')
        t_final = torch.zeros([batch, dim], dtype=torch.float32, device='cuda')

        for idx in range(len(inputs['pos1'])):

            head_entity = sequence_outputs[idx, inputs['pos1'][idx]: inputs['pos1_end'][idx] + 1]
            tail_entity = sequence_outputs[idx, inputs['pos2'][idx]: inputs['pos2_end'][idx] + 1]
            n, m = head_entity.shape
            n2, m2 = tail_entity.shape
            #import pdb
            #pdb.set_trace()

            temp_h = torch.softmax(torch.matmul(head_entity, h_state[idx].view(-1, 1)), 0).expand(n, head_entity.shape[1])
            h_final[idx] = torch.mean(temp_h * head_entity, 0)

            temp_t = torch.softmax(torch.matmul(tail_entity, t_state[idx].view(-1, 1)), 0).expand(n2, tail_entity.shape[1])
            t_final[idx] = torch.mean(temp_t * tail_entity, 0)
        #import pdb
        #pdb.set_trace()
        return h_final, t_final

    def valid_filter(self, sequence_output, valid_ids,e1_mask,e2_mask):
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype,
                                   device=sequence_output.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            valid_output[i][:temp.size(0)] = temp


        len_e1_word=[0]*batch_size
        len_e2_word=[0]*batch_size
        len_before_e1_word=[0]*batch_size
        len_before_e2_word=[0]*batch_size
        for i in range(len(e1_mask)):
            e1_meet=False
            e2_meet=False
            for j in range(len(e1_mask[0])):
                if e1_meet==False and valid_ids[i][j]==1:
                    len_before_e1_word[i]+=1
                if e2_meet==False and valid_ids[i][j]==1:
                    len_before_e2_word[i]+=1
                if e1_mask[i][j]==1 and valid_ids[i][j]==0:
                    e1_mask[i][j]=0
                if e2_mask[i][j]==1 and valid_ids[i][j]==0:
                    e2_mask[i][j]=0
                if e1_mask[i][j]==1:
                    len_e1_word[i]+=1
                    e1_meet=True
                if e2_mask[i][j]==1:
                    len_e2_word[i]+=1
                    e2_meet=True
        e1_mask_valid= torch.zeros(batch_size, max_len, dtype=sequence_output.dtype,
                                                         device=sequence_output.device)
        e2_mask_valid= torch.zeros(batch_size, max_len, dtype=sequence_output.dtype,
                                   device=sequence_output.device)
        for i in range(batch_size):
            for j in range(len_before_e1_word[i]-1,len_before_e1_word[i]+len_e1_word[i]-1):
                e1_mask_valid[i][j]=1
            for j in range(len_before_e2_word[i]-1,len_before_e2_word[i]+len_e2_word[i]-1):
                e2_mask_valid[i][j]=1

        return valid_output,e1_mask_valid,e2_mask_valid

    def max_pooling(self, sequence, e_mask):
        entity_output = sequence * torch.stack([e_mask] * sequence.shape[-1], 2) + torch.stack(
            [(1.0 - e_mask) * -1000.0] * sequence.shape[-1], 2)
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(sequence)

    def extract_entity(self, sequence, e_mask,mode):
        if mode=="max_pooling":
            return self.max_pooling(sequence, e_mask)

        if mode=="start":
            entity_output = torch.zeros(sequence.shape[0], sequence.shape[2], dtype=sequence.dtype,
                                       device=sequence.device)
            for i in range(sequence.shape[0]):
                for j in range(sequence.shape[1]):
                    if e_mask[i][j]==1:
                        entity_output[i]=sequence[i][j]
                        break
            return entity_output
        if mode=="mean":
            entity_output = torch.zeros(sequence.shape[0], sequence.shape[2], dtype=sequence.dtype,
                                        device=sequence.device)
            # num=0
            for i in range(sequence.shape[0]):
                num=0
                for j in range(sequence.shape[1]):
                    if e_mask[i][j]==1:
                        entity_output[i]=entity_output[i]+sequence[i][j]
                        num+=1
                if num!=0:
                    entity_output[i]=entity_output[i]/torch.tensor(num).cuda()
                else:
                    entity_output[i]=sequence[i][0]
            return entity_output

        if mode=="pos_token":
            entity_output = torch.zeros(sequence.shape[0], sequence.shape[2], dtype=sequence.dtype,
                                        device=sequence.device)
            for i in range(sequence.shape[0]):
                for j in range(sequence.shape[1]):
                    if e_mask[i][j]==1:
                        entity_output[i]=sequence[i][j-1]
                        break
            return entity_output


    def get_attention(self, val_out, dep_embed, adj):
        batch_size, max_len, feat_dim = val_out.shape #B*M*F
        val_us = val_out.unsqueeze(dim=2) #B*M*1*F
        val_us = val_us.repeat(1,1,max_len,1) #B*M*M*F
        val_cat = torch.cat((val_us, dep_embed), -1) #B*M*M*(2*F)
        atten_expand = (val_cat.float() * val_cat.float().transpose(1,2)) #矩阵对应元素相乘 B*M*M*(2*F)
        attention_score = torch.sum(atten_expand, dim=-1) #B*M*M
        attention_score = attention_score / feat_dim ** 0.5 #B*M*M
        # softmax
        exp_attention_score = torch.exp(attention_score)    #B*M*M
        exp_attention_score = torch.mul(exp_attention_score.float(), adj.float())  #B*M*M
        sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1,1,max_len) #B*M*M
        attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)   #B*M*M
        return attention_score

    def forward(self, inputs, cat=True):
        if not self.cat_entity_rep:
            #import pdb
            #pdb.set_trace()
            #import pdb
            #pdb.set_trace()
            x = self.bert(inputs['word'], attention_mask=inputs['mask'])['pooler_output']

            #x = self.bert(inputs['word'], attention_mask=inputs['mask'])['last_hidden_state']

            ##insert local feature
            #local_final = self.windows_sequence(x, 5, self.bilstm)
            #x = torch.cat([local_final, x], dim=-1)
            #x = self.linear(x)
            #x = torch.mean(x, 1)

            return x
        else:
            ##this is concanate the start tokens of two entity mentions
            #import pdb
            #pdb.set_trace()
            '''
            e1_e2_mask->parser_text
            valid_id->parser_mask
            '''
            outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
            outputs_fc=outputs['last_hidden_state']
            # outputs_fc=self.bert_fc(outputs_fc)

            outputs_fc_cls=outputs['pooler_output']
            # sequence_output,pooler_output=self.bert(inputs['word'], attention_mask=inputs['mask'])

            if cat:
                # outputs_parser  self.bert(inputs['parser_text'], attention_mask=inputs['parser_mask'])
                # sequence_outputs_raw = outputs_raw['last_hidden_state'] # [20, 128, 768]
                # sequence_outputsP_parser = outputs_parser['last_hidden_state']
                # parser_tensor_range = torch.arange(inputs['parser_text'].size()[0])  # inputs['word'].shape  [20, 128]
                # parser_h_state = outputs_parser['last_hidden_state'][parser_tensor_range, inputs["pos1_end"]] # h_state.shape [20, 768]
                # parser_t_state = outputs_parser['last_hidden_state'][parser_tensor_range, inputs["pos2_end"]] # [20, 768]

                e1_mask,e2_mask=inputs['parser_text'][:,0],inputs['parser_text'][:,1]
                valid_ids=inputs['parser_mask']
                sequence_output,e1_mask_valid,e2_mask_valid = self.valid_filter(outputs_fc, valid_ids,e1_mask,e2_mask)

                dep_adj_matrix,dep_type_matrix=inputs['matric'][:self.max_length,0],inputs['matric'][:self.max_length,1]
                dep_type_embedding_outputs = self.dep_type_embedding(dep_type_matrix)
                dep_adj_matrix = torch.clamp(dep_adj_matrix, 0, 1)
                for i, gcn_layer_module in enumerate(self.gcn_layer):
                    attention_score = self.get_attention(sequence_output, dep_type_embedding_outputs, dep_adj_matrix)
                    outputs_parser= gcn_layer_module(sequence_output, attention_score, dep_type_embedding_outputs)

                e1_h = self.extract_entity(outputs_fc, e1_mask,self.mode)
                e2_h = self.extract_entity(outputs_fc, e2_mask,self.mode)

                e1_h_parser = self.extract_entity(outputs_parser, e1_mask_valid,'mean')
                e2_h_parser = self.extract_entity(outputs_parser, e2_mask_valid,'mean')
                # #
                h_state,t_state=e1_h+e1_h_parser,e2_h+e2_h_parser
                # h_state,t_state=e1_h_parser,e2_h_parser
                # h_state,t_state=e1_h,e2_h

                # tensor_range = torch.arange(inputs['word'].size()[0])
                # h_state_parser = outputs_parser[tensor_range, inputs["pos1"]] # h_state.shape [20, 768]
                # t_state_parser = outputs_parser[tensor_range, inputs["pos2"]] # [20, 768]
                #
                # h_state=h_state_parser+outputs_fc[tensor_range, inputs["pos1"]]
                # t_state=t_state_parser+outputs_fc[tensor_range, inputs["pos2"]]

                # demo_tensor_range = torch.arange(inputs['word'].size()[0])  # inputs['word'].shape  [20, 128]
                # parser_h_state = outputs_raw['last_hidden_state'][demo_tensor_range, inputs["pos1_end"]] # h_state.shape [20, 768]
                # parser_t_state = outputs_raw['last_hidden_state'][demo_tensor_range, inputs["pos2_end"]] # [20, 768]
                # demo_h_state = outputs_raw['last_hidden_state'][demo_tensor_range, inputs["pos1"]] # h_state.shape [20, 768]
                # demo_t_state = outputs_raw['last_hidden_state'][demo_tensor_range, inputs["pos2"]] # [20, 768]
                #
                # h_state=torch.cat([demo_h_state,parser_h_state],dim=-1)
                # t_state=torch.cat([demo_t_state,parser_t_state],dim=-1)
                #
                #
                # h_state=self.fc(h_state)
                # t_state=self.fc(t_state)
                # h_state=self.tanh(h_state)
                # t_state=self.tanh(t_state)
                # parser_cls = outputs_parser['last_hidden_state'][:, 0] # h_state.shape [20, 768]
                # parser_avg = torch.mean(outputs_parser['last_hidden_state'],1) # [20, 768]

                # h_state=torch.cat([h_state,t_state],-1)
                # t_state=torch.cat([parser_cls,parser_avg],-1)

                # h_state=parser_cls
                # t_state=parser_avg

                # batch_size, max_len, feat_dim = sequence_outputs.shape

                ###TODO delete the element in the middle, no effect
                '''
                mask_matrix = torch.ones([batch_size, max_len, feat_dim], dtype=torch.float32, device='cuda')
                temp = torch.zeros([feat_dim], dtype=torch.float32, device='cuda')
                for i in range(batch_size):
                    mask_matrix[i][inputs["pos1"][i]] = temp
                    mask_matrix[i][inputs["pos2"][i]] = temp
                sequence_outputs = sequence_outputs * mask_matrix
                '''
                #############################################

                #TODO, add two attention for obtaining the relation representation. the entity attention and the global attention.
                #the format of the attention is the matrix multiply with the softmax layer.

                #take this into outside
                '''
                h_state, t_state = self.global_atten2(h_state, t_state, sequence_outputs)
                h_state = self.linear_h(h_state)
                t_state = self.linear_t(t_state)
            
                '''
                #state = torch.cat((h_state, t_state), -1)

                #return state, outputs['last_hidden_state']

            #     return h_state, t_state, sequence_output #,parser_cls,parser_avg
            # else:
            #     return pooler_output, sequence_output

                return h_state, t_state, outputs['last_hidden_state'] #,parser_cls,parser_avg
            else:
                # return outputs['pooler_output'], outputs['last_hidden_state']
                return outputs_fc_cls,outputs_fc
    def tokenize_agcn(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        pos_head=pos_head[2][0]
        pos_tail=pos_tail[2][0]
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos1_end_index = 1

        pos2_in_index = 1
        pos2_end_index = 1

        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
                pos1_end_index = len(tokens)

            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
                pos2_end_index = len(tokens)

            cur_pos += 1
            ## the operation above does like: insert '[unused0]','[unused2]' before and after the head entity; insert '[unused1]', '[unused3]' before and after the tail entity
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        length_tokens=len(indexed_tokens)
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        #import pdb
        #pdb.set_trace()

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        pos1_end_index = min(self.max_length, pos1_end_index)
        pos2_end_index = min(self.max_length, pos2_end_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, length_tokens, pos1_end_index - 1, pos2_end_index - 1,indexed_tokens,mask  #these positions are exactly the position of four special charaters

    def tokenize_agcn_valid(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        pos_head=pos_head[2][0]
        pos_tail=pos_tail[2][0]
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos1_end_index = 1

        pos2_in_index = 1
        pos2_end_index = 1

        e1_mask = [0]*self.max_length
        e2_mask = [0]*self.max_length
        valid = [0]

        for token in raw_tokens:
            token = token.lower()
            # if cur_pos == pos_head[0] or cur_pos == pos_tail[0]:
            #     if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
            #         tokens.append('[unused0]')
            #         pos1_in_index = len(tokens)
            #         valid.append(0)
            #     if cur_pos == pos_tail[0]:
            #         tokens.append('[unused1]')
            #         pos2_in_index = len(tokens)
            #         valid.append(0)
            #
            #     token = self.tokenizer.tokenize(token)
            #     tokens.extend(token)
            #     for m in range(len(token)):
            #         if m == 0:
            #             valid.append(1)
            #         else:
            #             valid.append(0)
            #     cur_pos+=1
            #     continue
            # if cur_pos == pos_head[-1] or cur_pos == pos_tail[-1]:
            #     token = self.tokenizer.tokenize(token)
            #     tokens.extend(token)
            #     for m in range(len(token)):
            #         if m == 0:
            #             valid.append(1)
            #         else:
            #             valid.append(0)
            #     if cur_pos == pos_head[-1]:
            #         tokens.append('[unused2]')
            #         pos1_end_index = len(tokens)
            #         valid.append(0)
            #
            #     if cur_pos == pos_tail[-1]:
            #         tokens.append('[unused3]')
            #         pos2_end_index = len(tokens)
            #         valid.append(0)
            #     cur_pos+=1
            #     continue
            # token = self.tokenizer.tokenize(token)
            # tokens.extend(token)
            # for m in range(len(token)):
            #     if m == 0:
            #         valid.append(1)
            #     else:
            #         valid.append(0)
            # cur_pos += 1
            if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
                valid.append(0)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
                valid.append(0)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                token = self.tokenizer.tokenize(token)
                tokens.extend(token)
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                    else:
                        valid.append(0)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
                pos1_end_index = len(tokens)
                valid.append(0)

            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
                pos2_end_index = len(tokens)
                valid.append(0)

            cur_pos += 1
            ## the operation above does like: insert '[unused0]','[unused2]' before and after the head entity; insert '[unused1]', '[unused3]' before and after the tail entity
        try:
            for i in range(pos1_in_index,pos1_end_index-1):
                e1_mask[i]=1
        except:
            pass
        try:
            for i in range(pos2_in_index,pos2_end_index-1):
                e2_mask[i]=1
        except:
            pass
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        if self.max_length>len(indexed_tokens):
            valid+=[0]*(self.max_length-len(indexed_tokens))
        else:
            valid=valid[:self.max_length]
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]
        length_tokens=len(indexed_tokens)
        #import pdb
        #pdb.set_trace()

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        pos1_end_index = min(self.max_length, pos1_end_index)
        pos2_end_index = min(self.max_length, pos2_end_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, length_tokens, pos1_end_index - 1, pos2_end_index - 1,[e1_mask,e2_mask],valid  #these positions are exactly the position of four special charaters

    def tokenize_parser(self, raw_tokens, head,tail, parser_tokens):
        # token -> index
        tokens = ['[CLS]']
        pos_head=[]
        pos_tail=[]
        pos1_in_index=pos2_in_index=pos1_end_index=pos2_end_index=0
        low_parser_tokens=[token.lower() for token in parser_tokens]
        for i in head[0].split(' '):
            if i.lower() in low_parser_tokens:
                pos_head.append(low_parser_tokens.index(i.lower()))
            else:
                continue
        if pos_head==[]:
            pos_head=head[0].split(' ')

        for i in tail[0].split(' '):
            if i.lower() in low_parser_tokens:
                pos_tail.append(low_parser_tokens.index(i.lower()))
            else:
                continue
        if pos_tail==[]:
            pos_tail=tail[0].split(' ')
        # pos_head=[low_parser_tokens.index(i.lower()) if i.lower() in low_parser_tokens else 0 for i in head[0].split(' ') ]
        # pos_tail=[low_parser_tokens.index(i.lower()) if i.lower() in low_parser_tokens else len(low_parser_tokens)-1 for i in tail[0].split(' ') ]

        cur_pos=0
        for token in low_parser_tokens:
            if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
                pos1_end_index = len(tokens)

            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
                pos2_end_index = len(tokens)

            cur_pos += 1

        # for token in parser_tokens:
        #     if  cur_pos == 1:
        #         tokens.append('[unused0]')
        #         tokens += self.tokenizer.tokenize(token.lower())
        #         pos1_end_index=len(tokens)
        #         tokens.append('[unused2]')
        #     elif cur_pos == len(parser_tokens):
        #         tokens.append('[unused1]')
        #         pos2_in_index=len(tokens)
        #         tokens += self.tokenizer.tokenize(token.lower())
        #         pos2_end_index=len(tokens)
        #         tokens.append('[unused3]')
        #     else:
        #         tokens += self.tokenizer.tokenize(token.lower())
        #     cur_pos+=1
        # tokens.append('[SEP]')
        #
        # cur_pos=0
        # for token in raw_tokens:
        #     token = token.lower()
        #     if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
        #         tokens.append('[unused0]')
        #         # pos1_in_index = len(tokens)
        #     if cur_pos == pos_tail[0]:
        #         tokens.append('[unused1]')
        #         # pos2_in_index = len(tokens)
        #     if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
        #         tokens += ['[unused4]']
        #     else:
        #         tokens += self.tokenizer.tokenize(token)
        #     if cur_pos == pos_head[-1]:
        #         tokens.append('[unused2]')
        #         # pos1_end_index = len(tokens)
        #
        #     if cur_pos == pos_tail[-1]:
        #         tokens.append('[unused3]')
        #         # pos2_end_index = len(tokens)
        #
        #     cur_pos += 1
        #     ## the operation above does like: insert '[unused0]','[unused2]' before and after the head entity; insert '[unused1]', '[unused3]' before and after the tail entity
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        #import pdb
        #pdb.set_trace()

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        pos1_end_index = min(self.max_length, pos1_end_index)
        pos2_end_index = min(self.max_length, pos2_end_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, len(indexed_tokens), pos1_end_index - 1, pos2_end_index - 1 #these positions are exactly the position of four special charaters

    def tokenize_parserAndRaw(self, raw_tokens, head,tail, parser_tokens):

        # token -> index

        def token_parser(parser_tokens):
            tokens = ['[CLS]']
            # print(parser_tokens)
            pos_head=parser_tokens['h_pos']
            pos_tail=parser_tokens['t_pos']
            if len(pos_head)!=1:
                index=0
                try:
                    while(index<=len(pos_head)-2):
                        if pos_head[index]+1==pos_head[index+1]:
                            index+=1
                            break
                        else:
                            del pos_head[index]
                except:
                    print(parser_tokens)
            if len(pos_tail)!=1:
                index=0
                try:
                    while(index<=len(pos_tail)-2):
                        if pos_tail[index]+1==pos_tail[index+1]:
                            index+=1
                            break
                        else:
                            del pos_tail[index]
                except:
                    print(parser_tokens)
            pos1_in_index = 1
            pos1_end_index = 1

            pos2_in_index = 1
            pos2_end_index = 1

            cur_pos=0
            for token in parser_tokens['tokens']:
                token = token.lower()
                if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
                    tokens.append('[unused0]')
                    pos1_in_index = len(tokens)
                if cur_pos == pos_tail[0]:
                    tokens.append('[unused1]')
                    pos2_in_index = len(tokens)
                if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                    tokens += ['[unused4]']
                else:
                    tokens += self.tokenizer.tokenize(token)
                if cur_pos == pos_head[-1]:
                    tokens.append('[unused2]')
                    pos1_end_index = len(tokens)

                if cur_pos == pos_tail[-1]:
                    tokens.append('[unused3]')
                    pos2_end_index = len(tokens)

                cur_pos += 1
            tokens.append('[SEP]')

            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            length_tokens=len(indexed_tokens)
            # padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]


            # pos
            pos1 = np.zeros((self.max_length), dtype=np.int32)
            pos2 = np.zeros((self.max_length), dtype=np.int32)
            for i in range(self.max_length):
                pos1[i] = i - pos1_in_index + self.max_length
                pos2[i] = i - pos2_in_index + self.max_length

            # mask
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(tokens)] = 1

            pos1_in_index = min(self.max_length, pos1_in_index)
            pos2_in_index = min(self.max_length, pos2_in_index)

            pos1_end_index = min(self.max_length, pos1_end_index)
            pos2_end_index = min(self.max_length, pos2_end_index)

            return indexed_tokens,mask,pos1_in_index-1,pos2_in_index-1,length_tokens

        def token_demo(raw_tokens, pos_head,pos_tail):
            # token -> index
            pos_head=pos_head[2][0]
            pos_tail=pos_tail[2][0]
            tokens = ['[CLS]']
            cur_pos = 0
            pos1_in_index = 1
            pos1_end_index = 1

            pos2_in_index = 1
            pos2_end_index = 1

            for token in raw_tokens:
                token = token.lower()
                if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
                    tokens.append('[unused0]')
                    pos1_in_index = len(tokens)
                if cur_pos == pos_tail[0]:
                    tokens.append('[unused1]')
                    pos2_in_index = len(tokens)
                if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                    tokens += ['[unused4]']
                else:
                    tokens += self.tokenizer.tokenize(token)
                if cur_pos == pos_head[-1]:
                    tokens.append('[unused2]')
                    pos1_end_index = len(tokens)

                if cur_pos == pos_tail[-1]:
                    tokens.append('[unused3]')
                    pos2_end_index = len(tokens)

                cur_pos += 1
                ## the operation above does like: insert '[unused0]','[unused2]' before and after the head entity; insert '[unused1]', '[unused3]' before and after the tail entity
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            # padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]

            #import pdb
            #pdb.set_trace()

            # pos
            pos1 = np.zeros((self.max_length), dtype=np.int32)
            pos2 = np.zeros((self.max_length), dtype=np.int32)
            for i in range(self.max_length):
                pos1[i] = i - pos1_in_index + self.max_length
                pos2[i] = i - pos2_in_index + self.max_length

            # mask
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(tokens)] = 1

            pos1_in_index = min(self.max_length, pos1_in_index)
            pos2_in_index = min(self.max_length, pos2_in_index)

            pos1_end_index = min(self.max_length, pos1_end_index)
            pos2_end_index = min(self.max_length, pos2_end_index)

            return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, len(indexed_tokens), pos1_end_index - 1, pos2_end_index - 1 #these positions are exactly the position of four special charaters

        parser_indexed_tokens,parser_mask,parser_pos1_in_index,parser_pos2_in_index,parser_length_tokens=token_parser(parser_tokens)

        indexed_tokens, pos1_in_index, pos2_in_index , mask, length_tokens, pos1_end_index, pos2_end_index=token_demo(raw_tokens,head,tail)
        parser_indexed_tokens_temp=parser_indexed_tokens[:parser_length_tokens]
        parser_indexed_tokens_temp.extend(indexed_tokens)
        indexed_tokens=parser_indexed_tokens_temp[:self.max_length]

        parser_mask_temp=parser_mask[:parser_length_tokens]
        mask=np.concatenate([parser_mask_temp,mask],axis=-1)[:self.max_length]

        pos1_in_index=min(parser_length_tokens+pos1_in_index,self.max_length)
        pos2_in_index=min(parser_length_tokens+pos2_in_index,self.max_length)

        return indexed_tokens, pos1_in_index, pos2_in_index , mask, length_tokens, parser_pos1_in_index,parser_pos2_in_index , parser_indexed_tokens,parser_mask

    def tokenize_parserAndRawAddClsAvg(self, raw_tokens, head,tail, parser_tokens):
        # token -> index
        tokens = ['[CLS]']
        pos_head=head[2][0]
        pos_tail=tail[2][0]
        pos1_in_index=0
        pos1_end_index=0
        pos2_in_index=0
        pos1_end_index=0
        # pos1_in_index=pos2_in_index=pos1_end_index=pos2_end_index=0
        # low_parser_tokens=[token.lower() for token in parser_tokens]
        # for i in head[0].split(' '):
        #     if i.lower() in low_parser_tokens:
        #         pos_head.append(low_parser_tokens.index(i.lower()))
        #     else:
        #         continue
        # if pos_head==[]:
        #     pos_head=head[0].split(' ')
        #
        # for i in tail[0].split(' '):
        #     if i.lower() in low_parser_tokens:
        #         pos_tail.append(low_parser_tokens.index(i.lower()))
        #     else:
        #         continue
        # if pos_tail==[]:
        #     pos_tail=tail[0].split(' ')
        # pos_head=[low_parser_tokens.index(i.lower()) if i.lower() in low_parser_tokens else 0 for i in head[0].split(' ') ]
        # pos_tail=[low_parser_tokens.index(i.lower()) if i.lower() in low_parser_tokens else len(low_parser_tokens)-1 for i in tail[0].split(' ') ]
        #
        # cur_pos=0
        # for token in low_parser_tokens:
        #     if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
        #         tokens.append('[unused0]')
        #         pos1_in_index = len(tokens)
        #     if cur_pos == pos_tail[0]:
        #         tokens.append('[unused1]')
        #         pos2_in_index = len(tokens)
        #     if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
        #         tokens += ['[unused4]']
        #     else:
        #         tokens += self.tokenizer.tokenize(token)
        #     if cur_pos == pos_head[-1]:
        #         tokens.append('[unused2]')
        #         pos1_end_index = len(tokens)
        #
        #     if cur_pos == pos_tail[-1]:
        #         tokens.append('[unused3]')
        #         pos2_end_index = len(tokens)
        #
        #     cur_pos += 1
        #
        # cur_pos=0
        # for token in parser_tokens:
        #     if  cur_pos == 1:
        #         tokens.append('[unused0]')
        #         tokens += self.tokenizer.tokenize(token.lower())
        #         pos1_end_index=len(tokens)
        #         tokens.append('[unused2]')
        #     elif cur_pos == len(parser_tokens):
        #         tokens.append('[unused1]')
        #         pos2_in_index=len(tokens)
        #         tokens += self.tokenizer.tokenize(token.lower())
        #         pos2_end_index=len(tokens)
        #         tokens.append('[unused3]')
        #     else:
        #         tokens += self.tokenizer.tokenize(token.lower())
        #     cur_pos+=1
        # tokens.append('[SEP]')

        cur_pos=0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
                pos1_end_index = len(tokens)

            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
                pos2_end_index = len(tokens)

            cur_pos += 1
        #     ## the operation above does like: insert '[unused0]','[unused2]' before and after the head entity; insert '[unused1]', '[unused3]' before and after the tail entity
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        #import pdb
        #pdb.set_trace()

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        pos1_end_index = min(self.max_length, pos1_end_index)
        pos2_end_index = min(self.max_length, pos2_end_index)


        #parser_token
        parser_tokensed=['[CLS]']
        for parser_token in parser_tokens:
            parser_tokensed += self.tokenizer.tokenize(parser_token.lower())
        parser_indexed_tokens = self.tokenizer.convert_tokens_to_ids(parser_tokensed)

        # padding
        while len(parser_indexed_tokens) < self.max_length:
            parser_indexed_tokens.append(0)
        parser_indexed_tokens = parser_indexed_tokens[:self.max_length]
        parser_mask = np.zeros((self.max_length), dtype=np.int32)
        parser_mask[:len(parser_tokensed)] = 1


        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, len(indexed_tokens), pos1_end_index - 1, pos2_end_index - 1,parser_indexed_tokens,parser_mask
        #these positions are exactly the position of four special charaters

    def tokenize_parser_pos(self, raw_tokens, pos_head,pos_tail, parser_tokens):
        # token -> index
        tokens = ['[CLS]']
        # print(parser_tokens)
        pos_head=parser_tokens['h_pos']
        pos_tail=parser_tokens['t_pos']
        pos1_in_index = 1
        pos1_end_index = 1

        pos2_in_index = 1
        pos2_end_index = 1


        cur_pos=0
        for token in parser_tokens['tokens']:
            token = token.lower()
            if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
                pos1_end_index = len(tokens)

            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
                pos2_end_index = len(tokens)

            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        #import pdb
        #pdb.set_trace()

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)

        pos1_end_index = min(self.max_length, pos1_end_index)
        pos2_end_index = min(self.max_length, pos2_end_index)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, len(indexed_tokens), pos1_end_index - 1, pos2_end_index - 1 ,indexed_tokens,mask#these positions are exactly the position of four special charaters

    def tokenize_demo_parser_pos(self, raw_tokens, pos_head,pos_tail, parser_tokens):

        def token_parser(parser_tokens):
            tokens = ['[CLS]']
            # print(parser_tokens)
            pos_head=parser_tokens['h_pos']
            pos_tail=parser_tokens['t_pos']
            if len(pos_head)!=1:
                index=0
                try:
                    while(index<=len(pos_head)-2):
                        if pos_head[index]+1==pos_head[index+1]:
                            index+=1
                            break
                        else:
                            del pos_head[index]
                except:
                    print(parser_tokens)
            if len(pos_tail)!=1:
                index=0
                try:
                    while(index<=len(pos_tail)-2):
                        if pos_tail[index]+1==pos_tail[index+1]:
                            index+=1
                            break
                        else:
                            del pos_tail[index]
                except:
                    print(parser_tokens)
            pos1_in_index = 1
            pos1_end_index = 1

            pos2_in_index = 1
            pos2_end_index = 1

            cur_pos=0
            for token in parser_tokens['tokens']:
                token = token.lower()
                if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
                    tokens.append('[unused0]')
                    pos1_in_index = len(tokens)
                if cur_pos == pos_tail[0]:
                    tokens.append('[unused1]')
                    pos2_in_index = len(tokens)
                if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                    tokens += ['[unused4]']
                else:
                    tokens += self.tokenizer.tokenize(token)
                if cur_pos == pos_head[-1]:
                    tokens.append('[unused2]')
                    pos1_end_index = len(tokens)

                if cur_pos == pos_tail[-1]:
                    tokens.append('[unused3]')
                    pos2_end_index = len(tokens)

                cur_pos += 1
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            # padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]

            #import pdb
            #pdb.set_trace()

            # pos
            pos1 = np.zeros((self.max_length), dtype=np.int32)
            pos2 = np.zeros((self.max_length), dtype=np.int32)
            for i in range(self.max_length):
                pos1[i] = i - pos1_in_index + self.max_length
                pos2[i] = i - pos2_in_index + self.max_length

            # mask
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(tokens)] = 1

            pos1_in_index = min(self.max_length, pos1_in_index)
            pos2_in_index = min(self.max_length, pos2_in_index)

            pos1_end_index = min(self.max_length, pos1_end_index)
            pos2_end_index = min(self.max_length, pos2_end_index)

            return indexed_tokens,mask,pos1_in_index-1,pos2_in_index-1

        def token_demo(raw_tokens, pos_head,pos_tail):
            # token -> index
            pos_head=pos_head[2][0]
            pos_tail=pos_tail[2][0]
            tokens = ['[CLS]']
            cur_pos = 0
            pos1_in_index = 1
            pos1_end_index = 1

            pos2_in_index = 1
            pos2_end_index = 1

            for token in raw_tokens:
                token = token.lower()
                if cur_pos == pos_head[0]: #if current position is the head position of the entity, insert '[unused0]'.
                    tokens.append('[unused0]')
                    pos1_in_index = len(tokens)
                if cur_pos == pos_tail[0]:
                    tokens.append('[unused1]')
                    pos2_in_index = len(tokens)
                if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                    tokens += ['[unused4]']
                else:
                    tokens += self.tokenizer.tokenize(token)
                if cur_pos == pos_head[-1]:
                    tokens.append('[unused2]')
                    pos1_end_index = len(tokens)

                if cur_pos == pos_tail[-1]:
                    tokens.append('[unused3]')
                    pos2_end_index = len(tokens)

                cur_pos += 1
                ## the operation above does like: insert '[unused0]','[unused2]' before and after the head entity; insert '[unused1]', '[unused3]' before and after the tail entity
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

            # padding
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]

            #import pdb
            #pdb.set_trace()

            # pos
            pos1 = np.zeros((self.max_length), dtype=np.int32)
            pos2 = np.zeros((self.max_length), dtype=np.int32)
            for i in range(self.max_length):
                pos1[i] = i - pos1_in_index + self.max_length
                pos2[i] = i - pos2_in_index + self.max_length

            # mask
            mask = np.zeros((self.max_length), dtype=np.int32)
            mask[:len(tokens)] = 1

            pos1_in_index = min(self.max_length, pos1_in_index)
            pos2_in_index = min(self.max_length, pos2_in_index)

            pos1_end_index = min(self.max_length, pos1_end_index)
            pos2_end_index = min(self.max_length, pos2_end_index)

            return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, len(indexed_tokens), pos1_end_index - 1, pos2_end_index - 1 #these positions are exactly the position of four special charaters

        parser_indexed_tokens,parser_mask,parser_pos1_in_index,parser_pos2_in_index=token_parser(parser_tokens)

        indexed_tokens, pos1_in_index, pos2_in_index , mask, length_tokens, pos1_end_index, pos2_end_index=token_demo(raw_tokens,pos_head,pos_tail)

        indexed_tokens=parser_indexed_tokens.append(indexed_tokens)[:self.max_length]
        pos1_in_index=len(parser_indexed_tokens)+pos1_in_index
        pos2_end_index=len(parser_indexed_tokens)+pos2_end_index


        return indexed_tokens, pos1_in_index, pos2_in_index , mask, length_tokens, parser_pos1_in_index,parser_pos2_in_index , parser_indexed_tokens,parser_mask

    ##TODO tokenize relation name and description
    def tokenize_rel(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        name, description = raw_tokens
        for token in name.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')
        for token in description.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask

    def tokenize_name(self, name):
        # for FewRel 2.0
        # token -> index
        tokens = ['[CLS]']
        for token in name.split('_'):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length_name:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length_name]

        # mask
        mask = np.zeros(self.max_length_name, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask

