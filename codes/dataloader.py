#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset


class EntityIdProxy:
    def __init__(self, nentity):
        self.virtual2real = {}
        self.real2virtual = {}
        for i in range(nentity):
            self.virtual2real[i] = i
            self.real2virtual[i] = i

    def get_real_id(self, virtual_id):
        return self.virtual2real[virtual_id]

    def get_virtual_id(self, real_id):
        return self.real2virtual[real_id]

    def update(self, old2new):
        new_virtual2real = {}
        new_real2virtual = {}
        for real_id, virtual_id in self.real2virtual.items():
            new_real2virtual[real_id] = old2new[virtual_id]
        for real_id, virtual_id in new_real2virtual.items():
            new_virtual2real[virtual_id] = real_id
        self.real2virtual = new_real2virtual
        self.virtual2real = new_virtual2real
        return None


class TrainDataset(Dataset):
    def __init__(self, opts, triples, nentity, nrelation, mode):
        self.opts = opts
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = self.opts.negative_sample_size
        self.mode = mode
        self.entity_proxy = EntityIdProxy(nentity)
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        v_head = self.entity_proxy.get_virtual_id(head)
        v_tail = self.entity_proxy.get_virtual_id(tail)

        if self.opts.sampling_method == 'uniform':
            if self.opts.only_true_negative:
                false_neg_mask = np.zeros(self.opts.negative_sample_size)
                while negative_sample_size < self.negative_sample_size:
                    negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
                    if self.mode == 'head-batch':
                        mask = np.in1d(
                            negative_sample,
                            self.true_head[(relation, v_tail)],
                            assume_unique=True,
                            invert=True
                        )
                    elif self.mode == 'tail-batch':
                        mask = np.in1d(
                            negative_sample,
                            self.true_tail[(v_head, relation)],
                            assume_unique=True,
                            invert=True
                        )
                    else:
                        raise ValueError('Training batch mode %s not supported' % self.mode)
                    negative_sample = negative_sample[mask]
                    negative_sample_list.append(negative_sample)
                    negative_sample_size += negative_sample.size
            else:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size)
                if self.mode == 'head-batch':
                    false_neg_mask = np.in1d(
                        negative_sample,
                        self.true_head[(relation, v_tail)],
                        assume_unique=True,
                    )
                elif self.mode == 'tail-batch':
                    false_neg_mask = np.in1d(
                        negative_sample,
                        self.true_tail[(v_head, relation)],
                        assume_unique=True,
                    )
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                negative_sample_list.append(negative_sample)

        elif self.opts.sampling_method == 'gaussian':
            if self.opts.only_true_negative:
                false_neg_mask = np.zeros(self.opts.negative_sample_size)
                while negative_sample_size < self.negative_sample_size:
                    if self.mode == 'head-batch':
                        negative_sample = np.random.normal(0.0, 1.0, self.opts.negative_sample_size) * self.opts.variance + v_head
                        negative_sample = (negative_sample.astype(np.int64) + self.nentity) % self.nentity
                        mask = np.in1d(
                            negative_sample,
                            self.true_head[(relation, v_tail)],
                            assume_unique=True,
                            invert=True
                        )
                    elif self.mode == 'tail-batch':
                        negative_sample = np.random.normal(0.0, 1.0, self.opts.negative_sample_size) * self.opts.variance + v_tail
                        negative_sample = (negative_sample.astype(np.int64) + self.nentity) % self.nentity
                        mask = np.in1d(
                            negative_sample,
                            self.true_tail[(v_head, relation)],
                            assume_unique=True,
                            invert=True
                        )
                    else:
                        raise ValueError('Training batch mode %s not supported' % self.mode)
                    negative_sample = negative_sample[mask]
                    negative_sample_list.append(negative_sample)
                    negative_sample_size += negative_sample.size
            else:
                if self.mode == 'head-batch':
                    negative_sample = np.random.normal(0.0, 1.0, self.opts.negative_sample_size) * self.opts.variance + v_head
                    negative_sample = (negative_sample.astype(np.int64) + self.nentity) % self.nentity
                    false_neg_mask = np.in1d(
                        negative_sample,
                        self.true_head[(relation, v_tail)],
                        assume_unique=True
                    )
                elif self.mode == 'tail-batch':
                    negative_sample = np.random.normal(0.0, 1.0, self.opts.negative_sample_size) * self.opts.variance + v_tail
                    negative_sample = (negative_sample.astype(np.int64) + self.nentity) % self.nentity
                    false_neg_mask = np.in1d(
                        negative_sample,
                        self.true_tail[(v_head, relation)],
                        assume_unique=True
                    )
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                negative_sample_list.append(negative_sample)
        else:
            raise ValueError('Sampling method %s not supported' % self.opts.sampling_method)
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = [self.entity_proxy.get_real_id(vid) for vid in negative_sample]
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
        false_neg_mask = torch.FloatTensor(false_neg_mask)
        return positive_sample, negative_sample, false_neg_mask, subsampling_weight, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        false_neg_mask = torch.stack([_[2] for _ in data], dim=0)
        subsample_weight = torch.cat([_[3] for _ in data], dim=0)
        mode = data[0][4]
        return positive_sample, negative_sample, false_neg_mask, subsample_weight, mode
    
    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation-1) not in count:
                count[(tail, -relation-1)] = start
            else:
                count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))                 

        return true_head, true_tail

    def update_proxy(self, old2new):
        # update proxy
        self.entity_proxy.update(old2new)
        # update true triples
        new_true_head = {}
        new_true_tail = {}
        for relation, tail in self.true_head:
            new_tail = old2new[tail]
            new_true_h = np.array([old2new[head] for head in self.true_head[(relation, tail)]])
            new_true_head[(relation, new_tail)] = new_true_h
        for head, relation in self.true_tail:
            new_head = old2new[head]
            new_true_t = np.array([old2new[head] for head in self.true_tail[(head, relation)]])
            new_true_tail[(new_head, relation)] = new_true_t
        self.true_head = new_true_head
        self.true_tail = new_true_tail

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
