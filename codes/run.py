#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
from sklearn.cluster import KMeans

import numpy as np
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F

import models

from dataloader import TrainDataset, TestDataset
from dataloader import BidirectionalOneShotIterator


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Entity Aware Negative Sampling for KGE',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-sm', '--sampling_method', type=str, default='uniform')
    parser.add_argument('-tn', '--only_true_negative', action='store_true')
    parser.add_argument('-sub', '--substitution_loss', action='store_true')
    parser.add_argument('-k', type=int, default=100)
    parser.add_argument('-v', '--variance', type=int, default=290)
    parser.add_argument('-subl', '--sub_loss_weight', type=float, default=1.0)
    parser.add_argument('-subr', '--sub_regularization', type=float, default=0.1)

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=1000, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    parser.add_argument('--reorder_steps', default=1000, type=int, help='re-ordering embedding ids every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def train_step(model, optimizer, train_iterator, args):
    '''
    A single train step. Apply back-propation and return the loss
    '''

    model.train()

    optimizer.zero_grad()

    positive_sample, negative_sample, false_neg_mask, subsampling_weight, mode = next(train_iterator)

    if args.cuda:
        positive_sample = positive_sample.cuda()
        negative_sample = negative_sample.cuda()
        false_neg_mask = false_neg_mask.cuda()
        subsampling_weight = subsampling_weight.cuda()

    negative_score = model((positive_sample, negative_sample), mode=mode)

    sub_loss = 0
    if args.substitution_loss:
        sub_rel = torch.zeros(positive_sample.size(0), 1, dtype=torch.long) + args.nrelation
        if args.cuda:
            sub_rel = sub_rel.cuda()
        if mode == 'head-batch':
            subs_positive_sample = torch.cat([positive_sample[:, 0:1], sub_rel, positive_sample[:, 2:]], dim=-1)
            negative_sub_score1 = model((subs_positive_sample, negative_sample), mode='tail-batch')
            subs_positive_sample = torch.cat([positive_sample[:, 2:], sub_rel, positive_sample[:, 0:1]], dim=-1)
            negative_sub_score2 = model((subs_positive_sample, negative_sample), mode='head-batch')
        else:
            subs_positive_sample = torch.cat([positive_sample[:, 2:], sub_rel, positive_sample[:, 0:1]], dim=-1)
            negative_sub_score1 = model((subs_positive_sample, negative_sample), mode='tail-batch')
            subs_positive_sample = torch.cat([positive_sample[:, 0:1], sub_rel, positive_sample[:, 2:]], dim=-1)
            negative_sub_score2 = model((subs_positive_sample, negative_sample), mode='head-batch')

        neg_sub_score = 0.5 * (negative_sub_score1 + negative_sub_score2)
        negative_score = negative_score - args.sub_regularization * neg_sub_score
        false_neg_add_mask = torch.zeros_like(false_neg_mask).masked_fill_(false_neg_mask == 1, float('-inf')).detach()
        if args.cuda:
            false_neg_add_mask = false_neg_add_mask.cuda()
        negative_score = negative_score + false_neg_add_mask

        sub_loss += args.sub_loss_weight * \
                    ((-1 * torch.nn.functional.logsigmoid(neg_sub_score.squeeze())) *
                     false_neg_mask.squeeze()).sum() / \
                    (false_neg_mask.squeeze().sum() + 1e-20)
        sub_loss += args.sub_regularization * torch.abs(neg_sub_score.unsqueeze(-1)).mean()

    if args.negative_adversarial_sampling:
        # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
        neg_weights = F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
        negative_score = (neg_weights * F.logsigmoid(-negative_score)).sum(dim=1)
    else:
        negative_score = F.logsigmoid(-negative_score).mean(dim=1)

    positive_score = model(positive_sample)

    positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

    if args.uni_weight:
        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
    else:
        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
    # positive_sample_loss = - positive_score.mean()
    # negative_sample_loss = - negative_score.mean()

    loss = (positive_sample_loss + negative_sample_loss) / 2
    loss += sub_loss

    if args.regularization != 0.0:
        # Use L3 regularization for ComplEx and DistMult
        regularization = args.regularization * (
                model.entity_embedding.norm(p=3) ** 3 +
                model.relation_embedding.norm(p=3).norm(p=3) ** 3
        )
        loss = loss + regularization
        regularization_log = {'regularization': regularization.item()}
    else:
        regularization_log = {}

    if torch.isnan(loss).item():
        # when only get false negative samples, do not update parameters
        loss = torch.tensor([0.0])
        positive_sample_loss = torch.tensor([0.0])
        negative_sample_loss = torch.tensor([0.0])
    else:
        loss.backward()
        optimizer.step()

    log = {
        **regularization_log,
        'positive_sample_loss': positive_sample_loss.item(),
        'negative_sample_loss': negative_sample_loss.item(),
        'loss': loss.item()
    }

    return log


def test_step(model, test_triples, all_true_triples, args):
    '''
    Evaluate the model on test or valid datasets
    '''

    model.eval()
    # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
    # Prepare dataloader for evaluation
    test_dataloader_head = DataLoader(
        TestDataset(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'head-batch'
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TestDataset.collate_fn
    )

    test_dataloader_tail = DataLoader(
        TestDataset(
            test_triples,
            all_true_triples,
            args.nentity,
            args.nrelation,
            'tail-batch'
        ),
        batch_size=args.test_batch_size,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=TestDataset.collate_fn
    )

    test_dataset_list = [test_dataloader_head, test_dataloader_tail]

    logs = []

    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])

    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                if args.cuda:
                    positive_sample = positive_sample.cuda()
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                batch_size = positive_sample.size(0)

                score = model((positive_sample, negative_sample), mode)
                score += filter_bias

                # Explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim=1, descending=True)

                if mode == 'head-batch':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)

                for i in range(batch_size):
                    # Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    # ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0 / ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })

                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    return metrics


def reorder_embeddings(model, train_dataset_head, train_dataset_tail, args):
    ids = torch.tensor([train_dataset_head.entity_proxy.get_real_id(vid) for vid in range(train_dataset_head.nentity)])
    if args.cuda:
        ids = ids.cuda()
    old_embeddings = model.get_ent_embeddings(ids)
    kmeans = KMeans(n_clusters=args.k).fit(old_embeddings.cpu())

    # sorting the clusters based on distance of centroids
    centers = torch.tensor(kmeans.cluster_centers_)
    center_labels = list(range(args.k))
    cluster_order_map = {}  # {center_label: order}
    order = 0
    cluster_order_map[0] = order
    current_center = centers[0]
    centers = centers[1:]
    center_labels = center_labels[1:]
    order += 1
    while len(centers) > 0:
        dists = torch.nn.functional.pairwise_distance(current_center.unsqueeze(0), centers)
        _, top_i = dists.topk(1, largest=False)
        top_i = top_i.item()
        current_label = center_labels[top_i]
        current_center = centers[top_i]
        cluster_order_map[current_label] = order
        center_labels = center_labels[:top_i] + center_labels[top_i + 1:]
        centers = torch.cat([centers[:top_i], centers[top_i + 1:]], dim=0)
        order += 1
    new_labels = [cluster_order_map[l] for l in kmeans.labels_]

    _, old_ids = torch.sort(torch.tensor(new_labels))
    old2new = {}
    for new_id, old_id in enumerate(old_ids):
        old2new[old_id.item()] = new_id
    train_dataset_head.update_proxy(old2new)
    train_dataset_tail.update_proxy(old2new)


def main(args):
    print(args)
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')
    
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    #All true triples
    all_true_triples = train_triples + valid_triples + test_triples

    # Build KGE model
    kge_model_class = getattr(models, args.model)
    kge_model = kge_model_class(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    
    if args.do_train:
        # Set training dataloader iterator
        train_dataset_head = TrainDataset(args, train_triples, nentity, nrelation, 'head-batch')
        train_dataloader_head = DataLoader(
            train_dataset_head,
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        train_dataset_tail = TrainDataset(args, train_triples, nentity, nrelation, 'tail-batch')
        train_dataloader_tail = DataLoader(
            train_dataset_tail,
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []
        
        #Training Loop
        for step in range(init_step, args.max_steps):
            
            log = train_step(kge_model, optimizer, train_iterator, args)
            
            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0 and step > 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)

            if args.sampling_method == 'gaussian' and step % args.reorder_steps == 0:
                reorder_embeddings(kge_model, train_dataset_head, train_dataset_tail, args)
        
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)


if __name__ == '__main__':
    main(parse_args())
