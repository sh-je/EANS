from .kge_model import KGEModel
import torch
import torch.nn as nn


class TransD(KGEModel):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding, double_relation_embedding):
        super(TransD, self).__init__(model_name, nentity, nrelation, hidden_dim, gamma,
                                     double_entity_embedding, double_relation_embedding)
        self.ent_transfer = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        self.rel_transfer = nn.Parameter(torch.zeros(nrelation + 1, self.relation_dim))
        nn.init.uniform_(
            tensor=self.ent_transfer,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        nn.init.uniform_(
            tensor=self.rel_transfer,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def get_ent_embeddings(self, ids):
        embs = torch.index_select(
            self.entity_embedding,
            dim=0,
            index=ids
        ).detach()
        transfers = torch.index_select(
            self.ent_transfer,
            dim=0,
            index=ids
        ).detach()
        return torch.cat([embs, transfers], dim=-1)

    def scoring(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def transfer(self, ent, e_transfer, r_transfer):
        ent = r_transfer * torch.sum(e_transfer * ent, dim=-1, keepdim=True) + ent
        return ent

    def forward(self, sample, mode='single'):
        head, relation, tail = self.get_sample_embeddings(sample, mode)
        if mode == 'single':
            h_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)
            r_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            t_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            h_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            r_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            t_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            h_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)
            r_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            t_transfer = torch.index_select(
                self.ent_transfer,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        else:
            raise ValueError('mode %s not supported' % mode)
        head = self.transfer(head, h_transfer, r_transfer)
        tail = self.transfer(tail, t_transfer, r_transfer)
        score = self.scoring(head, relation, tail, mode)
        return score
