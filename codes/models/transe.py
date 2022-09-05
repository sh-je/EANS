from .kge_model import KGEModel
import torch


class TransE(KGEModel):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding, double_relation_embedding):
        super(TransE, self).__init__(model_name, nentity, nrelation, hidden_dim, gamma,
                                     double_entity_embedding, double_relation_embedding)

    def scoring(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score
