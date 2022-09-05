from .kge_model import KGEModel


class DistMult(KGEModel):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding, double_relation_embedding):
        super(DistMult, self).__init__(model_name, nentity, nrelation, hidden_dim, gamma,
                                       double_entity_embedding, double_relation_embedding)

    def scoring(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score
