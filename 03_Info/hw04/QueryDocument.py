import numpy as np

SIGMA = 1.0
LIMIT_DELTA = 50 # Для экспоненты

class Query:
    def __init__(self, asessor_scores, test_mode=False):
        self.test_mode = test_mode
        self.count_docs = len(asessor_scores)
        if self.test_mode:
            self.UpdateScores(np.ones((self.count_docs, )))
            return

        self.asessor_scores = np.copy(asessor_scores)
        scores_sorted = -np.sort(-self.asessor_scores)
        self.dcg_norm = np.sum((2.0 ** scores_sorted - 1) / np.log(np.arange(2, self.count_docs+2))) # log2 - не нужен (мы его не пишем в дельте)
        if self.dcg_norm == 0:
            self.dcg_norm = 1.0

        self.permutations = np.tile(np.arange(0, self.count_docs), (self.count_docs, 1))
        self.UpdateScores(np.zeros((self.count_docs, )))
#        return self

    def __UpdateDeltaNDCG(self):
        self.delta_ndcg = np.abs((-1.0 / np.log(self.positions.reshape(-1, 1)+1) + 1.0 / np.log(self.positions[self.permutations]+1)) * \
                                 (((2 ** self.asessor_scores.reshape(-1, 1)) - 1) - ((2 ** self.asessor_scores[self.permutations]) - 1))) /\
                          self.dcg_norm

    def GetNDCG(self):
        return np.sum((2.0 ** self.asessor_scores - 1) / np.log(self.positions+1)) / self.dcg_norm

    def UpdateScores(self, new_scores):
        self.positions = np.zeros((self.count_docs, ), dtype=np.int32)
        self.positions[np.argsort(-new_scores).astype(np.int32)] = np.arange(1, self.count_docs+1)
        self.scores = np.copy(new_scores)
        if self.test_mode == True:
            return

        self.__UpdateDeltaNDCG()

        delta_scores = np.abs(SIGMA * (self.scores.reshape((-1, 1)) - self.scores[self.permutations]))
        delta_scores[delta_scores >= LIMIT_DELTA / SIGMA] = LIMIT_DELTA / SIGMA
        self.ro_ij = 1.0 / (1 + np.exp(SIGMA * delta_scores))
        #self.lambda_ij = -SIGMA * self.delta_ndcg * self.ro_ij

        correct_permutations = (self.asessor_scores.reshape((-1, 1)) > self.asessor_scores[self.permutations]).astype(np.int8)
        incorrect_permutations = (self.asessor_scores.reshape((-1, 1)) < self.asessor_scores[self.permutations]).astype(np.int8)
        valid_permutations = correct_permutations + incorrect_permutations

        # grad
        self.numerators = -np.sum(self.delta_ndcg * self.ro_ij * correct_permutations - \
                                  self.delta_ndcg * self.ro_ij * incorrect_permutations, axis=1)

        #hess
        self.denominators = np.sum(self.delta_ndcg * SIGMA * self.ro_ij * (1.0 - self.ro_ij) * valid_permutations, axis=1)
        self.denominators[self.denominators == 0] = 1
