import numpy as np
from sklearn.datasets import load_svmlight_file
import datetime
from xgboost import XGBRegressor

from QueryDocument import Query

N_WORKERS = 6;

class Data:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode

    def LoadFromSource(self, filename):
        print("Loading from source data was started", datetime.datetime.now())
        self.X, self.y, documents_query = load_svmlight_file(filename, query_id=True)
        print("Data was loaded!", datetime.datetime.now())

        print("Creating queries ...", datetime.datetime.now())
        self.queries = []
        self.query_document_indices = []
        self.unique_query_indices = np.unique(documents_query)
        for query_id in self.unique_query_indices:
            self.query_document_indices.append(np.where(documents_query == query_id)[0])
            self.queries.append(Query(self.y[self.query_document_indices[-1]], self.test_mode))
            if query_id % 1000 == 0:
                print(query_id)
        print("Queries were created", datetime.datetime.now())
        return self

EPOCH = 0
def ObjectiveFunction(data):
    def _objective_function(y_true, y_pred):
        global EPOCH
        print("Epoch = ", EPOCH, ";", datetime.datetime.now())
        EPOCH += 1
        for query_id, indices in enumerate(data.query_document_indices):
            data.queries[query_id].UpdateScores(y_pred[indices])

        return np.hstack(q.numerators for q in train_data.queries), \
              np.hstack(q.denominators for q in train_data.queries)
    return _objective_function


class LambdaMART:
    def __init__(self, train_data, learning_rate=0.1, n_trees=300, max_depth=3):
        objective_func = ObjectiveFunction(train_data)
        self.xgb_classifier = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate,
                                           n_estimators=n_trees, n_jobs=4, objective=objective_func)

    def fit(self, train_data):
       self.xgb_classifier.fit(train_data.X, train_data.y)
       return self

    def predict(self, test_data):
        results = self.xgb_classifier.predict(test_data.X)
        for query_id, indices in enumerate(test_data.query_document_indices):
             test_data.queries[query_id].UpdateScores(results[indices])

def SubmitPrediction(filename, test_data):
    with open(filename, 'w') as f:
        f.write("QueryId,DocumentId\n")
        document_base_idx = 0
        for query_idx, unique_query_idx in enumerate(test_data.unique_query_indices):
            doc_pos = test_data.queries[query_idx].positions
            pos_for_write = np.full((len(doc_pos), ), document_base_idx)
            pos_for_write[doc_pos-1] += np.arange(1, len(doc_pos)+1)
            for pos in pos_for_write:
               f.write(str(unique_query_idx)+","+str(pos)+"\n")
            document_base_idx += len(test_data.queries[query_idx].positions)

if __name__ == '__main__':
    filename_base = "sm_xgb_0.3-1500-8.txt"
    print(filename_base)

    train_data = Data().LoadFromSource("data/train.txt")
    model = LambdaMART(train_data, n_trees=1500, max_depth=8, learning_rate=0.3)
    model.fit(train_data)

    test_data = Data(True).LoadFromSource("data/test.txt")
    model.predict(test_data)
    SubmitPrediction("submission_"+filename_base, test_data)

    test_data2 = Data().LoadFromSource("data/train.txt")
    model.predict(test_data2)
    SubmitPrediction("td_"+filename_base, train_data)

    ndcg_sum = 0.0
    count_invalid = 0
    with open("qa_"+filename_base, 'w') as qa_file:
        for q in test_data2.queries:
            qa_file.write("QUERY: ===>\n")
            ass_sc = np.zeros((q.count_docs, ))
            ass_sc[q.positions-1] = q.asessor_scores
            qa_file.write("SORTED ASESSOR SCORES: "+str(ass_sc)+"\n")
            qa_file.write("NDCG: "+str(q.GetNDCG())+"\n\n")
            if q.GetNDCG() != 0:
                ndcg_sum += q.GetNDCG()
            else:
                count_invalid += 1
        qa_file.write("\n\n<=======> Avg score <=======>\n")
        qa_file.write(str(ndcg_sum / (len(test_data2.queries) - count_invalid)))