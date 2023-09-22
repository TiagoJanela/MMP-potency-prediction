# imports
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
# Sklearn
from sklearn import neighbors, metrics
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

from regression_shap_mmp.ML.ml_utils_reg import tanimoto_from_dense
from regression_shap_mmp.sveta.sveta.svm import ExplainingSVR

os.environ["TF_DETERMINISTIC_OPS"] = "1"
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


class MLModel:
    def __init__(self, data, ml_algorithm, opt_metric="neg_mean_absolute_error", reg_class="regression",
                 parameters='grid', cv_fold=5, random_seed=2002):

        self.data = data
        self.ml_algorithm = ml_algorithm
        self.opt_metric = opt_metric
        self.reg_class = reg_class
        self.cv_fold = cv_fold
        self.seed = random_seed
        self.parameters = parameters
        self.h_parameters = self.hyperparameters()
        self.model, self.cv_results = self.cross_validation()
        self.best_params = self.optimal_parameters()
        self.model = self.final_model()

    def hyperparameters(self):
        if self.parameters == "grid":

            if self.reg_class == "regression":
                if self.ml_algorithm == "MR":
                    return {'strategy': ['median']
                            }
                elif self.ml_algorithm == "SVR":
                    return {'C': [0.001, 0.1, 1, 10, 100, 1000, 10000],
                            }
                elif self.ml_algorithm == "RFR":
                    return {'n_estimators': [50, 100, 200],
                            'max_features': ['sqrt', 'log2'],
                            'min_samples_split': [2, 3, 5, 10],
                            'min_samples_leaf': [1, 2, 5, 10],
                            }
                elif self.ml_algorithm == "kNN":
                    return {"n_neighbors": [3, 5],
                            }
                elif self.ml_algorithm == "1-NN":
                    return {"n_neighbors": [1],
                            }

    def cross_validation(self):

        if self.reg_class == "regression":
            opt_metric = self.opt_metric
            if self.ml_algorithm == "MR":
                model = DummyRegressor()
            elif self.ml_algorithm == "SVR":
                model = ExplainingSVR()
            elif self.ml_algorithm == "RFR":
                model = RandomForestRegressor(random_state=self.seed)
            elif self.ml_algorithm == "kNN":
                model = neighbors.KNeighborsRegressor(metric='jaccard')
            elif self.ml_algorithm == "1-NN":
                model = neighbors.KNeighborsRegressor(metric='jaccard')

        cv_results = GridSearchCV(model,
                                  param_grid=self.h_parameters,
                                  cv=self.cv_fold,
                                  scoring=opt_metric,
                                  n_jobs=14)

        if self.ml_algorithm == "SVR":
            cv_results.fit(csr_matrix(self.data.features), self.data.labels)
        else:
            cv_results.fit(self.data.features, self.data.labels)

        return model, cv_results

    def optimal_parameters(self):
        best_params = self.cv_results.cv_results_['params'][self.cv_results.best_index_]
        return best_params

    def final_model(self):
        model = self.model.set_params(**self.best_params)
        if self.ml_algorithm == "SVR":
            return model.fit(csr_matrix(self.data.features), self.data.labels)
        else:
            return model.fit(self.data.features, self.data.labels)


class Model_Evaluation:
    def __init__(self, model, data, tr_data=None, model_id=None, model_loaded=None, reg_class="regression", ):
        self.reg_class = reg_class
        self.model_id = model_id
        self.model = model
        self.data = data
        self.tr_data = tr_data
        self.model_loaded = model_loaded
        self.labels, self.y_pred, self.predictions = self.model_predict(data)
        self.pred_performance = self.prediction_performance(data)

    def model_predict(self, data):

        if self.reg_class == "regression":

            if self.model_id == "SVR":
                data_features = csr_matrix(data.features)
            else:
                data_features = data.features

            if self.model_loaded is not None:
                y_prediction = self.model.predict(data_features)
            else:
                y_prediction = self.model.model.predict(data_features)
            labels = self.data.labels

            predictions = pd.DataFrame(list(
                zip(data.cid, labels, y_prediction, data.smiles, data.mmp_id, data.analog_series_id, data.train_test)),
                                       columns=["cid", "Experimental", "Predicted", 'smiles', 'mmp_id',
                                                'analog_series_id', 'train_test'])
            predictions['Target ID'] = data.target[0]
            predictions['Algorithm'] = self.model_id
            predictions['Residuals'] = [label_i - prediction_i for label_i, prediction_i in zip(labels, y_prediction)]
            predictions["similarity"] = data.similarity
            predictions["dPot"] = data.dPot

            if self.model_id == '1-NN' and self.tr_data is not None:
                nn = self.model.model.kneighbors(X=data.features, n_neighbors=1, return_distance=False)
                predictions['1-NN'] = self.tr_data[nn].cid
                predictions['1-NN_smiles'] = self.tr_data[nn].smiles
            else:
                predictions['1-NN'] = np.nan
                predictions['1-NN_smiles'] = np.nan

            return labels, y_prediction, predictions

    def prediction_performance(self, data, nantozero=False) -> pd.DataFrame:

        if self.reg_class == "regression":

            labels = self.labels
            pred = self.y_pred

            fill = 0 if nantozero else np.nan
            if len(pred) == 0:
                mae = fill
                mse = fill
                rmse = fill
                r2 = fill
                r = fill
            else:
                mae = mean_absolute_error(labels, pred)
                mse = metrics.mean_squared_error(labels, pred)
                rmse = metrics.mean_squared_error(labels, pred, squared=False)
                r2 = metrics.r2_score(labels, pred)
                r = stats.pearsonr(labels, pred)[0]

            target = data.target[0]
            model_name = self.model_id

            result_list = [{"MAE": mae,
                            "MSE": mse,
                            "RMSE": rmse,
                            "R2": r2,
                            "r": r,
                            "Dataset size": len(labels),
                            "Target ID": target,
                            "Algorithm": model_name}
                           ]

            # Prepare result dataset
            results = pd.DataFrame(result_list)
            results.set_index(["Target ID", "Algorithm", "Dataset size"], inplace=True)
            results.columns = pd.MultiIndex.from_product([["Value"], ["MAE", "MSE", "RMSE", "R2", "r"]],
                                                         names=["Value", "Metric"])
            results = results.stack().reset_index().set_index("Target ID")

            return results
