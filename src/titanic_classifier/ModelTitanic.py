import os
import joblib
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from feature_engine.selection import DropConstantFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from src.consts import TrainTestData, ModelsPerformances
from src.dataprep.Preprocess import Preprocess


class ModelTitanic:
    classifiers = {  # classifiers to examine for model selection
        "LogisticRegression": LogisticRegression(),
        "DummyClassifier": DummyClassifier(strategy='most_frequent'),
        "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic'),
        "RandomForestClassifier": RandomForestClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "ExtraTreeClassifier": ExtraTreeClassifier(),
        "ExtraTreesClassifier": ExtraTreeClassifier(),
        "AdaBoostClassifier": AdaBoostClassifier(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "RidgeClassifier": RidgeClassifier(),
        "SGDClassifier": SGDClassifier(),
        "BaggingClassifier": BaggingClassifier(),
        "BernoulliNB": BernoulliNB(),
        "SVC": SVC(),
        "CatBoostClassifier": CatBoostClassifier(silent=True),
    }

    def __init__(self, df_train: pd.DataFrame = None, label_encode: bool = True, one_hot_encode: bool = True):
        """
        Will make a model pipeline with preprocees step, model step etc.

        Args:
            df_train: titanic dataframe kind data with columns:
            ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
            'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'].
            label_encode: flag for performing label encoding (not necessary for lightgbm and catboost).
            one_hot_encode: flag for performing one hot encoding (not necessary for lightgbm and catboost).
        """
        assert df_train is not None, 'train dataframe shouldn\'t be None for ModelTitanic class'
        self._df_train = df_train
        self.preprocess = Preprocess(df_train=self._df_train, label_encode=label_encode, one_hot_encode=one_hot_encode)
        self._X_train = self.preprocess.df_train  # without dropping Survived
        self._y_train = self._X_train['Survived']  # target variable
        self._X_train = self._X_train.drop(columns=['Survived'], axis=1)
        self.features = list(self._X_train.columns)
        self.pipeline = Pipeline(steps=[], verbose=True)  # fit pipeline to be performed on test also

    @property
    def X_train(self) -> pd.DataFrame:
        return self._X_train

    @property
    def y_train(self) -> pd.DataFrame:
        return self._y_train

    def get_pipeline(self, model=None, model_name: str = 'model', train: bool = False) -> Pipeline:
        """
        Return a pipeline to preprocess data and bundle with a model.
        When used for model selection we don't need to perform preprocess and feature selection everytime so
        pipeline will be just the model.

        Args:
            model: scikit-learn instantiated model object, e.g. XGBClassifier (or scikit-learn model compatible
            e.g. xgboost library).
            model_name: model name to be in the pipeline.
            train: if used for model selection then we don't need to perform feature selection everytime as it is the
            same regardless of what model is being used to classify.

        Returns:
            Pipeline: Pipeline steps.
        """
        if train and os.path.exists(TrainTestData.TRAIN_PREPROCESSED_PATH):  # used for model selection
            assert model is not None, 'model shouldn\'t be None for model selection run'
            # function used for model selection during train
            pipeline = Pipeline(steps=[('model', model)])
            pipeline.set_output(transform="pandas")
            return pipeline

        # preprocess pipeline + feature selection + model to classify for test purposes
        pipeline = self.preprocess.preprocess_pipeline
        polynomial_features_step = ['poly', PolynomialFeatures(include_bias=False, interaction_only=True)]
        # dropping interaction features that are columns of zeros
        drop_constant_features_step = ['drop_constant_features', DropConstantFeatures()]
        feature_selection_step = ['feature_selection', RFECV(estimator=LogisticRegression(),
                                                             step=1, cv=5, scoring='f1')]
        # appending all steps to the pipeline
        pipeline.steps.append(polynomial_features_step)
        pipeline.steps.append(drop_constant_features_step)
        pipeline.steps.append(feature_selection_step)
        pipeline.set_output(transform="pandas")

        # running for creating preprocessed train dataset + feature selection
        if train and model is None:
            return pipeline

        # running for testing model with preprocessing data for example running on test dataset
        pipeline.steps.append([model_name, model])
        return pipeline

    def select_model(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Test a variety of classifiers and return their performance metrics on training data.
        modified code from:
        https://practicaldatascience.co.uk/machine-learning/how-to-create-a-contractual-churn-model.

        Args:
            X: Pandas dataframe containing X_train data.
            y: Pandas dataframe containing y_train data.

        Return:
            df: Pandas dataframe containing model performance data.
        """
        max_pipeline_score = 0

        # df of model performance
        df_models_performances = pd.DataFrame(columns=['model', 'time_executed', 'run_time_sec',
                                                       'f1', 'f1_std', 'roc_auc', 'roc_auc_std'])

        for key in tqdm(self.classifiers):
            start_time = time.time()
            pipeline = self.get_pipeline(model=self.classifiers[key], model_name=key, train=True)
            cv_f1 = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
            cv_roc_auc = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
            if cv_f1.mean() > max_pipeline_score:
                max_pipeline_score = cv_f1.mean()
                joblib.dump(pipeline, ModelsPerformances.SELECTED_MODEL_PIPE_PATH)

            # dd/mm/YY H:M:S
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

            row = {'model': key,
                   'time_executed': dt_string,
                   'run_time_sec': format(round((time.time() - start_time), 2)),
                   'f1': cv_f1.mean(),
                   'f1_std': cv_f1.std(),
                   'roc_auc': cv_roc_auc.mean(),
                   'roc_auc_std': cv_roc_auc.std(),
                   }

            df_models_performances = df_models_performances.append(row, ignore_index=True)

        df_models_performances = df_models_performances.sort_values(by='f1', ascending=False, ignore_index=True)
        df_models_performances.to_csv(ModelsPerformances.MODELS_PERFORMANCES_PATH)
        return df_models_performances

    def perform_model_selection(self, ):
        """
        Perform model selection, first creating the preprocessed train dataframe once at max as it is the same for
        all models (and actually not necessary for catboost/lightgbm model). Saves results in a csv file.
        """
        # creating preprocessed train dataset
        if not os.path.exists(TrainTestData.TRAIN_PREPROCESSED_PATH):
            print('Preprocessing train...')
            preprocess_pipeline = self.get_pipeline(train=True)
            self._X_train = preprocess_pipeline.fit_transform(self._X_train, self._y_train)
            # saving preprocessed train
            self._X_train.to_csv(TrainTestData.TRAIN_PREPROCESSED_PATH)
        else:
            print('Loading preprocessed train...')
            self._X_train = pd.read_csv(TrainTestData.TRAIN_PREPROCESSED_PATH)

        # model selection if not performed
        if not os.path.exists(ModelsPerformances.SELECTED_MODEL_PIPE_PATH):
            print('Performing Model Selection...')
            df_models_performances = self.select_model(self._X_train, self._y_train)
            print(df_models_performances)

    def train_ridge_classifier(self):
        pass
