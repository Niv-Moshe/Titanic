from preprocess import Preprocess
from model import ModelTitanic, train_path, test_path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import joblib
from sklearn.preprocessing import PolynomialFeatures
from feature_engine.selection import DropConstantFeatures
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import cross_val_score


def create_train_test_files():
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print('Creating Train and Test files')
        # create train test split
        titanic = pd.read_csv('data/titanic.csv')
        train, test = train_test_split(titanic, test_size=0.2, random_state=42)
        train.to_csv(train_path, index=False)
        test.to_csv(train_path, index=False)
    else:
        print('Train and Test files already exist')


def main():
    create_train_test_files()
    # Preprocess(df_path=train_path)
    model = ModelTitanic(df_train_path=train_path)
    model.train()
    pass


if __name__ == "__main__":
    # main()
    # pipe = joblib.load('results/model_selection_pipeline.pkl')
    # params = pipe.steps['model'].get_params()
    # print(params)
    preprocessor = Preprocess(df_path=train_path, label_encode=False, one_hot_encode=False)
    df_train = preprocessor.get_df()
    survived = df_train['Survived']  # target variable

    pipeline = preprocessor.get_preprocess_pipeline()
    # pipeline.steps.append(['poly', PolynomialFeatures(include_bias=False, interaction_only=True)])
    #
    # # dropping interaction features that are columns of zeros
    # pipeline.steps.append(['drop_constant_features', DropConstantFeatures()])
    # pipeline.steps.append(['feature_select',
    #                        RFECV(estimator=LogisticRegression(), step=1, cv=5, scoring='roc_auc')])
    pipeline.steps.append(['model', CatBoostClassifier(silent=True)])
    # pool_train = Pool(df_train, survived, cat_features=["Sex", "Embarked", "Honorifics", "CabinChar"])

    print(pipeline)
    cv = cross_val_score(pipeline, df_train, survived, cv=5, scoring='roc_auc')
    print(cv.mean())

