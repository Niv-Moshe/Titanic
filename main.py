import os
import pandas as pd
from sklearn.model_selection import train_test_split
import src.consts as consts
from src.titanic_classifier.ModelTitanic import ModelTitanic


def create_train_test_files():
    if not (os.path.exists(consts.TRAIN_PATH) and os.path.exists(consts.TEST_PATH)):
        print('Creating Train and Test files')
        # create train test split
        titanic = pd.read_csv('data/titanic.csv')
        train, test = train_test_split(titanic, test_size=0.2, random_state=42)
        train.to_csv(consts.TRAIN_PATH, index=False)
        test.to_csv(consts.TEST_PATH, index=False)
    else:
        print('Train and Test files already exist')


def main():
    create_train_test_files()
    df_train = pd.read_csv(consts.TRAIN_PATH)
    model = ModelTitanic(df_train=df_train)
    model.perform_model_selection()
    """
    import time
    import joblib
    from feature_engine.selection import DropConstantFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from src.dataprep.Preprocess import Preprocess

    preprocessor = Preprocess(df_train=df_train, label_encode=False, one_hot_encode=False)
    df_train = preprocessor.df_train
    X = df_train.drop(columns=['Survived'], axis=1)
    object_cols = X.select_dtypes('object').columns
    X[object_cols] = X[object_cols].astype('category')
    original_types = dict(X.dtypes)

    y = df_train['Survived']  # target variable
    pipeline = preprocessor.preprocess_pipeline
    X = pipeline.fit_transform(X)
    for col in X.columns:
        if col in original_types:
            X[col] = X[col].astype(original_types[col])
        else:
            X[col] = X[col].astype('category')
    # print(X.info())
    model = LGBMClassifier()
    cv = cross_val_score(model, X, y, cv=5, scoring='f1')
    print("------------")
    print(f"LGBMClassifier roc auc: {cv.mean()}")
    start_time = time.time()
    model = CatBoostClassifier(cat_features=list(X.select_dtypes('category').columns), silent=True)  # , one_hot_max_size=50)
    cv = cross_val_score(model, X, y, cv=5, scoring='f1')
    end_time = time.time()
    print("------------")
    print(f"CatBoostClassifier time took: {(end_time - start_time) / 60:.2f}")
    print(f"CatBoostClassifier roc auc: {cv.mean()}")
    """

    """
    # pipe = joblib.load(consts.SELECTED_MODEL_PIPE_PATH)
    # params = pipe.steps['model'].get_params()
    # print(params)

    preprocessor = Preprocess(df_train=df_train, label_encode=False, one_hot_encode=False)
    df_train = preprocessor.df_train
    survived = df_train['Survived']  # target variable

    pipeline = preprocessor.preprocess_pipeline
    # pipeline.steps.append(['poly', PolynomialFeatures(include_bias=False, interaction_only=True)])

    # dropping interaction features that are columns of zeros
    # pipeline.steps.append(['drop_constant_features', DropConstantFeatures()])
    # pipeline.steps.append(['feature_select',
    #                        RFECV(estimator=LogisticRegression(), step=1, cv=5, scoring='roc_auc')])
    # pipeline.steps.append(['model', CatBoostClassifier(silent=True)])
    # # pool_train = Pool(df_train, survived, cat_features=["Sex", "Embarked", "Honorifics", "CabinChar"])
    #
    # print(pipeline)
    # cv = cross_val_score(pipeline, df_train, survived, cv=5, scoring='roc_auc')
    # print(cv.mean())

    print(pipeline)
    X = pipeline.fit_transform(df_train)
    print(X)
    print(X.dtypes)
    print(X.shape)
    """
    pass


if __name__ == "__main__":
    main()
    pass
