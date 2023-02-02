from typing import List


class TrainTestData:
    # data paths
    TITANIC_PATH: str = 'C:/Users/nivm2/PycharmProjects/Titanic/data/titanic.csv'
    TRAIN_PATH: str = 'C:/Users/nivm2/PycharmProjects/Titanic/data/train.csv'
    TEST_PATH: str = 'C:/Users/nivm2/PycharmProjects/Titanic/data/test.csv'
    TEST_SIZE: float = 0.2
    RANDOM_STATE_SPLIT: int = 42
    # path for preprocessed (feature engineering and feature selection) train
    TRAIN_PREPROCESSED_PATH: str = 'C:/Users/nivm2/PycharmProjects/Titanic/data/train_preprocessed.csv'


class ModelsPerformances:
    # models performances experiments log path
    MODELS_PERFORMANCES_PATH: str = 'C:/Users/nivm2/PycharmProjects/Titanic/results/models_performances.csv'
    SELECTED_MODEL_PIPE_PATH: str = 'C:/Users/nivm2/PycharmProjects/Titanic/results/model_selection_pipeline.pkl'


class PreprocessUtilsConsts:
    MISSINGNESS_PCT: float = 0.005
    HONORIFICS_LIST: List[str] = [
        "Mrs", "Mr", "Ms", "Miss", "Master", "Don", "Rev", "Mme", "Major", "Mlle", "Col", "Capt", "Jonkheer",
        "Countess", "Dr",
    ]
    CABIN_NONE_CONST: str = 'X'
