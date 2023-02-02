import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.consts import TrainTestData
from src.titanic_classifier.ModelTitanic import ModelTitanic


def create_train_test_files():
    if not (os.path.exists(TrainTestData.TRAIN_PATH) and os.path.exists(TrainTestData.TEST_PATH)):
        print('Creating Train and Test files')
        # create train test split
        titanic = pd.read_csv(TrainTestData.TITANIC_PATH)
        train, test = train_test_split(titanic, test_size=TrainTestData.TEST_SIZE,
                                       random_state=TrainTestData.RANDOM_STATE_SPLIT)
        train.to_csv(TrainTestData.TRAIN_PATH, index=False)
        test.to_csv(TrainTestData.TEST_PATH, index=False)
    else:
        print('Train and Test files already exist')


def main():
    create_train_test_files()
    df_train = pd.read_csv(TrainTestData.TRAIN_PATH)
    model = ModelTitanic(df_train=df_train)
    model.perform_model_selection()


if __name__ == "__main__":
    main()
