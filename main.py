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
    # Preprocess(df_path=train_path)
    model = ModelTitanic(df_train_path=consts.TRAIN_PATH)
    model.train()
    pass


if __name__ == "__main__":
    main()
    pass
