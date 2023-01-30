import pandas as pd
from sklearn.model_selection import train_test_split
import os
from src.titanic_classifier.ModelTitanic import ModelTitanic, train_path, test_path


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
    main()
    pass
