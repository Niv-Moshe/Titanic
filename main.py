import pandas as pd


if __name__ == "__main__":
    train_titanic = pd.read_csv('data/train.csv')
    print(train_titanic)
    print(list(train_titanic.columns))
    pass
