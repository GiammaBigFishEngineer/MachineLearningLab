
import random
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import torch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

SEED = 42

def main():
    df = pd.read_csv("/Users/gianmariadifronzo/Desktop/machine_learning/lab05/train_dataset.csv")
    print(df.head())
    print(df.shape)
    print(df.describe())
    print(df.isnull().sum())
    y = df["target"]
    X = df.drop(columns=['target'])
    
    cont_col = [col for col in X.columns if col.startswith("cont_")]
    cat_col = [col for col in X.columns if col.startswith("cat_")]
    ord_col = [col for col in X.columns if col.startswith("ord_")]

    processor = ColumnTransformer(
        transformers=[
            ('cont', RobustScaler(), cont_col),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_col),
            ('ord', OrdinalEncoder(), ord_col)
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("processor", processor),
        ('imputer', SimpleImputer(strategy='median')),
        #('poly', PolynomialFeatures(degree=2, include_bias=False)),
        #('ridge', Ridge(alpha=0.5)),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=SEED))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model.fit(X_train,y_train)
    y_test_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    print("MAE:", mae)
    print("R2:", r2)

    plt.scatter(X["cont_12"], y)
    plt.xlabel("cont_12")
    plt.ylabel("target")
    plt.title("Relazione tra cont_5 e target")
    plt.show()

if __name__ == "__main__":
    main()