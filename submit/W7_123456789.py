# -*- coding: utf-8 -*-
# W7 Titanic Preprocessing Solution (for testing only)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 任務 1：載入資料
def load_data(file_path):
    df = pd.read_csv(file_path)
    # 統一欄位命名（首字母大寫）
    df.columns = [c.capitalize() for c in df.columns]
    missing_count = df.isnull().sum().sum()
    return df, int(missing_count)

# 任務 2：處理缺失值
def handle_missing(df):
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df

# 任務 3：移除異常值（修正版）
def remove_outliers(df):
    # 重新計算直到無異常值
    while True:
        mean = df["Fare"].mean()
        std = df["Fare"].std()
        upper = mean + 3 * std
        new_df = df[df["Fare"] <= upper]
        if len(new_df) == len(df):
            break
        df = new_df
    return df

# 任務 4：類別變數編碼
def encode_features(df):
    df_encoded = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=False)
    return df_encoded

# 任務 5：數值標準化
def scale_features(df):
    scaler = StandardScaler()
    df[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
    return df

# 任務 6：資料切割
def split_data(df):
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

# 任務 7：輸出結果
def save_data(df, output_path):
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    input_path = "data/titanic.csv"
    output_path = "data/titanic_processed.csv"

    df, missing_count = load_data(input_path)
    df = handle_missing(df)
    df = remove_outliers(df)
    df = encode_features(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_data(df, output_path)

    print("Titanic 資料前處理完成")
