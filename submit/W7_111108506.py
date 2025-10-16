# -*- coding: utf-8 -*-
# W6 Titanic Preprocessing Template
# 僅可修改 TODO 區塊，其餘部分請勿更動

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 任務 1：載入資料
def load_data(file_path):
    # TODO 1.1: 讀取 CSV
    # TODO 1.2: 統一欄位首字母大寫，並計算缺失值數量
    df = pd.read_csv("data/titanic.csv")
    df.columns = [c.capitalize() for c in df.columns]
    missing_count = df.isna().sum().sum()
    return df, int(missing_count)


# 任務 2：處理缺失值
def handle_missing(df):
    # TODO 2.1: 以 Age 中位數填補
    # TODO 2.2: 以 Embarked 眾數填補
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df


# 任務 3：移除異常值
def remove_outliers(df):
    # TODO 3.1: 計算 Fare 平均與標準差
    # TODO 3.2: 移除 Fare > mean + 3*std 的資料
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    fare_mean = df["Fare"].mean()
    fare_std = df["Fare"].std()
    df = df[df["Fare"] <= fare_mean + 3*fare_std]
    df = df[df["Fare"] >= 0]  # 移除負數（如果有）
    df.reset_index(drop=True, inplace=True)
    return df


# 任務 4：類別變數編碼
def encode_features(df):
    # TODO 4.1: 使用 pd.get_dummies 對 Sex、Embarked 進行編碼
    sex_dummies = pd.get_dummies(df["Sex"], prefix="Sex")
    embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked")
    df_encoded = pd.concat([df.drop(columns=["Sex", "Embarked"]), sex_dummies, embarked_dummies], axis=1)

    return df_encoded


# 任務 5：數值標準化
def scale_features(df):
    # TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled["Age"] = scaler.fit_transform(df[["Age"]])
    df_scaled["Fare"] = scaler.fit_transform(df[["Fare"]])
    return df_scaled


# 任務 6：資料切割
def split_data(df):
    # TODO 6.1: 將 Survived 作為 y，其餘為 X
    # TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
    X = df.drop("Survived", axis=1)
    Y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
    return X_train, X_test, y_train, y_test


# 任務 7：輸出結果
def save_data(df, output_path):
    # TODO 7.1: 將清理後資料輸出為 CSV (encoding='utf-8-sig')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    pass


# 主程式流程（請勿修改）
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