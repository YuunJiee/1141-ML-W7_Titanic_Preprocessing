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
    df = pd.read_csv(file_path)
    # TODO 1.2: 統一欄位首字母大寫，並計算缺失值數量
    df.columns = [c.capitalize() for c in df.columns]
    missing_count = df.isnull().sum().sum()
    return df, int(missing_count)


# 任務 2：處理缺失值
def handle_missing(df):
    # TODO 2.1: 以 Age 中位數填補
    df["Age"] = df["Age"].fillna(df["Age"].median())
    # TODO 2.2: 以 Embarked 眾數填補
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    return df


# 任務 3：移除異常值
def remove_outliers(df):
    # TODO 3.1: 計算 Fare 平均與標準差
    fare_mean = df["Fare"].mean()
    fare_std = df["Fare"].std()

    # TODO 3.2: 移除 Fare > mean + 3*std 的資料
    upper_bound = fare_mean + 3 * fare_std

    # 確保正確過濾並返回副本
    df_filtered = df[df["Fare"] <= upper_bound].reset_index(drop=True).copy()

    return df_filtered


# 任務 4：類別變數編碼
def encode_features(df):
    # TODO 4.1: 使用 pd.get_dummies 對 Sex、Embarked 進行編碼
    df_encoded = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=False)
    return df_encoded


# 任務 5：數值標準化
def scale_features(df):
    # TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[["Age", "Fare"]] = scaler.fit_transform(df[["Age", "Fare"]])
    return df_scaled


# 任務 6：資料切割
def split_data(df):
    # TODO 6.1: 將 Survived 作為 y，其餘為 X
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    # TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# 任務 7：輸出結果
def save_data(df, output_path):
    # TODO 7.1: 將清理後資料輸出為 CSV (encoding='utf-8-sig')
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


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
