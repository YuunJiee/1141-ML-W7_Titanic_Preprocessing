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
    """
    移除 Fare > mean + 3*std 的資料，採用迭代方式直到條件滿足。
    注意：此函式會先用中位數填補 Fare 的缺值，然後執行迭代移除。
    """
    # 先填補缺值（避免計算 mean/std 時被 NaN 干擾）
    if "Fare" not in df.columns:
        return df  # 若沒有該欄位，直接回傳原 df（由呼叫端決定）
    df = df.copy()
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # 迭代移除：每輪以當前 df 的 mean/std 決定閾值，移除超過閾值的資料
    while True:
        if len(df) == 0:
            break

        mean = df["Fare"].mean()
        std = df["Fare"].std()

        # 若 std 為 NaN（例如全部值相同或只有一筆），直接跳出
        if pd.isna(std) or std == 0:
            break

        threshold = mean + 3 * std

        # 若目前最大值已在閾值內，表示條件滿足，結束
        if df["Fare"].max() <= threshold:
            break

        # 否則移除超過閾值的列（一次性移除全部超過者）
        df = df[df["Fare"] <= threshold]

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