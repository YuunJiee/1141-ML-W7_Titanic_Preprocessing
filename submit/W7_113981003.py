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
    df['Age'].fillna(df['Age'].median(), inplace=True)
    # TODO 2.2: 以 Embarked 眾數填補 (mode() 會返回 Series，取 [0] 取得第一個值)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    return df


# 任務 3：移除異常值
def remove_outliers(df):
    # TODO 3.1: 計算 Fare 平均與標準差
    fare_mean = df['Fare'].mean()
    fare_std = df['Fare'].std()
    # TODO 3.2: 移除 Fare > mean + 3*std 的資料
    df = df[df['Fare'] <= (fare_mean + 3 * fare_std)]
    return df


# 任務 4：類別變數編碼
def encode_features(df):
    # TODO 4.1: 使用 pd.get_dummies 對 Sex、Embarked 進行編碼
    # drop_first=True 可避免共線性問題
    df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df_encoded


# 任務 5：數值標準化
def scale_features(df):
    # TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
    scaler = StandardScaler()
    # 將要標準化的欄位選出
    columns_to_scale = ['Age', 'Fare']
    # 對選定的欄位進行擬合與轉換，並將結果存回原 DataFrame
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    df_scaled = df
    return df_scaled


# 任務 6：資料切割
def split_data(df):
    # 為了模型訓練，先移除不必要的文字欄位
    df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Passengerid'])
    
    # TODO 6.1: 將 Survived 作為 y，其餘為 X
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# 任務 7：輸出結果
def save_data(df, output_path):
    # TODO 7.1: 將清理後資料輸出為 CSV (encoding='utf-8-sig')
    # index=False 避免將 DataFrame 的索引寫入 CSV 檔案中
    df.to_csv(output_path, encoding='utf-8-sig', index=False)


# 主程式流程（請勿修改）
if __name__ == "__main__":
    # 假設您的資料檔位於與 .py 檔同層的 data 資料夾中
    input_path = "data/titanic.csv"
    output_path = "data/titanic_processed.csv"

    # 建立範例資料以供測試
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
    try:
        # 嘗試從網路下載 titanic.csv
        sample_df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        sample_df.to_csv(input_path, index=False)
    except Exception as e:
        print(f"無法下載範例資料，請確認您的'data/titanic.csv'檔案已存在。錯誤: {e}")
        # 如果下載失敗，您需要手動將 titanic.csv 放入 data 資料夾中
    

    df, missing_count = load_data(input_path)
    print(f"原始資料缺失值總數: {missing_count}")
    df = handle_missing(df)
    print("缺失值處理完成")
    df = remove_outliers(df)
    print("異常值移除完成")
    df_encoded = encode_features(df)
    print("類別變數編碼完成")
    df_scaled = scale_features(df_encoded)
    print("數值標準化完成")
    X_train, X_test, y_train, y_test = split_data(df_scaled)
    print(f"資料切割完成，訓練集大小: {X_train.shape}, 測試集大小: {X_test.shape}")
    save_data(df_scaled, output_path)
    print(f"處理後資料已儲存至 {output_path}")

    print("\nTitanic 資料前處理完成")