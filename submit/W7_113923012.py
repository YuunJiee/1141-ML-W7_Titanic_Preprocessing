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
    df = pd.read_csv(file_path)
    df.columns = [c.capitalize() for c in df.columns]
    # 計算所有欄位的缺失值總數
    missing_count = df.isnull().sum().sum()
    return df, int(missing_count)


# 任務 2：處理缺失值
def handle_missing(df):
    # TODO 2.1: 以 Age 中位數填補
    age_median = df['Age'].median()
    df['Age'].fillna(age_median, inplace=True)
    
    # TODO 2.2: 以 Embarked 眾數填補
    # mode()[0] 取得眾數（可能有多個，取第一個）
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'].fillna(embarked_mode, inplace=True)
    
    return df


# 任務 3：移除異常值
def remove_outliers(df):
    # TODO 3.1: 計算 Fare 平均與標準差
    fare_mean = df['Fare'].mean()
    fare_std = df['Fare'].std()
    
    # TODO 3.2: 移除 Fare > mean + 3*std 的資料
    threshold = fare_mean + 3 * fare_std
    df = df[df['Fare'] <= threshold].copy() # 使用 .copy() 避免 SettingWithCopyWarning
    
    return df


# 任務 4：類別變數編碼
def encode_features(df):
    # TODO 4.1: 使用 pd.get_dummies 對 Sex、Embarked 進行編碼
    # drop_first=True 避免共線性問題
    df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # 移除 'Name', 'Ticket', 'Cabin', 'Passengerid' 這些通常不直接用於模型的欄位
    columns_to_drop = ['Name', 'Ticket', 'Cabin', 'Passengerid']
    # 僅在欄位存在時才刪除
    columns_to_drop = [col for col in columns_to_drop if col in df_encoded.columns]
    df_encoded = df_encoded.drop(columns=columns_to_drop)
    
    return df_encoded


# 任務 5：數值標準化
def scale_features(df):
    # TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
    scaler = StandardScaler()
    
    # 選取要標準化的數值欄位
    numerical_features = ['Age', 'Fare']
    
    # 確保這些欄位存在於 DataFrame 中
    existing_numerical_features = [col for col in numerical_features if col in df.columns]

    if existing_numerical_features:
        # 進行標準化
        df[existing_numerical_features] = scaler.fit_transform(df[existing_numerical_features])
    
    df_scaled = df
    return df_scaled


# 任務 6：資料切割
def split_data(df):
    # TODO 6.1: 將 Survived 作為 y，其餘為 X
    if 'Survived' in df.columns:
        y = df['Survived']
        X = df.drop('Survived', axis=1)
    else:
        # 如果 'Survived' 不存在 (例如，用於測試集，雖然本例是訓練集)
        print("Warning: 'Survived' column not found for splitting.")
        return None, None, None, None
        
    # TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


# 任務 7：輸出結果
def save_data(df, output_path):
    # TODO 7.1: 將清理後資料輸出為 CSV (encoding='utf-8-sig')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')


# 主程式流程（請勿修改）
if __name__ == "__main__":
    # 假設 'data/titanic.csv' 檔案存在於正確路徑
    input_path = "data/titanic.csv"
    output_path = "data/titanic_processed.csv"

    # 確保資料夾存在
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
        # 注意：如果 'data/titanic.csv' 不存在，程式將會報錯

    try:
        df, missing_count = load_data(input_path)
        print(f"原始資料載入，總缺失值數量: {missing_count}")
        
        df = handle_missing(df)
        print("缺失值處理完成 (Age: 中位數, Embarked: 眾數)")
        
        original_rows = len(df)
        df = remove_outliers(df)
        print(f"異常值移除完成 (Fare > mean + 3*std), 移除 {original_rows - len(df)} 筆資料")
        
        df = encode_features(df)
        print("類別變數編碼完成 (Sex, Embarked)")
        
        df = scale_features(df)
        print("數值標準化完成 (Age, Fare)")
        
        X_train, X_test, y_train, y_test = split_data(df)
        
        if X_train is not None:
            print(f"資料切割完成: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
            save_data(df, output_path)
            print(f"清理後資料已輸出至: {output_path}")

        print("Titanic 資料前處理完成")

    except FileNotFoundError:
        print(f"錯誤: 找不到輸入檔案 {input_path}，請確認路徑是否正確。")
    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")