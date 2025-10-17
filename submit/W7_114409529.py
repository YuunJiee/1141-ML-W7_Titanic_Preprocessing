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
    df = pd.read_csv('data/titanic.csv')
    df.columns = [c.capitalize() for c in df.columns]
    missing_count = df.isnull().sum().sum()
    return df, int(missing_count)


# 任務 2：處理缺失值
def handle_missing(df):
    # TODO 2.1: 以 Age 中位數填補
    df['Age']=df['Age'].fillna(df['Age'].median())
    # TODO 2.2: 以 Embarked 眾數填補
    df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
    return df


# 任務 3：移除異常值

def remove_outliers(df):
    counter=-2
    while counter!=len(df):#其目標是永遠不要有Fare > mean + 3*std的情形，如果刪除資料導致std和mean更改則只能再做一次測試
        counter=len(df)
    # TODO 3.1: 計算 Fare 平均與標準差
        desc=df['Fare'].describe()
        std=desc['std']
        mean=desc['mean']
    # TODO 3.2: 移除 Fare > mean + 3*std 的資料
        up=mean+3*std
        Outer_df_Fare=df[df['Fare']>up] #哪些資料超出範圍
        df=df.drop(index=Outer_df_Fare.index)

    return df


# 任務 4：類別變數編碼
def encode_features(df):
    # TODO 4.1: 使用 pd.get_dummies 對 Sex、Embarked 進行編碼
    df_encoded=pd.get_dummies(df, columns=['Sex','Embarked'])

    return df_encoded


# 任務 5：數值標準化
def scale_features(df):
    # TODO 5.1: 使用 StandardScaler 標準化 Age、Fare
    scaler = StandardScaler()
    df.loc[:,['Age','Fare']]=scaler.fit_transform(df[['Age','Fare']])
    df_scaled = df#我能動的東西有限，我只能硬去利用變數賦值對接
    
    return df_scaled#df_scaled是要求的輸出


# 任務 6：資料切割
def split_data(df):
    # TODO 6.1: 將 Survived 作為 y，其餘為 X
    y=df['Survived']
    X=df.drop('Survived',axis=1)
    # TODO 6.2: 使用 train_test_split 切割 (test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# 任務 7：輸出結果
def save_data(df, output_path):
    # TODO 7.1: 將清理後資料輸出為 CSV (encoding='utf-8-sig')
    df.to_csv(output_path)


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

