import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_and_save():
    df = pd.read_excel("data_training_tin_dung.xlsx", sheet_name="customer")
    
    # Mã hóa cột text
    le = LabelEncoder()
    df["xi10_nghe_nghiep"] = le.fit_transform(df["xi10_nghe_nghiep"].astype(str))
    
    feature_cols = [c for c in df.columns if c not in ["id", "label"]]
    X = df[feature_cols].fillna(0)
    y = df["label"]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    with open("model.pkl", "wb") as f:
        pickle.dump((model, feature_cols, le), f)
    print("✅ Đã huấn luyện và lưu model.pkl")

if __name__ == "__main__":
    train_and_save()