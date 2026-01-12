import pandas as pd
import numpy as np
import sqlite3
import joblib
import datetime
import os
from catboost import CatBoostClassifier, Pool

# Config
DB_FILE = 'hr_database.db'
MODEL_FILE = 'best_attrition_model.pkl'
DASHBOARD_DATA = 'dashboard_data.csv'
SOURCE_CSV = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'

def run_pipeline():
    print(f"⚙️ Pipeline Triggered at {datetime.datetime.now()}...")
    
    if not os.path.exists(DB_FILE):
        print("❌ Database not found.")
        return False

    # 1. Connect DB
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql("SELECT * FROM current_employees", conn)
    except Exception as e:
        print(f"❌ DB Error: {e}")
        conn.close()
        return False
    conn.close()
    
    print(f"   -> Fetched {len(df)} rows from Database.")
    
    # 2. Preprocessing
    df_train = df.copy()
    
    # Drop unused columns
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    
    # Manage Target
    y = None
    if 'Attrition' in df_train.columns:
        # Convert Yes/No to 1/0
        y = df_train['Attrition'].map({'Yes': 1, 'No': 0})
        X = df_train.drop(columns=['Attrition'])
    else:
        X = df_train

    # Identify Categorical Features (สำหรับ CatBoost)
    cat_features = [col for col in X.columns if X[col].dtype == 'object']
    
    # Fill Missing Values
    for col in cat_features:
        X[col] = X[col].fillna('Missing').astype(str)
        
    # 3. Train Model (Logic: ถ้ามี Target ให้เทรนใหม่เป็น CatBoost, ถ้าไม่มีให้โหลดตัวเก่า)
    if y is not None:
        print("🚀 Training New CatBoost Model...")
        model = CatBoostClassifier(
            iterations=500, 
            learning_rate=0.1, 
            depth=6, 
            loss_function='Logloss',
            verbose=0
        )
        model.fit(X, y, cat_features=cat_features)
        joblib.dump(model, MODEL_FILE)
        print("✅ Model Retrained and Saved.")
    else:
        if os.path.exists(MODEL_FILE):
            print("⚠️ No target found. Loading existing model...")
            model = joblib.load(MODEL_FILE)
        else:
            print("❌ No model and no target to train. Exiting.")
            return False

    # 4. Predict & Save for Dashboard (HYBRID FIX)
    # ส่วนนี้แก้ให้รองรับทั้งโมเดลเก่า (Sklearn) และโมเดลใหม่ (CatBoost)
    try:
        # Case A: ถ้าเป็นโมเดล CatBoost (ของใหม่)
        if isinstance(model, CatBoostClassifier):
            # print("🔹 Detected CatBoost Model. Using Pool...")
            prediction_pool = Pool(data=X, cat_features=cat_features)
            probs = model.predict_proba(prediction_pool)[:, 1]

        # Case B: ถ้าเป็นโมเดลเก่า (Sklearn / Logistic Regression)
        else:
            print("🔸 Detected Legacy Model (Sklearn). Adapting data...")
            
            # 1. จำลองการทำ One-Hot Encoding
            X_legacy = pd.get_dummies(X)
            
            # 2. หาชื่อ Features ที่โมเดลเก่าต้องการ
            model_features = None
            if hasattr(model, 'feature_names_in_'):
                model_features = model.feature_names_in_
            elif hasattr(model, 'get_feature_names_out'):
                model_features = model.get_feature_names_out()
                
            # 3. บังคับให้คอลัมน์ตรงกับโมเดลเก่า (Reindex)
            # ถ้าขาดคอลัมน์ไหน (เช่น EducationField_3) จะเติม 0 ให้
            if model_features is not None:
                X_legacy = X_legacy.reindex(columns=model_features, fill_value=0)
            
            # 4. ทำนายผลแบบปกติ
            probs = model.predict_proba(X_legacy)[:, 1]

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return False

    # Save to CSV
    df['employee_resignation_probability'] = probs
    df.to_csv(DASHBOARD_DATA, index=False)
    os.utime(DASHBOARD_DATA, None)
    
    print(f"✅ Dashboard Data Updated Successfully!")
    return True

def regenerate_database(num_rows=4000):
    """Create simulation data and replace original database"""
    print(f"🚀 Generating {num_rows} mock employees...")
    
    if not os.path.exists(SOURCE_CSV):
        print(f"❌ Error: File '{SOURCE_CSV}' not found.")
        return False
        
    df_orig = pd.read_csv(SOURCE_CSV)
    new_data = {}
    
    # Loop to create data
    for col in df_orig.columns:
        if col in ['EmployeeNumber']: 
            continue
            
        if df_orig[col].dtype == 'object':
            unique_vals = df_orig[col].unique()
            new_data[col] = np.random.choice(unique_vals, num_rows)
        elif df_orig[col].nunique() < 15:
            unique_vals = df_orig[col].unique()
            new_data[col] = np.random.choice(unique_vals, num_rows)
        else:
            min_val = df_orig[col].min()
            max_val = df_orig[col].max()
            new_data[col] = np.random.randint(min_val, max_val + 1, num_rows)

    df_new = pd.DataFrame(new_data)
    df_new['EmployeeNumber'] = np.arange(10000, 10000 + num_rows)
    
    # Update Database
    print("⚙️ Updating Database...")
    conn = sqlite3.connect(DB_FILE)
    df_new.to_sql('current_employees', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"✅ Database Replaced with {num_rows} records.")
    
    # Trigger Pipeline immediately
    return run_pipeline()

if __name__ == "__main__":
    run_pipeline()