import pandas as pd
import numpy as np
import sqlite3
import joblib
import datetime
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
    # Drop the unused columns
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    
    # Manage Target
    y = None
    if 'Attrition' in df_train.columns:
        # แปลง Yes/No เป็น 1/0 (ถ้ามี)
        y = df_train['Attrition'].map({'Yes': 1, 'No': 0})
        X = df_train.drop(columns=['Attrition'])
    else:
        X = df_train

    # Encoding
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    # Scaling
    scaler = StandardScaler()
    if not X.select_dtypes(include=['number']).empty:
        X_scaled = scaler.fit_transform(X.select_dtypes(include=['number']))
    else:
        X_scaled = X
    
    # 3. Train Model (Retrain)
    if y is not None:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        joblib.dump(model, MODEL_FILE)
    else:
        if os.path.exists(MODEL_FILE):
            model = joblib.load(MODEL_FILE)
        else:
            return False

    # 4. Predict & Save
    probs = model.predict_proba(X_scaled)[:, 1]
    df['employee_resignation_probability'] = probs
    
    # Save for Dashboard
    df.to_csv(DASHBOARD_DATA, index=False)
    
    # Update Timestamp
    os.utime(DASHBOARD_DATA, None)
    
    print(f"✅ Dashboard Data Updated Successfully!")
    return True

# --- The New Functions (based on the Code you requested) ---
def regenerate_database(num_rows=4000):
    """Create simulation data follow number that define and replace original database"""
    print(f"🚀 Generating {num_rows} mock employees...")
    
    if not os.path.exists(SOURCE_CSV):
        print(f"❌ Error: File '{SOURCE_CSV}' not found.")
        return False
        
    df_orig = pd.read_csv(SOURCE_CSV)
    new_data = {}
    
    # Loop to create data column by column
    for col in df_orig.columns:
        # Skip unnecessary columns
        if col in ['EmployeeNumber', 'Attrition']:
            continue
            
        # Generate random data
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

    # Create DataFrame
    df_new = pd.DataFrame(new_data)
    
    # Generate new ID
    df_new['EmployeeNumber'] = np.arange(10000, 10000 + num_rows)
    
    # บันทึก Save CSV (Option)
    csv_filename = f"generated_employees_test_{num_rows}.csv"
    df_new.to_csv(csv_filename, index=False)
    
    # Update Database
    print("⚙️ Updating Database...")
    conn = sqlite3.connect(DB_FILE)
    df_new.to_sql('current_employees', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"✅ Database Replaced with {num_rows} records.")
    
    # Call run_pipeline to immediately update the Dashboard
    return run_pipeline()

if __name__ == "__main__":
    run_pipeline()