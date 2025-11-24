import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- 🎨 CSS STYLING ---
st.markdown("""
<style>
    /* 1. Metrics Box */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    /* 2. Container Border Styling (Card) */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        background-color: white;
    }

    /* 3. Sidebar Header */
    .css-164nlkn {
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Cover Image
st.image(
    #"https://images.unsplash.com/photo-1556761175-5973dc0f32e7?q=80&w=2832&auto=format&fit=crop",
    '/Users/barabank/Capstone project/HR.png',
    use_container_width=True,
)

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# 2. Data Loading Function
# ==========================================
@st.cache_data
def load_data():
    file_path = 'train_with_resignation_probabilities.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        return None

df = load_data()

# ==========================================
# 🔄 PROCESSING: Decode & Unscale Data
# ==========================================
if df is not None:
    # --- 1. Helper Function ---
    def get_label(val, mapping_dict):
        if val is None: return val
        try:
            if pd.isna(val): return val
        except:
            pass
        try:
            closest_key = min(mapping_dict.keys(), key=lambda k: abs(k - float(val)))
            return mapping_dict[closest_key]
        except:
            return val

    # --- 2. Decode Categorical Variables ---
    maps = {
        'Department': {-2.8450: 'Human Resources', -1.0767: 'Sales', 0.6917: 'Research & Development'},
        'JobRole': {
            -1.0897: 'Sales Executive', -0.6575: 'Research Scientist', -0.2253: 'Laboratory Technician',
            0.2069: 'Manufacturing Director', 0.6391: 'Healthcare Representative', 1.0713: 'Manager',
            1.5035: 'Sales Representative', 1.9357: 'Research Director', 2.3679: 'Human Resources'
        },
        'MaritalStatus': {-1.5599: 'Divorced', -0.2974: 'Single', 0.9652: 'Married'},
        'BusinessTravel': {-2.0236: 'Non-Travel', -0.1615: 'Travel_Rarely', 1.7006: 'Travel_Frequently'},
        'Gender': {0.0: 'Female', 1.0: 'Male'},
        'OverTime': {0.0: 'No', 1.0: 'Yes'},
        'Attrition': {0.0: 'No', 1.0: 'Yes'},
        'JobSatisfaction': {-1.5488: 'Low', -0.6480: 'Medium', 0.2528: 'High', 1.1535: 'Very High'},
        'EnvironmentSatisfaction': {-1.5776: 'Low', -0.6587: 'Medium', 0.2602: 'High', 1.1791: 'Very High'},
        'RelationshipSatisfaction': {-1.6002: 'Low', -0.6800: 'Medium', 0.2402: 'High', 1.1604: 'Very High'},
        'WorkLifeBalance': {-2.4486: 'Bad', -1.0555: 'Good', 0.3376: 'Better', 1.7308: 'Best'},
        'Age': { 
            -2.0708: 18, -1.9618: 19, -1.8528: 20, -1.7438: 21, -1.6348: 22, -1.5258: 23, -1.4168: 24, 
            -1.3078: 25, -1.1988: 26, -1.0898: 27, -0.9808: 28, -0.8718: 29, -0.7628: 30, -0.6538: 31, 
            -0.5448: 32, -0.4358: 33, -0.3268: 34, -0.2178: 35, -0.1088: 36, 0.0002: 37, 0.1092: 38, 
            0.2182: 39, 0.3272: 40, 0.4362: 41, 0.5452: 42, 0.6542: 43, 0.7632: 44, 0.8722: 45, 
            0.9812: 46, 1.0902: 47, 1.1992: 48, 1.3082: 49, 1.4172: 50, 1.5262: 51, 1.6352: 52, 
            1.7442: 53, 1.8532: 54, 1.9622: 55, 2.0712: 56, 2.1802: 57, 2.2892: 58, 2.3982: 59, 2.5072: 60
        },
        'YearsAtCompany': {
            -1.1588: 0, -0.9944: 1, -0.8301: 2, -0.6657: 3, -0.5013: 4, -0.3370: 5, -0.1726: 6, -0.0082: 7,
            0.1561: 8, 0.3205: 9, 0.4848: 10, 0.6492: 11, 0.8136: 12, 0.9779: 13, 1.1423: 14, 1.3067: 15,
            1.4710: 16, 1.6354: 17, 1.7998: 18, 1.9641: 19, 2.1285: 20, 2.2929: 21, 2.4572: 22, 2.6216: 23,
            2.7860: 24, 2.9503: 25, 3.1147: 26, 3.2791: 27, 3.6078: 29, 3.7721: 30, 3.9365: 31, 4.1009: 32,
            4.2652: 33, 4.4296: 34, 4.7583: 36, 4.9227: 37
        }
    }

    for col, mapping in maps.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: get_label(x, mapping))

    if 'EducationField' in df.columns:
        edu_map = {
            '5': 'Life Sciences', '5.0': 'Life Sciences', '4': 'Medical', '4.0': 'Medical',
            '3': 'Marketing', '3.0': 'Marketing', '2': 'Technical Degree', '2.0': 'Technical Degree',
            '0': 'Other', '0.0': 'Other', 'Human Resources': 'Human Resources'
        }
        df['EducationField'] = df['EducationField'].astype(str).map(edu_map).fillna(df['EducationField'])

    # --- 3. Unscale Numerical Variables ---
    unscale_params = {
        'MonthlyIncome': {'mean': 6544.02, 'std': 4651.76},
        'DistanceFromHome': {'mean': 9.36, 'std': 8.18},
        'DailyRate': {'mean': 803.99, 'std': 401.17},
        'Education': {'mean': 2.91, 'std': 1.03},
        'JobLevel': {'mean': 2.08, 'std': 1.09},
        'StockOptionLevel': {'mean': 0.79, 'std': 0.85},
        'NumCompaniesWorked': {'mean': 2.69, 'std': 2.49},
        'PercentSalaryHike': {'mean': 15.24, 'std': 3.68},
        'YearsSinceLastPromotion': {'mean': 2.18, 'std': 3.21},
        'JobInvolvement': {'mean': 2.74, 'std': 0.70},
        'HourlyRate': {'mean': 65.50, 'std': 20.36},
        'MonthlyRate': {'mean': 14390.24, 'std': 7189.78},
        'PerformanceRating': {'mean': 3.16, 'std': 0.36},
        'TotalWorkingYears': {'mean': 11.36, 'std': 7.80},
        'TrainingTimesLastYear': {'mean': 2.76, 'std': 1.26},
        'YearsInCurrentRole': {'mean': 4.23, 'std': 3.57},
        'YearsWithCurrManager': {'mean': 4.20, 'std': 3.56}
    }

    for col, params in unscale_params.items():
        if col in df.columns:
            real_col_name = f"{col}_Real"
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[real_col_name] = (df[col] * params['std']) + params['mean']
            
            if col in ['Education', 'JobLevel', 'StockOptionLevel', 'NumCompaniesWorked', 
                       'PercentSalaryHike', 'PerformanceRating', 'TotalWorkingYears', 
                       'TrainingTimesLastYear', 'YearsInCurrentRole', 'YearsWithCurrManager', 
                       'YearsSinceLastPromotion', 'HourlyRate', 'MonthlyRate', 'DailyRate']:
                df[real_col_name] = df[real_col_name].round(0).astype(int)

# ==========================================
# 3. Sidebar: Filters & Controls
# ==========================================
st.sidebar.header("⚙️ Dashboard Filters")

if df is not None:
    st.sidebar.subheader("1. Filter Data")
    all_departments = df['Department'].unique()
    selected_dept = st.sidebar.multiselect("Select Department", options=all_departments, default=all_departments)

    if 'JobRole' in df.columns:
        all_roles = df['JobRole'].unique()
        selected_role = st.sidebar.multiselect("Select Job Role", options=all_roles, default=all_roles)
    else:
        selected_role = []

    st.sidebar.subheader("2. Risk Settings")
    risk_threshold = st.sidebar.slider("High Risk Threshold", 0.0, 1.0, 0.5, 0.05)
    
    st.sidebar.subheader("3. Display Settings")
    top_n_employees = st.sidebar.number_input("Number of Top Employees to Show", min_value=1, max_value=100, value=5, step=1)
    
    # Select Employee Details
    exclude_keywords = ['_scaled', 'EducationField_', 'EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    exclude_exact = [
        'Department', 'JobRole', 'employee_resignation_probability', 'Attrition',
        'MonthlyIncome', 'DistanceFromHome', 'DailyRate', 'Education', 'JobLevel', 'StockOptionLevel',
        'NumCompaniesWorked', 'PercentSalaryHike', 'YearsSinceLastPromotion', 'JobInvolvement',
        'HourlyRate', 'MonthlyRate', 'PerformanceRating', 'TotalWorkingYears', 'TrainingTimesLastYear',
        'YearsInCurrentRole', 'YearsWithCurrManager',
        'Education_Real', 'StockOptionLevel_Real', 'JobInvolvement_Real'
    ]
    
    available_cols = []
    for c in df.columns:
        if c in exclude_exact: continue
        if any(keyword in c for keyword in exclude_keywords): continue
        available_cols.append(c)
    
    available_cols = sorted(available_cols)
    default_cols = ['Age', 'JobSatisfaction', 'OverTime']
    for c in ['MonthlyIncome_Real', 'DistanceFromHome_Real', 'JobLevel_Real']:
        if c in available_cols: default_cols.append(c)
        
    selected_card_details = st.sidebar.multiselect("Select Details to Show on Cards", options=available_cols, default=default_cols)

    if selected_dept:
        if 'JobRole' in df.columns and selected_role:
             filtered_df = df[(df['Department'].isin(selected_dept)) & (df['JobRole'].isin(selected_role))]
        else:
            filtered_df = df[df['Department'].isin(selected_dept)]
    else:
        filtered_df = pd.DataFrame()
        st.sidebar.warning("Please select at least one Department.")
        
    st.sidebar.markdown("---")
    st.sidebar.info(f"Showing data for {len(filtered_df)} employees")

else:
    st.error("File 'train_with_resignation_probabilities.csv' not found.")
    st.stop()

# ==========================================
# 4. Main Dashboard Layout
# ==========================================

st.title("📊 Employee Turnover Prediction Dashboard")
st.markdown(f"**Overview:** Analyzing retention risks for **{len(filtered_df)}** employees. **Threshold:** >{risk_threshold*100:.0f}%")

if filtered_df.empty:
    st.warning("No data available for the current selection.")
    st.stop()

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📈 Risk Overview", "🔍 Deep Dive Analysis", "🤖 Predict New Data"])

# --- Tab 1: Risk Overview ---
with tab1:
    # 1. Metrics Row
    col_m1, col_m2, col_m3 = st.columns(3)
    avg_risk = filtered_df['employee_resignation_probability'].mean()
    high_risk_employees = filtered_df[filtered_df['employee_resignation_probability'] > risk_threshold]
    
    col_m1.metric("Average Probability", f"{avg_risk:.1%}")
    col_m2.metric(f"High Risk Employees", f"{len(high_risk_employees)} Persons")
    col_m3.metric("High Risk Percentage", f"{(len(high_risk_employees)/len(filtered_df)*100):.1f}%")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    
    # Left: Donut Chart
    with col1:
        st.subheader("Risk Category Breakdown (Employees by Risk Level)")
        def categorize_risk(prob, threshold):
            if prob < (threshold * 0.5): return 'Safe Zone'
            elif prob < threshold: return 'Watchlist'
            else: return 'High Risk'

        filtered_df['Risk_Category'] = filtered_df['employee_resignation_probability'].apply(lambda x: categorize_risk(x, risk_threshold))
        risk_counts = filtered_df['Risk_Category'].value_counts()
        color_map = {'Safe Zone': '#4CAF50', 'Watchlist': '#FFC107', 'High Risk': '#FF5252'}
        colors = [color_map.get(cat, '#9E9E9E') for cat in risk_counts.index]

        fig_donut, ax_donut = plt.subplots(figsize=(8, 5))
        wedges, texts, autotexts = ax_donut.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=90, colors=colors, pctdistance=0.85, wedgeprops=dict(width=0.3))
        plt.setp(texts, size=10, weight="bold")
        plt.setp(autotexts, size=10, weight="bold", color="white")
        st.pyplot(fig_donut)
    
    # Right: High Risk Cards (Vertical Layout inside Card)
    with col2:
        st.subheader(f"🚨 Top {top_n_employees} High Risk Employees")
        top_risk_display = filtered_df[filtered_df['employee_resignation_probability'] > risk_threshold]
        top_risk_display = top_risk_display.sort_values(by='employee_resignation_probability', ascending=False).head(top_n_employees)

        if top_risk_display.empty:
            st.success("No high risk employees found! 🎉")
        else:
            for index, row in top_risk_display.iterrows():
                # สร้างการ์ดด้วย st.container (มีขอบมนตาม CSS)
                with st.container(border=True):
                    # 1. แสดงรูปภาพตรงกลางด้วย HTML (แก้ไขใหม่)
                    st.markdown(f"""
                        <div style="display: flex; justify-content: center; margin-top: 10px; margin-bottom: 10px;">
                            <img src="https://i.pravatar.cc/150?u={index}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover;">
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # 2. ข้อมูลหลัก (ID, Risk, Role) จัดกึ่งกลาง
                    st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 1.2em;'>Employee ID: {index}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; color: #FF5252; font-weight: bold;'>Risk: {row['employee_resignation_probability']:.1%}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; color: gray; font-size: 0.9em; margin-bottom: 10px;'>{row.get('JobRole','-')} | {row.get('Department','-')}</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # 3. รายละเอียดข้อมูล (เรียงบรรทัดและจัดกึ่งกลาง)
                    if selected_card_details:
                        for col_name in selected_card_details:
                            val = row[col_name]
                            val_str = str(val)
                            
                            # Format numeric
                            if isinstance(val, (int, float)):
                                if 'Income' in col_name or 'Rate' in col_name: 
                                    val_str = f"${val:,.0f}"
                                elif 'Distance' in col_name: 
                                    val_str = f"{val:.0f} km" 
                                elif 'Percent' in col_name: 
                                    val_str = f"{val}%"
                            
                            label = col_name.replace('_Real','').replace('_',' ')
                            st.markdown(f"<div style='text-align: center; margin-bottom: 4px;'><b>{label}:</b> {val_str}</div>", unsafe_allow_html=True)
                    
                    # 4. เพิ่มช่องว่างด้านล่างสุด (Margin Bottom)
                    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# --- Tab 2: Deep Dive Analysis ---
with tab2:
    st.header("Key Drivers & HR Insights")
    viz_df = filtered_df.copy()
    
    col_driver, col_dept = st.columns(2)
    with col_driver:
        st.subheader("1. Key Drivers")
        numeric_df = df.select_dtypes(include=[np.number])
        if 'employee_resignation_probability' in numeric_df.columns:
            corr_target = numeric_df.corrwith(numeric_df['employee_resignation_probability']).sort_values(ascending=False)
            corr_target = corr_target.drop(['employee_resignation_probability', 'Attrition'], errors='ignore')
            corr_target = corr_target[~corr_target.index.str.contains('_Real')]
            
            top_drivers = pd.concat([corr_target.head(5), corr_target.tail(5)])
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.barplot(x=top_drivers.values, y=top_drivers.index, palette='coolwarm', ax=ax_corr)
            ax_corr.set_title('Correlation with Resignation Risk')
            st.pyplot(fig_corr)

    with col_dept:
        st.subheader("2. Risk by Department")
        if 'Department' in viz_df.columns:
            dept_risk = viz_df.groupby('Department')['employee_resignation_probability'].mean().reset_index()
            fig_dept, ax_dept = plt.subplots(figsize=(8, 6))
            sns.barplot(x='Department', y='employee_resignation_probability', data=dept_risk, palette='Blues_d', ax=ax_dept)
            ax_dept.set_title('Average Risk by Department')
            st.pyplot(fig_dept)
            
    st.markdown("---")
    col_sat, col_tenure = st.columns(2)
    with col_sat:
        st.subheader("3. Satisfaction vs Risk")
        if 'JobSatisfaction' in viz_df.columns:
            order_list = ['Low', 'Medium', 'High', 'Very High']
            actual_order = [x for x in order_list if x in viz_df['JobSatisfaction'].unique()]
            fig_sat, ax_sat = plt.subplots(figsize=(8, 5))
            sns.barplot(x='JobSatisfaction', y='employee_resignation_probability', data=viz_df, order=actual_order, palette='viridis', ax=ax_sat)
            ax_sat.set_title('Job Satisfaction Impact')
            st.pyplot(fig_sat)
    
    with col_tenure:
        st.subheader("4. Tenure vs Risk")
        if 'YearsAtCompany' in viz_df.columns:
             fig_ten, ax_ten = plt.subplots(figsize=(8, 5))
             sns.lineplot(x='YearsAtCompany', y='employee_resignation_probability', data=viz_df, color='purple', ax=ax_ten)
             ax_ten.set_title('Risk over Years at Company')
             st.pyplot(fig_ten)
             
    st.markdown("---")
    st.subheader("5. Financial & Commute Analysis")
    col_income, col_commute = st.columns(2)
    
    with col_income:
        if 'MonthlyIncome_Real' in viz_df.columns:
            fig_inc, ax_inc = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x='MonthlyIncome_Real', y='employee_resignation_probability', data=viz_df, hue='Department', alpha=0.6, palette='viridis', ax=ax_inc)
            ax_inc.set_title('Income vs. Risk')
            ax_inc.set_xlabel('Monthly Income ($)')
            st.pyplot(fig_inc)
            
    with col_commute:
        if 'DistanceFromHome_Real' in viz_df.columns:
            fig_com, ax_com = plt.subplots(figsize=(8, 5))
            viz_df['Risk_Status'] = viz_df['employee_resignation_probability'].apply(lambda x: 'High' if x > risk_threshold else 'Normal')
            sns.boxplot(x='Risk_Status', y='DistanceFromHome_Real', data=viz_df, palette='Set2', ax=ax_com)
            ax_com.set_title('Commute Distance vs. Risk')
            ax_com.set_xlabel('Risk Status')
            ax_com.set_ylabel('Distance (km)')
            st.pyplot(fig_com)

# --- Tab 3: Prediction Refresh ---
with tab3:
    st.header("🔄 Predict New Employees")
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    MODEL_PATH = 'best_attrition_model.pkl'
    
    if uploaded_file is not None:
        try:
            # 1. Load Raw Data
            new_data_raw = pd.read_csv(uploaded_file)
            
            # ==========================================
            # 🔧 FIX: Create '_Real' columns to match Sidebar
            # ==========================================
            # Create '_Real' column names to match Sidebar options for consistent display
            for col in new_data_raw.columns:
                if col in unscale_params: # Check if it's a numerical column we previously unscaled
                    new_data_raw[f"{col}_Real"] = new_data_raw[col]
            
            # Copy other numeric columns that didn't need unscaling but might be selected
            numeric_cols_fix = ['Education', 'JobLevel', 'StockOptionLevel', 'JobInvolvement', 'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance', 'EnvironmentSatisfaction', 'JobSatisfaction']
            for c in numeric_cols_fix:
                if c in new_data_raw.columns:
                    new_data_raw[f"{c}_Real"] = new_data_raw[c]
            # ==========================================

            st.markdown("### 1. Raw Data Preview")
            st.dataframe(new_data_raw.head())
            
            if st.button("Run Prediction"):
                if os.path.exists(MODEL_PATH):
                    try:
                        # --- PREPROCESSING START ---
                        X_new = new_data_raw.copy()
                        
                        # 1. Encode Basic Columns (Convert Text -> Number)
                        for col, mapping in maps.items():
                            if col in X_new.columns and col != 'EducationField':
                                reverse_map = {v: k for k, v in mapping.items()}
                                X_new[col] = X_new[col].map(reverse_map).fillna(X_new[col])

                        # 2. Handle EducationField (Manual One-Hot Encoding)
                        if 'EducationField' in X_new.columns:
                            edu_col = X_new['EducationField']
                        else:
                            edu_col = pd.Series(['Other'] * len(X_new))

                        # Map Text -> Number
                        edu_map_vals = {
                            'Life Sciences': 5, 'Medical': 4, 'Marketing': 3, 
                            'Technical Degree': 2, 'Other': 0, 'Human Resources': 1
                        }
                        edu_num = edu_col.map(edu_map_vals).fillna(0)
                        
                        # Create Dummy Columns
                        for i in [0, 1, 2, 3, 4, 5]:
                            X_new[f'EducationField_{i}'] = (edu_num == i).astype(int)

                        # 3. Ensure Base Features Exist
                        base_features = [
                            'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 
                            'Education', 'EnvironmentSatisfaction', 'Gender', 
                            'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 
                            'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
                            'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 
                            'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 
                            'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 
                            'YearsSinceLastPromotion', 'YearsWithCurrManager'
                        ]
                        for f in base_features:
                            if f not in X_new.columns: X_new[f] = 0
                        
                        # 4. Ensure Numeric & Scale Base Features
                        for col in base_features:
                            X_new[col] = pd.to_numeric(X_new[col], errors='coerce').fillna(0)
                            if col in unscale_params:
                                params = unscale_params[col]
                                X_new[col] = (X_new[col] - params['mean']) / params['std']

                        # --- REORDER & PREDICT ---
                        loaded_object = joblib.load(MODEL_PATH)
                        
                        # Helper to find the actual model (Bypass Pipeline/GridSearch)
                        def get_leaf_estimator(obj):
                            if hasattr(obj, 'best_estimator_'): return get_leaf_estimator(obj.best_estimator_)
                            if hasattr(obj, 'steps'): return get_leaf_estimator(obj.steps[-1][1])
                            return obj

                        final_model = get_leaf_estimator(loaded_object)
                        
                        # Auto-Reorder columns to match model expectations
                        model_feature_names = None
                        if hasattr(final_model, 'feature_names_'): model_feature_names = final_model.feature_names_
                        elif hasattr(final_model, 'get_feature_names'): model_feature_names = final_model.get_feature_names()
                        elif hasattr(final_model, 'feature_names_in_'): model_feature_names = final_model.feature_names_in_

                        if model_feature_names is not None:
                            X_final = pd.DataFrame()
                            for feature in model_feature_names:
                                if feature in X_new.columns:
                                    X_final[feature] = X_new[feature]
                                else:
                                    X_final[feature] = 0
                            X_new = X_final
                        
                        # Predict
                        probs = final_model.predict_proba(X_new)[:, 1]
                        new_data_raw['employee_resignation_probability'] = probs
                        
                        st.success("✅ Predictions generated successfully!")
                        st.markdown("### 2. Prediction Dashboard")

                        # --- DASHBOARD DISPLAY ---
                        
                        # Metrics Row
                        col_m1, col_m2, col_m3 = st.columns(3)
                        new_avg_risk = probs.mean()
                        new_high_risk_count = (probs > risk_threshold).sum()
                        new_high_risk_pct = (new_high_risk_count / len(probs)) * 100
                        
                        col_m1.metric("Average Probability", f"{new_avg_risk:.1%}")
                        col_m2.metric("High Risk Employees", f"{new_high_risk_count} Persons")
                        col_m3.metric("High Risk Percentage", f"{new_high_risk_pct:.1f}%")
                        
                        st.markdown("---")
                        
                        col_d1, col_d2 = st.columns([2, 1])
                        
                        # Left: Donut Chart (Risk Breakdown)
                        with col_d1:
                            st.subheader("Risk Category Breakdown")
                            
                            # Categorize Risk Function
                            def categorize_risk(prob, threshold):
                                if prob < (threshold * 0.5): return 'Safe Zone'
                                elif prob < threshold: return 'Watchlist'
                                else: return 'High Risk'

                            new_data_raw['Risk_Category'] = new_data_raw['employee_resignation_probability'].apply(lambda x: categorize_risk(x, risk_threshold))
                            risk_counts_new = new_data_raw['Risk_Category'].value_counts()
                            
                            # Color Mapping
                            color_map = {'Safe Zone': '#4CAF50', 'Watchlist': '#FFC107', 'High Risk': '#FF5252'}
                            colors = [color_map.get(cat, '#9E9E9E') for cat in risk_counts_new.index]

                            # Plot Donut Chart
                            fig_donut_new, ax_donut_new = plt.subplots(figsize=(8, 5))
                            wedges, texts, autotexts = ax_donut_new.pie(
                                risk_counts_new, 
                                labels=risk_counts_new.index, 
                                autopct='%1.1f%%', 
                                startangle=90, 
                                colors=colors, 
                                pctdistance=0.85, 
                                wedgeprops=dict(width=0.3)
                            )
                            plt.setp(texts, size=10, weight="bold")
                            plt.setp(autotexts, size=10, weight="bold", color="white")
                            st.pyplot(fig_donut_new)
                        
                        # Right: Top High Risk Cards
                        with col_d2:
                            st.subheader(f"🚨 Top {top_n_employees} High Risk")
                            top_risk_new = new_data_raw[new_data_raw['employee_resignation_probability'] > risk_threshold]
                            top_risk_new = top_risk_new.sort_values(by='employee_resignation_probability', ascending=False).head(top_n_employees)
                            
                            if top_risk_new.empty:
                                st.success("No high risk employees found in this batch! 🎉")
                            else:
                                for index, row in top_risk_new.iterrows():
                                    with st.container(border=True):
                                        # Avatar Image
                                        st.markdown(f"""
                                            <div style="display: flex; justify-content: center; margin-top: 10px; margin-bottom: 10px;">
                                                <img src="https://i.pravatar.cc/150?u={index + 500}" style="width: 100px; height: 100px; border-radius: 50%; object-fit: cover;">
                                            </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Employee Info Header
                                        emp_id = row.get('EmployeeNumber', index)
                                        st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 1.2em;'>ID: {emp_id}</div>", unsafe_allow_html=True)
                                        st.markdown(f"<div style='text-align: center; color: #FF5252; font-weight: bold;'>Risk: {row['employee_resignation_probability']:.1%}</div>", unsafe_allow_html=True)
                                        st.markdown(f"<div style='text-align: center; color: gray; font-size: 0.9em; margin-bottom: 10px;'>{row.get('JobRole','-')} | {row.get('Department','-')}</div>", unsafe_allow_html=True)
                                        
                                        st.markdown("---")
                                        
                                        # Display Selected Details (Works with both Original and _Real names)
                                        if selected_card_details:
                                            for col_name in selected_card_details:
                                                # Check if column exists directly or as a base name
                                                val = None
                                                if col_name in row:
                                                    val = row[col_name]
                                                elif col_name.replace('_Real', '') in row:
                                                    val = row[col_name.replace('_Real', '')]
                                                
                                                if val is not None:
                                                    val_str = str(val)
                                                    # Format numbers nicely
                                                    if isinstance(val, (int, float)):
                                                        if 'Income' in col_name or 'Rate' in col_name: val_str = f"${val:,.0f}"
                                                        elif 'Distance' in col_name: val_str = f"{val:.0f} km" 
                                                        elif 'Percent' in col_name: val_str = f"{val}%"
                                                    
                                                    label = col_name.replace('_Real','').replace('_',' ')
                                                    st.markdown(f"<div style='text-align: center; margin-bottom: 4px;'><b>{label}:</b> {val_str}</div>", unsafe_allow_html=True)
                                        
                                        # Bottom Spacing
                                        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

                        # Download Section
                        st.markdown("---")
                        st.markdown("### 📥 Download Results")
                        csv = new_data_raw.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Results (CSV)", csv, "prediction_results.csv", "text/csv")
                        
                    except ValueError as ve:
                        st.error(f"⚠️ Data Mismatch: {ve}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning(f"⚠️ Model '{MODEL_PATH}' not found.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
st.markdown("---")
st.caption("HR Analytics Dashboard | Built with Royson Dsouza & Sarawut Boonyarat")