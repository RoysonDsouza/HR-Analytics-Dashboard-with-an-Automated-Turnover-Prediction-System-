import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np
from matplotlib.figure import Figure
from datetime import datetime
import time

# Import auto_pipeline if available
try:
    import auto_pipeline
except ImportError:
    auto_pipeline = None

# ==========================================
# 1. Page Configuration & Session State
# ==========================================
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- UI Settings Option ---
IMAGE_OPTIONS = {
    'Local (HR.png)': '/Users/barabank/Capstone project/HR.png', 
    'Online (Office)': "https://images.unsplash.com/photo-1556761175-5973dc0f32e7?q=80&w=2832&auto=format&fit=crop",
}

COLOR_OPTIONS = {
    'Red (Default)': '#FF4B4B',
    'Blue (Corporate)': '#1F77B4',     
    'Green (HR Focus)': '#2CA02C',     
    'Purple (Vibrant)': '#9467BD',     
    'Orange (Alert)': '#FF7F0E',        
    'Teal (Calm)': '#17BECF',           
    'Pink (Energetic)': '#E377C2',     
    'Grey (Neutral)': '#7F7F7F',        
    'Yellow (Warning)': '#BCBD22',     
    'Brown (Earth)': '#8C564B',        
}

# Init Session State
if 'cover_image_key' not in st.session_state:
    st.session_state.cover_image_key = list(IMAGE_OPTIONS.keys())[1] # Default to Online
if 'primary_color' not in st.session_state:
    st.session_state.primary_color = '#FF4B4B'
if 'uploaded_cover_file' not in st.session_state:
    st.session_state.uploaded_cover_file = None

# Callbacks
def update_color():
    st.session_state.primary_color = COLOR_OPTIONS[st.session_state.color_selector]

def upload_callback():
    st.session_state.uploaded_cover_file = st.session_state.cover_uploader

def select_default_callback():
    st.session_state.uploaded_cover_file = None 

# --- 🎨 CSS STYLING ---
st.markdown(f"""
<style>
    :root {{ --primary-color: {st.session_state.primary_color}; }}
    
    /* Metrics Box */
    div[data-testid="stMetric"] {{
        background-color: #ffffff;
        border: 1px solid #f0f0f0;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }}
    
    /* Card Style */
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        background-color: white;
    }}

    /* Headers */
    .css-164nlkn {{ font-weight: bold; }}

    /* Dynamic Elements Color */
    .stButton>button, .stRadio div[role="radio"], .stSelectbox div[role="listbox"] {{
        border-color: var(--primary-color);
        color: var(--primary-color);
    }}
    
    /* Active Tab Highlight */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        border-bottom-color: var(--primary-color) !important;
        color: var(--primary-color) !important;
    }}

    /* MultiSelect Tag Background Color */
    span[data-baseweb="tag"] {{
        background-color: {st.session_state.primary_color} !important;
        color: white !important;
    }}

    /* Update Time Badge (From App 5) */
    .update-badge {{
        background-color: #f0f2f6;
        color: #555;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.9em;
        border: 1px solid #ddd;
        display: inline-block;
    }}
</style>
""", unsafe_allow_html=True)

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# 2. Data Loading & Helper Functions
# ==========================================
DATA_FILE = 'dashboard_data.csv'

def get_db_update_time(file_path):
    if os.path.exists(file_path):
        mod_time = os.path.getmtime(file_path)
        return datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
    return "Unknown"

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    elif os.path.exists('train_with_resignation_probabilities.csv'):
        return pd.read_csv('train_with_resignation_probabilities.csv')
    return None

df = load_data()
current_file = DATA_FILE if os.path.exists(DATA_FILE) else 'train_with_resignation_probabilities.csv'
last_update_str = get_db_update_time(current_file)

# --- Data Processing (Map & Unscale) ---
if df is not None:
    # Helper to map scaled values back to labels
    def get_label(val, mapping_dict):
        if val is None: return val
        try:
            if pd.isna(val): return val
        except: pass
        try:
            closest_key = min(mapping_dict.keys(), key=lambda k: abs(k - float(val)))
            return mapping_dict[closest_key]
        except: return val

    # Maps & Scaling Params
    maps = {
        'Department': {-2.8450: 'Human Resources', -1.0767: 'Sales', 0.6917: 'Research & Development'},
        'JobRole': {-1.0897: 'Sales Executive', -0.6575: 'Research Scientist', -0.2253: 'Laboratory Technician', 0.2069: 'Manufacturing Director', 0.6391: 'Healthcare Representative', 1.0713: 'Manager', 1.5035: 'Sales Representative', 1.9357: 'Research Director', 2.3679: 'Human Resources'},
        'MaritalStatus': {-1.5599: 'Divorced', -0.2974: 'Single', 0.9652: 'Married'},
        'BusinessTravel': {-2.0236: 'Non-Travel', -0.1615: 'Travel_Rarely', 1.7006: 'Travel_Frequently'},
        'Gender': {0.0: 'Female', 1.0: 'Male'},
        'OverTime': {0.0: 'No', 1.0: 'Yes'},
        'Attrition': {0.0: 'No', 1.0: 'Yes'},
        'JobSatisfaction': {-1.5488: 'Low', -0.6480: 'Medium', 0.2528: 'High', 1.1535: 'Very High'},
        'EnvironmentSatisfaction': {-1.5776: 'Low', -0.6587: 'Medium', 0.2602: 'High', 1.1791: 'Very High'},
        'RelationshipSatisfaction': {-1.6002: 'Low', -0.6800: 'Medium', 0.2402: 'High', 1.1604: 'Very High'},
        'WorkLifeBalance': {-2.4486: 'Bad', -1.0555: 'Good', 0.3376: 'Better', 1.7308: 'Best'},
        'Age': {-2.0708: 18, -1.9618: 19, -1.8528: 20, -1.7438: 21, -1.6348: 22, -1.5258: 23, -1.4168: 24, -1.3078: 25, -1.1988: 26, -1.0898: 27, -0.9808: 28, -0.8718: 29, -0.7628: 30, -0.6538: 31, -0.5448: 32, -0.4358: 33, -0.3268: 34, -0.2178: 35, -0.1088: 36, 0.0002: 37, 0.1092: 38, 0.2182: 39, 0.3272: 40, 0.4362: 41, 0.5452: 42, 0.6542: 43, 0.7632: 44, 0.8722: 45, 0.9812: 46, 1.0902: 47, 1.1992: 48, 1.3082: 49, 1.4172: 50, 1.5262: 51, 1.6352: 52, 1.7442: 53, 1.8532: 54, 1.9622: 55, 2.0712: 56, 2.1802: 57, 2.2892: 58, 2.3982: 59, 2.5072: 60},
        'YearsAtCompany': {-1.1588: 0, -0.9944: 1, -0.8301: 2, -0.6657: 3, -0.5013: 4, -0.3370: 5, -0.1726: 6, -0.0082: 7, 0.1561: 8, 0.3205: 9, 0.4848: 10, 0.6492: 11, 0.8136: 12, 0.9779: 13, 1.1423: 14, 1.3067: 15, 1.4710: 16, 1.6354: 17, 1.7998: 18, 1.9641: 19, 2.1285: 20, 2.2929: 21, 2.4572: 22, 2.6216: 23, 2.7860: 24, 2.9503: 25, 3.1147: 26, 3.2791: 27, 3.6078: 29, 3.7721: 30, 3.9365: 31, 4.1009: 32, 4.2652: 33, 4.4296: 34, 4.7583: 36, 4.9227: 37}
    }

    # Apply Maps
    for col, mapping in maps.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: get_label(x, mapping))

    # EducationField
    if 'EducationField' in df.columns:
        edu_map = {'5': 'Life Sciences', '5.0': 'Life Sciences', '4': 'Medical', '4.0': 'Medical', '3': 'Marketing', '3.0': 'Marketing', '2': 'Technical Degree', '2.0': 'Technical Degree', '0': 'Other', '0.0': 'Other', 'Human Resources': 'Human Resources'}
        df['EducationField'] = df['EducationField'].astype(str).map(edu_map).fillna(df['EducationField'])

    # Unscale Params
    unscale_params = {
        'MonthlyIncome': {'mean': 6544.02, 'std': 4651.76}, 'DistanceFromHome': {'mean': 9.36, 'std': 8.18}, 'DailyRate': {'mean': 803.99, 'std': 401.17}, 'Education': {'mean': 2.91, 'std': 1.03}, 'JobLevel': {'mean': 2.08, 'std': 1.09}, 'StockOptionLevel': {'mean': 0.79, 'std': 0.85}, 'NumCompaniesWorked': {'mean': 2.69, 'std': 2.49}, 'PercentSalaryHike': {'mean': 15.24, 'std': 3.68}, 'YearsSinceLastPromotion': {'mean': 2.18, 'std': 3.21}, 'JobInvolvement': {'mean': 2.74, 'std': 0.70}, 'HourlyRate': {'mean': 65.50, 'std': 20.36}, 'MonthlyRate': {'mean': 14390.24, 'std': 7189.78}, 'PerformanceRating': {'mean': 3.16, 'std': 0.36}, 'TotalWorkingYears': {'mean': 11.36, 'std': 7.80}, 'TrainingTimesLastYear': {'mean': 2.76, 'std': 1.26}, 'YearsInCurrentRole': {'mean': 4.23, 'std': 3.57}, 'YearsWithCurrManager': {'mean': 4.20, 'std': 3.56}
    }

    for col, params in unscale_params.items():
        if col in df.columns:
            real_col_name = f"{col}_Real"
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[real_col_name] = (df[col] * params['std']) + params['mean']
            if col in ['Education', 'JobLevel', 'StockOptionLevel', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsInCurrentRole', 'YearsWithCurrManager', 'YearsSinceLastPromotion', 'HourlyRate', 'MonthlyRate', 'DailyRate']:
                df[real_col_name] = df[real_col_name].round(0).astype(int)

# ==========================================
# 3. Sidebar (Settings & Admin)
# ==========================================
st.sidebar.title("⚙️ Control Panel")

# --- Appearance ---
st.sidebar.subheader("🎨 Appearance")
# Theme Color
st.sidebar.selectbox(
    "Theme Color", 
    options=list(COLOR_OPTIONS.keys()),
    index=list(COLOR_OPTIONS.keys()).index(next(key for key, val in COLOR_OPTIONS.items() if val == st.session_state.primary_color)),
    key='color_selector',
    on_change=update_color
)

# Cover Image
st.sidebar.file_uploader("Upload Cover Image", type=['png', 'jpg'], key='cover_uploader', on_change=upload_callback)
st.sidebar.selectbox("Or Select Default", options=list(IMAGE_OPTIONS.keys()), key='cover_image_key', on_change=select_default_callback)

# Graph Settings
st.sidebar.subheader("📊 Graph Settings")
chart_style = st.sidebar.selectbox(
    "Risk Breakdown Chart Style",
    ["Pie Chart", "Donut Chart", "Bar Chart"],
    index=1
)

st.sidebar.markdown("---")

# --- Admin Control ---
st.sidebar.subheader("🛠️ Admin / Simulation")
st.sidebar.info("Simulate New DB Generation")
num_employees = st.sidebar.number_input("Generate Rows:", min_value=100, max_value=10000, value=4000, step=100)

# --- 🎨 CSS: Customize Sidebar Primary Button to match 'Browse files' style ---
st.markdown("""
<style>
    /* Target only Primary buttons within the Sidebar */
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #f8f9fb !important; /* Very light gray/off-white background */
        color: #31333f !important;             /* Dark gray text color */
        border: 1px solid #d6d6d8 !important;  /* Light gray border line */
        border-radius: 8px !important;         /* Rounded corners */
        padding: 0.25rem 0.75rem !important;   /* Button padding */
        transition: 0.2s;                      /* Smooth transition effect */
    }

    /* Hover effect to make it interactive */
    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        border-color: #b0b0b2 !important;     /* Slightly darker border on hover */
        background-color: #f0f2f6 !important; /* Slightly darker gray on hover */
    }
    
    /* Ensure text stays dark even when focused */
    section[data-testid="stSidebar"] button[kind="primary"]:focus {
        color: #31333f !important;
        border-color: #d6d6d8 !important;
    }
</style>
""", unsafe_allow_html=True)

if st.sidebar.button("🔄 Generate & Update DB", type="primary"):
    if auto_pipeline is None:
        st.sidebar.error("auto_pipeline.py not found!")
    else:
        with st.sidebar.status("Processing new database...", expanded=True) as status:
            st.write(f"Generating {num_employees} mock records...")
            time.sleep(0.5)
            st.write("Replacing Database...")
            
            # Call the pipeline function
            try:
                success = auto_pipeline.regenerate_database(num_employees) 
                if success:
                    status.update(label=f"✅ DB Updated ({num_employees} rows)!", state="complete", expanded=False)
                    time.sleep(0.5)
                    st.cache_data.clear() # Clear Cache
                    st.rerun() 
                else:
                    status.update(label="❌ Failed", state="error")
                    st.sidebar.error("Check source CSV.")
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.sidebar.error(f"Pipeline Error: {e}")

st.sidebar.markdown("---")

# --- Filters ---
st.sidebar.subheader("🔍 Filters")

if df is not None:
    all_departments = df['Department'].unique()
    selected_dept = st.sidebar.multiselect("Department", options=all_departments, default=all_departments)

    if 'JobRole' in df.columns:
        all_roles = df['JobRole'].unique()
        selected_role = st.sidebar.multiselect("Select Job Role", options=all_roles, default=all_roles)
    else:
        selected_role = []

    risk_threshold = st.sidebar.slider("High Risk Threshold", 0.0, 1.0, 0.5, 0.05)
    top_n_employees = st.sidebar.number_input("Show Top N Risk Employees", min_value=1, max_value=100, value=5, step=1)
    
    # Filter Columns for Card Display
    exclude_cols = ['Department', 'JobRole', 'employee_resignation_probability', 'Attrition', 'MonthlyIncome', 'DistanceFromHome', 'DailyRate', 'Education', 'JobLevel', 'StockOptionLevel', 'NumCompaniesWorked', 'PercentSalaryHike', 'YearsSinceLastPromotion', 'JobInvolvement', 'HourlyRate', 'MonthlyRate', 'PerformanceRating', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsInCurrentRole', 'YearsWithCurrManager', 'Education_Real', 'StockOptionLevel_Real', 'JobInvolvement_Real', 'EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    available_cols = sorted([c for c in df.columns if c not in exclude_cols])
    default_cols = ['Age', 'JobSatisfaction', 'OverTime']
    
    for c in ['MonthlyIncome_Real', 'DistanceFromHome_Real', 'JobLevel_Real']:
        if c in available_cols: default_cols.append(c)
    
    final_defaults = [c for c in default_cols if c in available_cols]
    selected_card_details = st.sidebar.multiselect("Card Details", options=available_cols, default=final_defaults)

    filtered_df = df[(df['Department'].isin(selected_dept))] if selected_dept else df
    if selected_role:
        filtered_df = filtered_df[filtered_df['JobRole'].isin(selected_role)]
    
    st.sidebar.info(f"Showing {len(filtered_df)} employees")
else:
    st.error("Data not found.")
    st.stop()

# ==========================================
# 4. Main Dashboard Layout
# ==========================================

# --- Header & Update Badge ---
c1, c2 = st.columns([3, 1])
with c1:
    st.title("📊 HR Analytics Dashboard")
with c2:
    st.markdown(f"""
        <div style="text-align: right; margin-top: 20px;">
            <span class="update-badge">🕒 Last Updated: {last_update_str}</span>
        </div>
    """, unsafe_allow_html=True)

# --- Cover Image ---
if st.session_state.uploaded_cover_file is not None:
    image_to_display = st.session_state.uploaded_cover_file
else:
    image_to_display = IMAGE_OPTIONS[st.session_state.cover_image_key]

try:
    st.image(image_to_display, use_container_width=True)
except:
    st.warning("Image not found. Using placeholder.")

st.markdown(f"**Overview:** Analyzing retention risks for **{len(filtered_df)}** employees. **Threshold:** >{risk_threshold*100:.0f}%")

if filtered_df.empty:
    st.warning("No data available for the current selection.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Risk Overview", "🔍 Deep Dive Analysis", "🤖 Predict New Data"])

# --- Tab 1: Risk Overview ---
with tab1:
    col_m1, col_m2, col_m3 = st.columns(3)
    avg_risk = filtered_df['employee_resignation_probability'].mean()
    high_risk_employees = filtered_df[filtered_df['employee_resignation_probability'] > risk_threshold]
    
    col_m1.metric("Average Probability", f"{avg_risk:.1%}")
    col_m2.metric("High Risk Employees", f"{len(high_risk_employees)} Persons")
    col_m3.metric("High Risk Percentage", f"{(len(high_risk_employees)/len(filtered_df)*100):.1f}%")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Risk Category Breakdown")
        
        # Risk categorization
        def categorize_risk(prob, threshold):
            if prob < (threshold * 0.5): return 'Safe Zone'
            elif prob < threshold: return 'Watchlist'
            else: return 'High Risk'
        
        filtered_df['Risk_Category'] = filtered_df['employee_resignation_probability'].apply(lambda x: categorize_risk(x, risk_threshold))
        risk_counts = filtered_df['Risk_Category'].value_counts()
        
        if not risk_counts.empty:
            # Explicit Figure for Streamlit
            fig_chart = Figure(figsize=(5, 5))
            ax_chart = fig_chart.subplots()
            
            color_map = {'Safe Zone': '#4CAF50', 'Watchlist': '#FFC107', 'High Risk': '#FF5252'}
            current_colors = [color_map.get(label, '#808080') for label in risk_counts.index]
            
            # Switch based on Chart Style selection (App 4 Feature)
            if chart_style == "Bar Chart":
                bars = ax_chart.bar(risk_counts.index, risk_counts.values, color=current_colors)
                ax_chart.set_ylabel("Count")
                ax_chart.bar_label(bars, fmt='%d', fontsize=10)
            elif chart_style == "Donut Chart":
                _ = ax_chart.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=current_colors, startangle=90, wedgeprops=dict(width=0.4))
                ax_chart.axis('equal')
            else:
                # Pie
                _ = ax_chart.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=current_colors, startangle=90)
                ax_chart.axis('equal')
            
            st.pyplot(fig_chart)
        else:
            st.info("No data available for risk breakdown.")

    with col2:
        st.subheader(f"🚨 Top {top_n_employees} High Risk")
        if not high_risk_employees.empty:
            latest = high_risk_employees.sort_values('employee_resignation_probability', ascending=False).head(top_n_employees)
            
            for index, row in latest.iterrows():
                with st.container(border=True):
                    emp_id = row.get('EmployeeNumber')
                    if pd.isna(emp_id): emp_id = index 
                    else: emp_id = int(emp_id)

                    st.markdown(f"""
                        <div style="display: flex; justify-content: center; margin-top: 10px; margin-bottom: 10px;">
                            <img src="https://i.pravatar.cc/150?u={emp_id}" style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover;">
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"<div style='text-align: center;'><b>ID: {emp_id}</b></div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; color: #FF5252; font-weight: bold;'>Risk: {row['employee_resignation_probability']:.1%}</div>", unsafe_allow_html=True)
                    st.markdown("---")
                    
                    if selected_card_details:
                        for col_name in selected_card_details:
                            val = row.get(col_name)
                            val_str = str(val)
                            
                            if isinstance(val, (int, float)):
                                if 'Income' in col_name or 'Rate' in col_name: val_str = f"${val:,.0f}"
                                elif 'Distance' in col_name: val_str = f"{int(val)} Km"
                            
                            st.markdown(f"<div style='text-align: center; font-size: 0.9em;'><b>{col_name.replace('_Real','')}:</b> {val_str}</div>", unsafe_allow_html=True)
                    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
        else:
            st.success("No high risk employees found.")

# --- Tab 2: Deep Dive Analysis ---
with tab2:
    st.header("Key Drivers & HR Insights")
    viz_df = filtered_df.copy()
    col_driver, col_dept = st.columns(2)
    
    with col_driver:
        st.subheader("1. Key Drivers")
        numeric_df = df.select_dtypes(include=[np.number])
        if 'employee_resignation_probability' in numeric_df.columns:
            corr = numeric_df.corrwith(numeric_df['employee_resignation_probability']).sort_values(ascending=False).drop(['employee_resignation_probability', 'Attrition'], errors='ignore')
            corr = corr[~corr.index.str.contains('_Real')]
            top_drivers = pd.concat([corr.head(5), corr.tail(5)])
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

# --- Tab 3: Predict New Data (Enhanced Visuals) ---
with tab3:
    st.header("🔄 Predict New Employees")
    uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
    MODEL_PATH = 'best_attrition_model.pkl'
    
    if uploaded_file is not None:
        try:
            # 1. Load Data
            new_data_raw = pd.read_csv(uploaded_file)
            
            # (Optional) สร้างคอลัมน์ _Real ไว้สำหรับแสดงผล (ถ้ามี logic unscale)
            # เพื่อให้ sidebar ที่เลือก 'MonthlyIncome_Real' ทำงานได้กับไฟล์ที่อัปโหลด
            for col in new_data_raw.columns:
                # ถ้า user อัปโหลดไฟล์ที่มีคอลัมน์ปกติ เราจะ copy ไปชื่อ _Real เพื่อให้ match กับตัวเลือกใน Sidebar
                if f"{col}_Real" not in new_data_raw.columns:
                    new_data_raw[f"{col}_Real"] = new_data_raw[col]
            
            st.markdown("### 1. Raw Data Preview")
            st.dataframe(new_data_raw.head())
            
            if st.button("Run Prediction"):
                if os.path.exists(MODEL_PATH):
                    try:
                        # 2. Load Model
                        final_model = joblib.load(MODEL_PATH)
                        
                        # 3. Prepare Data
                        X_new = new_data_raw.copy()
                        drop_cols_pred = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'Attrition', 'employee_resignation_probability']
                        # ลบคอลัมน์ _Real ที่เราเพิ่งสร้างหลอกๆ ออกก่อนส่งเข้า Model
                        X_new = X_new.drop(columns=[c for c in X_new.columns if '_Real' in c]) 
                        X_new = X_new.drop(columns=[c for c in drop_cols_pred if c in X_new.columns])
                        
                        # --- CatBoost Preparation ---
                        cat_cols = [col for col in X_new.columns if X_new[col].dtype == 'object']
                        for col in cat_cols:
                            X_new[col] = X_new[col].fillna('Missing').astype(str)
                            
                        # Align Columns
                        if hasattr(final_model, 'feature_names_'):
                            model_features = final_model.feature_names_
                            for f in model_features:
                                if f not in X_new.columns:
                                    X_new[f] = 0 if f not in cat_cols else 'Missing'
                            X_new = X_new[model_features]

                        # 4. Predict
                        probs = final_model.predict_proba(X_new)[:, 1]
                        
                        # 5. Show Results
                        new_data_raw['employee_resignation_probability'] = probs
                        
                        st.success("✅ Predictions generated successfully!")
                        st.markdown("### 2. Prediction Dashboard")

                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("Average Probability", f"{probs.mean():.1%}")
                        col_m2.metric("High Risk Employees", f"{(probs > risk_threshold).sum()}")
                        col_m3.metric("High Risk Percentage", f"{((probs > risk_threshold).sum() / len(probs)) * 100:.1f}%")
                        
                        st.markdown("---")
                        
                        # Layout: Graph (Left) vs High Risk Cards (Right)
                        col_d1, col_d2 = st.columns([2, 1])
                        
                        # --- Left: Graph ---
                        with col_d1:
                            st.subheader("Risk Category Breakdown")
                            def categorize_risk_temp(prob, threshold):
                                if prob < (threshold * 0.5): return 'Safe Zone'
                                elif prob < threshold: return 'Watchlist'
                                else: return 'High Risk'
                                
                            new_data_raw['Risk_Category'] = new_data_raw['employee_resignation_probability'].apply(lambda x: categorize_risk_temp(x, risk_threshold))
                            n_counts = new_data_raw['Risk_Category'].value_counts()
                            
                            if not n_counts.empty:
                                fig_pred = Figure(figsize=(5, 5))
                                ax_pred = fig_pred.subplots()
                                n_colors = [{'Safe Zone': '#4CAF50', 'Watchlist': '#FFC107', 'High Risk': '#FF5252'}.get(x, '#999') for x in n_counts.index]
                                
                                if chart_style == "Bar Chart":
                                    bars = ax_pred.bar(n_counts.index, n_counts.values, color=n_colors)
                                    ax_pred.bar_label(bars, fmt='%d')
                                else:
                                    ax_pred.pie(n_counts, labels=n_counts.index, autopct='%1.1f%%', colors=n_colors, startangle=90)
                                st.pyplot(fig_pred)

                        # --- Right: High Risk Cards (Updated Visuals) ---
                        with col_d2:
                            st.subheader(f"🚨 Top High Risk (New Data)")
                            # Filter & Sort
                            n_top = new_data_raw[new_data_raw['employee_resignation_probability'] > risk_threshold].sort_values(by='employee_resignation_probability', ascending=False).head(top_n_employees)
                            
                            if n_top.empty:
                                st.info("No high risk found.")
                            else:
                                for idx, row in n_top.iterrows():
                                    with st.container(border=True): # ใช้กรอบ Card แบบ Tab 1
                                        # Handle Employee ID
                                        emp_id_val = row.get('EmployeeNumber')
                                        if pd.isna(emp_id_val): emp_id_display = idx 
                                        else: emp_id_display = int(float(emp_id_val)) # Handle float ID

                                        # 1. Image Profile
                                        st.markdown(f"""
                                            <div style="display: flex; justify-content: center; margin-top: 10px; margin-bottom: 10px;">
                                                <img src="https://i.pravatar.cc/150?u={emp_id_display}" style="width: 80px; height: 80px; border-radius: 50%; object-fit: cover;">
                                            </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # 2. Header Info
                                        st.markdown(f"<div style='text-align: center;'><b>ID: {emp_id_display}</b></div>", unsafe_allow_html=True)
                                        st.markdown(f"<div style='text-align: center; color: #FF5252; font-weight: bold;'>Risk: {row['employee_resignation_probability']:.1%}</div>", unsafe_allow_html=True)
                                        st.markdown("---")
                                        
                                        # 3. Dynamic Details (from Sidebar)
                                        # ใช้ตัวแปร selected_card_details จาก Sidebar
                                        if selected_card_details:
                                            for cname in selected_card_details:
                                                # พยายามหาค่า (รองรับทั้งชื่อปกติ และชื่อมี _Real)
                                                val = None
                                                clean_name = cname.replace('_Real', '')
                                                
                                                if cname in row: val = row[cname]
                                                elif clean_name in row: val = row[clean_name]
                                                
                                                if val is not None:
                                                    disp_val = str(val)
                                                    # จัด Format เงิน/ระยะทาง ให้สวยงาม
                                                    try:
                                                        if isinstance(val, (int, float)):
                                                            if 'Income' in cname or 'Rate' in cname: disp_val = f"${val:,.0f}"
                                                            elif 'Distance' in cname: disp_val = f"{int(val)} Km"
                                                    except: pass
                                                    
                                                    st.markdown(f"<div style='text-align: center; font-size: 0.9em;'><b>{clean_name}:</b> {disp_val}</div>", unsafe_allow_html=True)
                                        
                                        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

                        st.markdown("---")
                        st.download_button(
                            "Download Results (CSV)",
                            new_data_raw.to_csv(index=False).encode('utf-8'),
                            "prediction_results.csv",
                            "text/csv"
                        )

                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                else:
                    st.warning("Model file not found. Please run the pipeline first.")
        except Exception as e:
            st.error(f"File Error: {e}")

st.markdown("---")
st.caption("HR Analytics Dashboard | Built with Royson Dsouza & Sarawut Boonyarat")