import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Adult Income Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    body, .stApp {
        background-color: #F4F7F7;
    }

    .main > div {
        padding-top: 2rem;
    }
    
    .stTitle {
        text-align: center;
        color: #1a1a1a;
        font-size: 3rem !important;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .prediction-card {
        background: #93DA97;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .prediction-result {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #071E3D;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        border-left: 4px solid #2563eb;
        margin: 0.5rem 0;
    }
    
    .sidebar-header {
        background: #071E3D;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


model = joblib.load('optimal_model.pkl')

st.markdown("""
<div style="background: #071E3D; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 4px 20px rgba(37, 99, 235, 0.15);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 3rem; font-weight: 700;">💰 Adult Income Predictor</h1>
    <p style="color: rgba(255, 255, 255, 0.9); text-align: center; font-size: 1.2rem; margin: 0.5rem 0 0 0; font-weight: 400;">
        Predict whether an individual earns more than $50,000 per year using demographic data and machine learning models.
    </p>
</div>
""", unsafe_allow_html=True)

with st.expander("📊 Explore Dataset & Model Insights", expanded=False):
    st.markdown("### 📈 Dataset Overview")
    st.markdown("**Adult Income Dataset** - Comprehensive demographic and employment data")
    
    df1 = pd.read_csv('adult.csv')
    df1 = df1.drop(columns=['fnlwgt', 'education-num'])
    cols_to_remove = [col for col in df1.columns if col in ['workclass_ ?', 'occupation_ ?', 'native-country_ ?']]
    df1 = df1.drop(columns=cols_to_remove)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", f"{len(df1):,}")
    with col2:
        st.metric("Features", f"{len(df1.columns)-1}")
    
    st.dataframe(df1, use_container_width=True)

    # st.markdown("### 🎯 Features (X)")
    X_raw = df1.drop('target', axis=1)
    # st.dataframe(X_raw, use_container_width=True)

    # st.markdown("### 🏷️ Target Variable (y)")
    y_raw = df1.target
    # st.write(y_raw)

with st.expander("📊 Feature-Target Relationships", expanded=False):
    
    st.markdown("### Age Distribution by Income Level")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    df1.boxplot(column='age', by='target', ax=ax1)
    ax1.set_title('Age Distribution by Income Level')
    ax1.set_xlabel('Income Level')
    ax1.set_ylabel('Age')
    plt.suptitle('')  
    st.pyplot(fig1)
    
    st.markdown("### Education Level Distribution")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    education_counts = df1.groupby(['education', 'target']).size().unstack()
    education_counts.plot(kind='bar', ax=ax2, color=['#ff7f7f', '#7fbf7f'])
    ax2.set_title('Education Level by Income')
    ax2.set_xlabel('Education Level')
    ax2.set_ylabel('Count')
    ax2.legend(['≤ $50K', '> $50K'])
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    
    st.markdown("### Working Hours Distribution")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    df1.boxplot(column='hours-per-week', by='target', ax=ax3)
    ax3.set_title('Working Hours by Income Level')
    ax3.set_xlabel('Income Level')
    ax3.set_ylabel('Hours per Week')
    plt.suptitle('')
    st.pyplot(fig3)
    
    st.markdown("### Gender Distribution by Income")
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    gender_income = df1.groupby(['sex', 'target']).size().unstack()
    gender_income.plot(kind='bar', ax=ax4, color=['#ff7f7f', '#7fbf7f'])
    ax4.set_title('Gender Distribution by Income Level')
    ax4.set_xlabel('Gender')
    ax4.set_ylabel('Count')
    ax4.legend(['≤ $50K', '> $50K'])
    plt.xticks(rotation=0)
    st.pyplot(fig4)
    
    st.markdown("### Top Occupations by Income")
    fig5, ax5 = plt.subplots(figsize=(12, 8))
    top_occupations = df1['occupation'].value_counts().head(10).index
    df_top_occ = df1[df1['occupation'].isin(top_occupations)]
    occ_income = df_top_occ.groupby(['occupation', 'target']).size().unstack()
    occ_income.plot(kind='barh', ax=ax5, color=['#ff7f7f', '#7fbf7f'])
    ax5.set_title('Top 10 Occupations by Income Level')
    ax5.set_xlabel('Count')
    ax5.set_ylabel('Occupation')
    ax5.legend(['≤ $50K', '> $50K'])
    st.pyplot(fig5)
    
    st.markdown("### Numeric Features Correlation")
    numeric_cols = ['age', 'hours-per-week', 'capital-gain', 'capital-loss']
    df_numeric = df1[numeric_cols].copy()

    df_numeric['income_numeric'] = df1['target'].map({' <=50K': 0, ' >50K': 1})
    
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    correlation_matrix = df_numeric.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax6)
    ax6.set_title('Correlation Matrix: Numeric Features vs Income')

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2 style="margin: 0;">⚙️ Input Features</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Configure prediction parameters</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 👤 Personal Information")
    age = st.slider("Age", 17, 90, 35, help="Your current age")
    sex = st.selectbox("Sex", ['Female', 'Male'])
    race = st.selectbox("Race", [
        'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'
    ])

    st.markdown("### 🎓 Education & Work")
    education = st.selectbox("Education Level", [
        'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
        'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
        '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'
    ], help="Highest education level achieved")

    workclass = st.selectbox("Work Class", [
        'Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
        'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'
    ], help="Type of employment")

    occupation = st.selectbox("Occupation", [
        'Tech-support', 'Craft-repair', 'Other-service', 'Sales',
        'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ], help="Your primary occupation")

    hours_per_week = st.slider("Hours per Week", 1, 99, 40, help="Average hours worked per week")

    st.markdown("### 💑 Relationships")
    marital_status = st.selectbox("Marital Status", [
        'Married-civ-spouse', 'Divorced', 'Never-married',
        'Separated', 'Widowed', 'Married-spouse-absent'
    ], help="Your current marital status")

    relationship = st.selectbox("Relationship Status", [
        'Wife', 'Own-child', 'Husband', 'Not-in-family',
        'Other-relative', 'Unmarried'
    ], help="Relationship status")

    st.markdown("### 💰 Financial Information")
    capital_gain = st.slider("Capital Gain", 0, 99999, 0, help="Investment income, if any")
    capital_loss = st.slider("Capital Loss", 0, 4356, 0, help="Investment losses, if any")

    st.markdown("### 🌍 Location")
    native_country = st.selectbox("Native Country", [
        'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
        'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan',
        'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras',
        'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam',
        'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic',
        'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
        'Hungary', 'Guatemala', 'Nicaragua', 'Scotland',
        'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago',
        'Peru', 'Hong', 'Holand-Netherlands'
    ], help="Country of birth")

    data = {
        'age': age,
        'workclass': workclass,
        'education': education,
        'marital-status': marital_status,   
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }
    input_df = pd.DataFrame(data, index=[0])
    input_adult = pd.concat([input_df, X_raw], axis=0)

with st.expander("🔍 Input Features Summary", expanded=False):
    st.markdown("### Your Input Parameters")
    st.dataframe(input_df, use_container_width=True)

encode = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'education']
df_adult = pd.get_dummies(input_adult, prefix=encode)

X = df_adult[1:]
input_row = df_adult[:1]

target_mapper = {
    ' <=50K': 0,
    ' >50K': 1
}

def target_encode(val):
    return target_mapper.get(val)

y = y_raw.apply(target_encode)

# with st.expander(" Data Preparation Details", expanded=False):
#     st.markdown("### Encoded Feature Vector")
#     st.dataframe(input_row, use_container_width=True)

clf = model
clf.fit(X, y)

prediction = model.predict(input_row)
prediction_proba = model.predict_proba(input_row)

df1_prediction_proba = pd.DataFrame(prediction_proba)
df1_prediction_proba.columns = ['≤ $50K', '> $50K']

st.markdown("## Prediction Results")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 Probability Distribution")
    st.dataframe(df1_prediction_proba, 
                column_config={
                    '≤ $50K': st.column_config.ProgressColumn(
                        'Probability ≤ $50K',
                        format='%.2f',
                        width='medium',
                        min_value=0.0,
                        max_value=1.0
                    ),
                    '> $50K': st.column_config.ProgressColumn(
                        'Probability > $50K',
                        format='%.2f',
                        width='medium',
                        min_value=0.0,
                        max_value=1.0
                    )
                }, 
                hide_index=True,
                use_container_width=True)

with col2:
    adult_salary = np.array(['≤ $50K', '> $50K'])
    result = adult_salary[prediction[0]]
    confidence = max(prediction_proba[0]) * 100
    
    if result == '> $50K':
        st.markdown(f"""
        <div class="prediction-card">
            <h3 style="margin-top: 0;">💰 Predicted Income</h3>
            <div class="prediction-result">{result}</div>
            <p style="font-size: 1.2rem;">Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: #E55050; padding: 2rem; border-radius: 15px; text-align: center; color: #2d3436;">
            <h3 style="margin-top: 0;">💼 Predicted Income</h3>
            <div class="prediction-result" style="color: #2d3436;">{result}</div>
            <p style="font-size: 1.2rem;">Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("### 📈 Prediction Confidence Visualization")

fig = go.Figure(data=[
    go.Bar(
        x=['≤ $50K', '> $50K'],
        y=prediction_proba[0],
        marker_color=['#ff7f7f', '#7fbf7f'],
        text=[f'{prob:.2%}' for prob in prediction_proba[0]],
        textposition='auto',
    )
])

fig.update_layout(
    title="Prediction Probabilities",
    title_x=0.5,
    xaxis_title="Income Category",
    yaxis_title="Probability",
    yaxis=dict(range=[0, 1]),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(size=14),
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

