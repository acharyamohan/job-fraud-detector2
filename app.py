#!/usr/bin/env python3
"""
Job Fraud Detection System - Streamlit Version
Built for the DS-1 Hackathon Challenge
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import re
import string
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Job Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background: #ffe6e6;
        border-left: 4px solid #ff4444;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .safe-alert {
        background: #e6ffe6;
        border-left: 4px solid #44ff44;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stTab {
        background: white;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

class JobFraudDetector:
    """
    A comprehensive job fraud detection system using machine learning
    """
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove phone numbers
        text = re.sub(r'\d{3}-\d{3}-\d{4}|\(\d{3}\)\s*\d{3}-\d{4}', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, df):
        """Extract engineered features from job postings"""
        features = pd.DataFrame()
        
        # Text length features
        features['title_length'] = df['title'].fillna('').str.len()
        features['description_length'] = df['description'].fillna('').str.len()
        features['requirements_length'] = df.get('requirements', pd.Series(['']*len(df))).fillna('').str.len()
        
        # Fraud indicators
        fraud_keywords = [
            'easy money', 'work from home', 'no experience', 'guaranteed income',
            'urgent hiring', 'immediate start', 'cash payment', 'wire transfer',
            'western union', 'money order', 'advance fee', 'processing fee',
            'registration fee', 'training fee', 'equipment fee', 'lottery',
            'inheritance', 'confidential', 'prince', 'beneficiary'
        ]
        
        # Count fraud keywords
        text_cols = ['title', 'description', 'requirements']
        for col in text_cols:
            if col in df.columns:
                text_data = df[col].fillna('').str.lower()
                features[f'{col}_fraud_keywords'] = text_data.apply(
                    lambda x: sum(1 for keyword in fraud_keywords if keyword in x)
                )
        
        # Suspicious patterns
        features['has_urgency'] = df['description'].fillna('').str.contains(
            r'urgent|immediate|asap|right away|now', case=False, regex=True
        ).astype(int)
        
        features['has_money_mention'] = df['description'].fillna('').str.contains(
            r'\$\d+|\d+\s*dollars|money|payment|salary|income', case=False, regex=True
        ).astype(int)
        
        features['has_contact_info'] = df['description'].fillna('').str.contains(
            r'contact|call|email|phone|whatsapp', case=False, regex=True
        ).astype(int)
        
        # Company features
        features['company_missing'] = (df.get('company', pd.Series(['']*len(df))) == '').astype(int)
        features['company_confidential'] = df.get('company', pd.Series(['']*len(df))).str.contains(
            'confidential|private|undisclosed', case=False, na=False
        ).astype(int)
        
        # Location features
        features['location_remote'] = df.get('location', pd.Series(['']*len(df))).str.contains(
            'remote|work from home|anywhere', case=False, na=False
        ).astype(int)
        
        # Fill any missing values
        features = features.fillna(0)
        
        return features
    
    def detect_fraud_simple(self, job_data):
        """Simplified fraud detection with proper DataFrame handling"""
        fraud_keywords = [
            'easy money', 'work from home', 'no experience needed', 'make money fast',
            'guaranteed income', 'urgent hiring', 'immediate start', 'cash payment',
            'wire transfer', 'western union', 'money order', 'advance fee',
            'lottery', 'inheritance', 'prince', 'beneficiary', 'confidential',
            'processing fee', 'registration fee', 'training fee', 'equipment fee'
        ]

        suspicious_patterns = [
            r'\$\d+\s*(per|\/)\s*(hour|day|week)',  # Unrealistic pay rates
            r'contact.*immediately',
            r'send.*money',
            r'personal.*information',
            r'bank.*details',
            r'social.*security'
        ]

        # Create a copy of the original dataframe
        results_df = job_data.copy()
        
        # Initialize result columns
        results_df['fraud_probability'] = 0.0
        results_df['prediction'] = 'Genuine'
        results_df['risk_level'] = 'Low'
        
        for idx, job in job_data.iterrows():
            fraud_score = 0
            description = str(job.get('description', '')).lower()
            title = str(job.get('title', '')).lower()
            company = str(job.get('company', '')).lower()
            location = str(job.get('location', '')).lower()
            requirements = str(job.get('requirements', '')).lower()

            # Check for fraud keywords
            all_text = f"{title} {description} {company} {requirements}".lower()
            for keyword in fraud_keywords:
                if keyword in all_text:
                    fraud_score += 0.3

            # Check for suspicious patterns
            for pattern in suspicious_patterns:
                if re.search(pattern, all_text, re.IGNORECASE):
                    fraud_score += 0.25

            # Additional heuristics
            if len(description) < 50:
                fraud_score += 0.2
            if '!!!' in title or '$$$' in title:
                fraud_score += 0.4
            if company in ['', 'n/a', 'confidential']:
                fraud_score += 0.3
            if location == 'remote' and fraud_score > 0.3:
                fraud_score += 0.2
            if not requirements or len(requirements) < 20:
                fraud_score += 0.15

            # Normalize score to probability
            fraud_probability = min(max(fraud_score, 0), 1)
            is_fraud = fraud_probability > 0.5

            # Update the results in the dataframe
            results_df.loc[idx, 'fraud_probability'] = round(fraud_probability, 2)
            results_df.loc[idx, 'prediction'] = 'Fraudulent' if is_fraud else 'Genuine'
            results_df.loc[idx, 'risk_level'] = 'High' if fraud_probability > 0.7 else 'Medium' if fraud_probability > 0.4 else 'Low'

        return results_df

def generate_sample_data():
    """Generate sample job data for demonstration"""
    sample_jobs = [
        {
            'title': 'Software Engineer',
            'company': 'Tech Corp',
            'location': 'San Francisco, CA',
            'description': 'Join our team to build scalable web applications using React and Node.js. 3+ years experience required.',
            'requirements': "Bachelor's degree in Computer Science, 3+ years React experience, strong problem-solving skills"
        },
        {
            'title': 'EASY MONEY!!! Work from home NOW!!!',
            'company': 'Confidential',
            'location': 'Remote',
            'description': 'Make $5000 per week working from home! No experience needed! Send $100 registration fee to get started immediately!',
            'requirements': 'None! Just send money!'
        },
        {
            'title': 'Data Analyst',
            'company': 'Analytics Inc',
            'location': 'New York, NY',
            'description': 'Analyze large datasets to drive business insights. Python, SQL, and statistical analysis experience required.',
            'requirements': "Master's degree preferred, 2+ years experience with Python/R, strong analytical skills"
        },
        {
            'title': 'URGENT! Money Processing Agent',
            'company': 'Global Finance Solutions',
            'location': 'Remote',
            'description': 'Process wire transfers and money orders from home. Guaranteed $3000 weekly income! Contact immediately!',
            'requirements': 'Must have bank account for processing payments'
        },
        {
            'title': 'Marketing Manager',
            'company': 'Brand Solutions LLC',
            'location': 'Chicago, IL',
            'description': 'Lead marketing campaigns for B2B clients. Develop strategies, manage budgets, and analyze performance metrics.',
            'requirements': '5+ years marketing experience, MBA preferred, strong communication skills'
        },
        {
            'title': 'Customer Service Representative',
            'company': 'ServicePlus Inc',
            'location': 'Austin, TX',
            'description': 'Handle customer inquiries and provide excellent service. Full training provided.',
            'requirements': 'High school diploma, good communication skills, customer service experience preferred'
        }
    ]
    
    return pd.DataFrame(sample_jobs)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Job Fraud Detection System</h1>
        <p>Protecting job seekers from fraudulent postings using AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize detector
    detector = JobFraudDetector()

    # Sidebar
    with st.sidebar:
        st.header("📋 Data Input Options")
        
        input_method = st.radio(
            "Choose input method:",
            ["Upload CSV File", "Use Sample Data", "Manual Entry"]
        )
        
        if input_method == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload your job dataset (CSV)",
                type=['csv'],
                help="CSV should contain columns: title, company, location, description, requirements"
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    # Ensure required columns exist
                    required_cols = ['title', 'description']
                    optional_cols = ['company', 'location', 'requirements']
                    
                    # Check for required columns
                    missing_required = [col for col in required_cols if col not in df.columns]
                    if missing_required:
                        st.error(f"Missing required columns: {missing_required}")
                    else:
                        # Add missing optional columns
                        for col in optional_cols:
                            if col not in df.columns:
                                df[col] = ''
                        
                        st.success(f"✅ Loaded {len(df)} job postings")
                        st.session_state['job_data'] = df
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        elif input_method == "Use Sample Data":
            if st.button("🎯 Load Sample Data"):
                df = generate_sample_data()
                st.session_state['job_data'] = df
                st.success(f"✅ Loaded {len(df)} sample job postings")
        
        elif input_method == "Manual Entry":
            st.subheader("Enter Job Details")
            
            title = st.text_input("Job Title*", placeholder="e.g., Software Engineer")
            company = st.text_input("Company", placeholder="e.g., Tech Corp")
            location = st.text_input("Location", placeholder="e.g., New York, NY")
            description = st.text_area("Job Description*", placeholder="Detailed job description...")
            requirements = st.text_area("Requirements", placeholder="Job requirements and qualifications...")
            
            if st.button("🔍 Analyze Single Job"):
                if title and description:
                    single_job = pd.DataFrame([{
                        'title': title,
                        'company': company,
                        'location': location,
                        'description': description,
                        'requirements': requirements
                    }])
                    st.session_state['job_data'] = single_job
                else:
                    st.error("Please fill in at least Title and Description")

    # Main content
    if 'job_data' in st.session_state:
        df = st.session_state['job_data']
        
        # Process the data
        with st.spinner("🔄 Analyzing job postings for fraud..."):
            df_results = detector.detect_fraud_simple(df)

        # Calculate statistics
        total_jobs = len(df_results)
        fraudulent_jobs = len(df_results[df_results['prediction'] == 'Fraudulent'])
        genuine_jobs = len(df_results[df_results['prediction'] == 'Genuine'])
        high_risk_jobs = len(df_results[df_results['risk_level'] == 'High'])

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Jobs", total_jobs, help="Total number of job postings analyzed")
        
        with col2:
            st.metric("Fraudulent", fraudulent_jobs, delta=f"{fraudulent_jobs/total_jobs*100:.1f}%", 
                     delta_color="inverse", help="Jobs classified as fraudulent")
        
        with col3:
            st.metric("Genuine", genuine_jobs, delta=f"{genuine_jobs/total_jobs*100:.1f}%", 
                     help="Jobs classified as genuine")
        
        with col4:
            st.metric("High Risk", high_risk_jobs, delta=f"{high_risk_jobs/total_jobs*100:.1f}%", 
                     delta_color="inverse", help="Jobs with high fraud risk")

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📋 Results Table", "⚠️ Suspicious Jobs", "📈 Analytics"])

        with tab1:
            # Dashboard view
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart for fraud vs genuine
                fig_pie = px.pie(
                    values=[genuine_jobs, fraudulent_jobs],
                    names=['Genuine', 'Fraudulent'],
                    title="Job Classification Distribution",
                    color_discrete_map={'Genuine': '#10B981', 'Fraudulent': '#EF4444'}
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Risk level distribution
                risk_counts = df_results['risk_level'].value_counts()
                fig_risk = px.bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    title="Risk Level Distribution",
                    color=risk_counts.index,
                    color_discrete_map={'Low': '#10B981', 'Medium': '#F59E0B', 'High': '#EF4444'}
                )
                fig_risk.update_layout(showlegend=False)
                st.plotly_chart(fig_risk, use_container_width=True)

        with tab2:
            # Results table
            st.subheader("📋 All Job Analysis Results")
            
            # Add styling based on prediction
            def style_prediction(val):
                if val == 'Fraudulent':
                    return 'background-color: #fee2e2; color: #dc2626;'
                else:
                    return 'background-color: #dcfce7; color: #16a34a;'
            
            def style_risk(val):
                if val == 'High':
                    return 'background-color: #fee2e2; color: #dc2626;'
                elif val == 'Medium':
                    return 'background-color: #fef3c7; color: #d97706;'
                else:
                    return 'background-color: #dcfce7; color: #16a34a;'
            
            # Display results - ensure all columns exist
            display_cols = []
            available_cols = ['title', 'company', 'location', 'prediction', 'fraud_probability', 'risk_level']
            
            for col in available_cols:
                if col in df_results.columns:
                    display_cols.append(col)
            
            if display_cols:
                try:
                    styled_df = df_results[display_cols].style\
                        .applymap(style_prediction, subset=['prediction'] if 'prediction' in display_cols else [])\
                        .applymap(style_risk, subset=['risk_level'] if 'risk_level' in display_cols else [])\
                        .format({'fraud_probability': '{:.2f}'} if 'fraud_probability' in display_cols else {})
                    
                    st.dataframe(styled_df, use_container_width=True, height=400)
                except Exception as e:
                    # Fallback to simple display if styling fails
                    st.dataframe(df_results[display_cols], use_container_width=True, height=400)
            
            # Download results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

        with tab3:
            # Suspicious jobs detail
            st.subheader("⚠️ Most Suspicious Job Postings")
            
            suspicious_jobs = df_results[df_results['prediction'] == 'Fraudulent'].sort_values(
                'fraud_probability', ascending=False
            )
            
            if len(suspicious_jobs) > 0:
                for idx, job in suspicious_jobs.iterrows():
                    with st.expander(f"🚨 {job['title']} - Risk Score: {job['fraud_probability']:.2f}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Company:** {job.get('company', 'N/A')}")
                            st.write(f"**Location:** {job.get('location', 'N/A')}")
                            st.write(f"**Risk Level:** {job['risk_level']}")
                        
                        with col2:
                            st.write(f"**Fraud Probability:** {job['fraud_probability']:.2f}")
                            st.write(f"**Classification:** {job['prediction']}")
                        
                        st.write("**Description:**")
                        st.write(job.get('description', 'N/A'))
                        
                        if job.get('requirements'):
                            st.write("**Requirements:**")
                            st.write(job['requirements'])
            else:
                st.success("🎉 No suspicious jobs found in this dataset!")

        with tab4:
            # Analytics
            st.subheader("📈 Fraud Detection Analytics")
            
            # Fraud probability distribution
            fig_hist = px.histogram(
                df_results,
                x='fraud_probability',
                nbins=20,
                title="Fraud Probability Distribution",
                labels={'fraud_probability': 'Fraud Probability', 'count': 'Number of Jobs'}
            )
            fig_hist.update_traces(marker_color='lightblue', marker_line_color='navy', marker_line_width=1)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Feature analysis
            st.subheader("📊 Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info("**Common Fraud Indicators Found:**")
                fraud_jobs = df_results[df_results['prediction'] == 'Fraudulent']
                if len(fraud_jobs) > 0:
                    # Analyze common patterns in fraudulent jobs
                    fraud_keywords = ['urgent', 'immediate', 'easy money', 'guaranteed', 'no experience']
                    found_keywords = []
                    
                    for keyword in fraud_keywords:
                        count = fraud_jobs['description'].str.contains(keyword, case=False, na=False).sum()
                        if count > 0:
                            found_keywords.append(f"• '{keyword}': {count} jobs")
                    
                    if found_keywords:
                        for keyword in found_keywords:
                            st.write(keyword)
                    else:
                        st.write("• No common fraud keywords detected in this sample")
            
            with col2:
                st.success("**Protection Tips:**")
                st.write("• Be wary of jobs requiring upfront payments")
                st.write("• Verify company information independently")
                st.write("• Avoid jobs with unrealistic salary promises")
                st.write("• Check for proper contact information")
                st.write("• Research the company's online presence")

    else:
        # Welcome screen
        st.markdown("""
        ## 🚀 Welcome to the Job Fraud Detection System
        
        This system helps identify potentially fraudulent job postings using advanced machine learning techniques.
        
        ### 🔍 How it works:
        1. **Upload your data** or use sample data
        2. **AI analyzes** each job posting for fraud indicators
        3. **Get results** with risk scores and classifications
        4. **Review suspicious** postings in detail
        
        ### 📊 Features:
        - Real-time fraud detection
        - Detailed risk analysis
        - Interactive visualizations
        - Downloadable results
        - Comprehensive reporting
        
        **Get started by selecting an input method from the sidebar!**
        """)
        
        # Show sample data structure
        st.subheader("📋 Expected Data Format")
        sample_format = pd.DataFrame({
            'title': ['Software Engineer', 'Marketing Manager'],
            'company': ['Tech Corp', 'Marketing Inc'],
            'location': ['San Francisco, CA', 'New York, NY'],
            'description': ['Join our team...', 'Lead marketing campaigns...'],
            'requirements': ['Bachelor degree...', '5+ years experience...']
        })
        st.dataframe(sample_format, use_container_width=True)
        
        st.info("💡 **Tip:** Your CSV file should contain these columns for best results. Missing columns will be handled gracefully.")

if __name__ == "__main__":
    main()
