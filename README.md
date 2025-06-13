# 🛡️ Job Fraud Detection System

An AI-powered system to detect fraudulent job postings and protect job seekers from scams.

## 🚀 Live Demo

**Try it now:** [Job Fraud Detector](https://job-fraud-detector2.streamlit.app/))

## 📋 Features

- **Real-time Analysis**: Upload CSV files or analyze individual job postings
- **AI-Powered Detection**: Advanced pattern recognition for fraud indicators
- **Interactive Dashboard**: Visual analytics and insights
- **Risk Assessment**: Jobs classified as High, Medium, or Low risk
- **Downloadable Reports**: Export results for further analysis

## 🔍 How It Works

1. **Data Input**: Upload job posting data or use sample data
2. **AI Analysis**: System analyzes text patterns, keywords, and suspicious indicators
3. **Risk Scoring**: Each job gets a fraud probability score (0-1)
4. **Classification**: Jobs are marked as Genuine or Fraudulent
5. **Insights**: View detailed analytics and fraud patterns

## 📊 Key Fraud Indicators

- Unrealistic salary promises
- Requests for upfront payments
- Vague job descriptions
- Urgency keywords ("immediate", "urgent")
- Missing company information
- Suspicious contact methods

## 🛠️ Technology Stack

- **Python**: Core logic and data processing
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning components
- **RegEx**: Pattern matching for fraud detection

## 📈 Sample Results

The system typically achieves:
- High accuracy in detecting obvious fraud patterns
- Low false positive rates for legitimate jobs
- Comprehensive risk assessment with explanations

## 🚀 Local Development

```bash
# Clone the repository
git clone https://github.com/your-username/job-fraud-detector.git
cd job-fraud-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📝 Data Format

Upload CSV files with these columns:
- `title`: Job title
- `company`: Company name
- `location`: Job location
- `description`: Detailed job description
- `requirements`: Job requirements and qualifications

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request



**Built for the DS-1 Hackathon Challenge** 🏆
