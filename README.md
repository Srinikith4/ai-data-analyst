# AI Data Analyst Dashboard

An AI-powered data analytics dashboard built using Streamlit, Machine Learning, and Plotly.  
It allows users to explore and analyze shopping data using natural language queries.



# Overview

This project simulates an AI Data Analyst that understands queries such as:

- Top 5 products  
- Male footwear sales in winter  
- Which location has the highest revenue  
- Female clothing trends  

The system uses a combination of:
- Machine Learning (TF-IDF + Logistic Regression) for intent detection  
- Rule-based filtering for extracting conditions like gender, category, and season  

It then generates interactive visualizations to present insights.



# Features

- Interactive dashboard layout (similar to BI tools)
- ML-based query understanding
- Natural language filtering (gender, category, season)
- Dynamic top-N query handling (e.g., "top 5", "top 10")
- Interactive Plotly visualizations
- Sidebar filters for custom analysis
- Real-time data insights



# Tech Stack

- Python  
- Streamlit  
- Pandas  
- Plotly  
- Scikit-learn  



# Example Queries

You can try queries like:

- top 5 products  
- best selling category  
- male footwear in winter  
- female clothing sales  
- top locations  
- age distribution  
- which gender spends more  
- top 10 locations in summer  
- male accessories in winter  
- female footwear trends  



# Project Structure
ai-data-analyst/
│
├── app.py
├── shopping_trends.csv
├── README.md
└── requirements.txt


# Future Improvements

- Improve NLP understanding for complex queries  
- Expand training dataset for better accuracy  
- Add chat-style UI  
- Support multiple datasets


This project demonstrates a hybrid approach combining Machine Learning and rule-based logic for building a practical AI-driven analytics tool.



J.V.Sri Nikith
