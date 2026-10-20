# customer-churn-intelligence-platform


- Connect SQL database to Python – pandas and sqlite3
- Data import using SQL query in Python – pandas and sqlite3
- Data Cleaning – numpy and pandas
- Data types, rename cols, select specific cols, QCs, handle missing/null values
- Feature Engineering – numpy and pandas
- Create new calculated cols, data transformation, use filters
- Data Analysis – numpy and pandas
- EDA - aggregation, group by, pivot table
- Data Visualization – matplotlib and seaborn
- Insights on ppt – key insights, next steps, impact to show in resume

<img width="1377" height="700" alt="Screenshot 2026-10-20 104707" src="https://github.com/user-attachments/assets/30534fd1-92c3-4d1e-8025-5fe769408afd" />

### Insights
Insights
- Churn Rate: 28.6% | Retention Rate: 71.4%
- Most of the churn is from basic subscription plan – nothing to worry in terms of major revenue impact
- Most of the churn happened in the month of Sep 2024 and, most affected state is Karnataka
- Average Tenure (Days): 1,451 | ARPU is Rs 18.8
- Total Revenue = 395
- Revenue loss due to churn = 74 | CLTV Lost = 2,047
- % Revenue loss = 18%
- monthly vs annual churn = 55.6% vs 8.3%
Action Items
• Check what happened in Karnataka – was there any increase in price, user complaints, tech issues, etc
• Did we increased subs price for Basic plan or recently made any changes – specially in sep month?
• Check what competitor is doing – bcoz one user has moved to competitor
• Focus on customers with ‘High’ & ‘Med’ churn risk, check their LTV (to make priority list), complaint request, reach out to
them through email, sms, calls and fix their issue

## Portfolio Project - Summary
• Churn & Revenue Impact (general summary)
Engineered an end-to-end churn analytics pipeline for an OTT subscription platform by integrating multi-table subscriber
data across acquisition type, contract structure, and plan tier (20+ KPIs). Uncovered a significant churn disparity between
monthly and annual contract segments, quantified MRR leakage and CLTV erosion attributed to high-risk cohorts, and
delivered a data-backed contract-migration retention strategy to reduce involuntary subscriber loss.
• Risk Scoring & Segmentation (Risk Analysis & Segmentation)
Developed a multi-dimensional churn risk scoring model by synthesising subscription tenure, plan type, and support
escalation signals across three relational tables (20+ KPIs), segmented the customer base into risk tiers using composite
churn scores, exposed a significant lifetime value gap between churned and retained cohorts, and recommended
prioritising Premium annual-plan retention over Basic monthly acquisition to maximise long-term revenue yield.
• Support Intelligence & Cancellation Analysis (working on live data)
Performed cross-functional support- churn correlation analysis by joining complaint, escalation, and CSAT data with
subscription records, identified that escalated support interactions were disproportionately concentrated among churned
customers, decomposed cancellation drivers into competitor switching, pricing sensitivity, and content dissatisfaction, and
translated findings into a prioritised product and pricing roadmap presented to the platform's growth team. 
