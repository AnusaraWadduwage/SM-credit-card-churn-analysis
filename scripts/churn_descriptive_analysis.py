# %% [markdown]
# **Task 3**

# %% [markdown]
# **Step 1: Load the DataSet**

# %%
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from pathlib import Path

csv_path = Path("data/BankChurners.csv")
if not csv_path.is_file():
    csv_path = Path("../data/BankChurners.csv")

df = pd.read_csv(csv_path)
df.head(10)

# %%
df.shape

# %% [markdown]
# The BankChurners dataset contains 10,127 customer records and 23 columns containing demographic, relationship, credit, and transaction-related information. The target variable is Attrition_Flag, which identifies existing and attrited customers.

# %% [markdown]
# **Basic dataset inspection**

# %%
df.info()

# %% [markdown]
# #Variable Description
# 
# 1.CLIENTNUM - Unique customer identifier
# 
# 2.Attrition_Flag - Customer status: existing/attrited
# 
# 3.Customer_Age - Customer age
# 
# 4.Gender - Customer gender
# 
# 5.Dependent_count - Number of dependents
# 
# 6.Education_Level - Customer's education category
# 
# 7.Marital_Status - Customer's marital status
# 
# 8.Income_Category - Customer's income category
# 
# 9.Card_Category - Credit card category
# 
# 10.Months_on_book - Duration of customer relationship
# 
# 11.Total_Relationship_Count - Number of products/relationships held
# 
# 12.Months_Inactive_12_mon - Number of inactive months in previous 12 months
# 
# 13.Contacts_Count_12_mon - Number of contacts in previous 12 months
# 
# 14.Credit_Limit - Credit limit
# 
# 15.Total_Revolving_Bal - Total revolving balance
# 
# 16.Avg_Open_To_Buy - Average available credit
# 
# 17.Total_Amt_Chng_Q4_Q1 - Change in transaction amount from Q1 to Q4
# 
# 18.Total_Trans_Amt - Total transaction amount
# 
# 19.Total_Trans_Ct - Total transaction count
# 
# 20.Total_Ct_Chng_Q4_Q1 - Change in transaction count from Q1 to Q4
# 
# 21.Avg_Utilization_Ratio - Average credit utilization ratio
# 
# Note: The final two Naive Bayes columns are pre-generated classifier-related variables and were not considered meaningful customer characteristics for the descriptive analysis.

# %%
df.describe()

# %% [markdown]
# CLIENTNUM is a unique customer identifier and was not treated as a meaningful analytical variable.

# %%
#Categorical variables
df.describe(include="object")

# %% [markdown]
# **Step 2: Data Quality**

# %%
#Missing values
df.isnull().sum()

# %%
#Total missing values
print("Total missing values:", df.isnull().sum().sum())

# %%
#Duplicate values
print("Number of duplicate rows:", df.duplicated().sum())

# %%
#Inspect category frequencies and identify "Unknown" values
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    print(f"\n{col}")
    print(df[col].value_counts())

# %%
# Count Unknown values in categorical variables

unknown_counts = {}

for col in categorical_cols:
    unknown_counts[col] = (df[col] == "Unknown").sum()

pd.Series(unknown_counts, name="Unknown_Count")

# %% [markdown]
# Data Quality Interpretation
# 
# No conventional missing (NaN) values or duplicate records were identified in
# the dataset. However, "Unknown" categories were found in Education_Level
# (1,519 records), Marital_Status (749 records), and Income_Category
# (1,112 records).
# 
# These records were retained because an unknown value in one demographic
# variable does not necessarily make the remaining customer information invalid.
# Removing these records could result in unnecessary loss of useful customer data.

# %% [markdown]
# **Step 3: Outlier Detection**

# %% [markdown]
# Step 3.1: Detect outliers using IQR method

# %%
# Select meaningful numerical variables for outlier analysis

numeric_cols = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

# %%
# Detect outliers using the IQR method

outlier_counts = {}

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_counts[col] = (
        (df[col] < lower_bound) | (df[col] > upper_bound)
    ).sum()

pd.Series(outlier_counts, name="Outlier_Count")

# %% [markdown]
# Step 3.2: Visualize potential outliers using boxplots

# %%
# Visualize numerical variables using boxplots

for col in numeric_cols:
    plt.figure(figsize=(7, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()

# %% [markdown]
# Step 3.3: Outlier Interpretation

# %% [markdown]
# The IQR method identified potential outliers in several numerical variables, including Customer_Age, Months_on_book, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, and Total_Ct_Chng_Q4_Q1. The boxplots were used to visually examine these extreme observations. These values were retained for the descriptive analysis because extreme values in financial and customer behavioural variables may represent genuine customer characteristics rather than data errors. Therefore, automatically removing them could result in the loss of meaningful information.

# %% [markdown]
# **Step 4: Descriptive Analysis**

# %% [markdown]
# 
# Step 4.1: Customer Attrition Distribution

# %%
#Target Variable
df["Attrition_Flag"].value_counts()

# %%
# Customer attrition percentages
df["Attrition_Flag"].value_counts(normalize=True).mul(100).round(2)

# %%
plt.figure(figsize=(7, 4))

sns.countplot(
    data=df,
    x="Attrition_Flag"
)

plt.title("Distribution of Customer Attrition")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")
plt.show()

# %% [markdown]
# The dataset contains 8,500 existing customers and 1,627 attrited customers. Approximately 16.07% of customers are attrited, compared with 83.93% existing customers. This indicates an imbalanced distribution of the target variable, with attrited customers forming the minority group.

# %% [markdown]
# Step 4.2: Numerical Variables by Customer Attrition Status

# %%
# Compare key numerical variables by attrition status

comparison_cols = [
    'Customer_Age',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Avg_Utilization_Ratio'
]

attrition_summary = df.groupby('Attrition_Flag')[comparison_cols].mean().round(2)

attrition_summary

# %% [markdown]
# Numerical Comparison: Attrited customers show lower average transaction counts (44.93 vs. 68.67), transaction amounts (3,095.03 vs. 4,654.66), revolving balances (672.82 vs. 1,256.60), and relationship counts (3.28 vs. 3.91) than existing customers. They also show higher average inactivity (2.69 vs. 2.27 months) and contact counts (2.97 vs. 2.36). These are descriptive differences and do not by themselves establish statistical significance.

# %% [markdown]
# Step 4.3: Visualize Key Differences

# %%
# Visualize key numerical differences by attrition status

key_variables = [
    'Total_Trans_Ct',
    'Total_Trans_Amt',
    'Total_Revolving_Bal',
    'Months_Inactive_12_mon',
    'Total_Relationship_Count'
]

for col in key_variables:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x='Attrition_Flag', y=col)
    plt.title(f'{col} by Customer Attrition Status')
    plt.xlabel('Customer Status')
    plt.ylabel(col)
    plt.show()

# %% [markdown]
# Attrited customers exhibit lower transaction activity, lower revolving balances, fewer relationships with the bank, and higher inactivity than existing customers. These patterns suggest potential associations with customer attrition, but inferential analysis is required to determine whether the observed differences are statistically significant.

# %% [markdown]
# Step 4.4: Categorical Variables vs Attrition

# %%
# Churn percentages within each categorical variable

categorical_features = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

for col in categorical_features:
    print(f"\n{col}")

    churn_percentage = pd.crosstab(
        df[col],
        df['Attrition_Flag'],
        normalize='index'
    ) * 100

    print(churn_percentage.round(2))

# %%
# Visualize categorical variables by attrition status

for col in categorical_features:
    plt.figure(figsize=(8, 4))

    sns.countplot(
        data=df,
        x=col,
        hue='Attrition_Flag'
    )

    plt.title(f'Customer Attrition by {col}')
    plt.xlabel(col)
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# Categorical Comparison: Descriptive differences in attrition rates are observed across several customer categories. Female customers show a higher attrition rate (17.36%) than male customers (14.62%). Attrition also varies across education, income, marital-status, and card categories. For example, the observed attrition rate is 25.00% among Platinum card customers. However, category sizes differ substantially, so these percentages should be interpreted cautiously and inferential analysis is required before concluding that these characteristics are significantly associated with attrition.

# %% [markdown]
# Step 4.5: Correlation Analysis

# %%
# Correlation matrix for numerical variables
correlation_matrix = df[numeric_cols].corr()

# Correlation heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()
plt.show()

# %% [markdown]
# Correlation Analysis: The correlation matrix indicates several notable relationships among the numerical variables. Total transaction amount and total transaction count show a strong positive correlation (r = 0.81), indicating that customers who make more transactions generally have higher transaction amounts. Customer age and months on book are also strongly positively correlated (r = 0.79). Credit limit and average open-to-buy show an extremely strong positive relationship (r ≈ 1.00). These correlations describe associations between variables and do not imply causation.

# %% [markdown]
# **Step 5: Initial Business Insights**

# %% [markdown]
# Initial Business Insights
# 
# 1.
# Customer attrition is a notable retention concern: Out of 10,127 customers, 1,627 (16.07%) are attrited customers, while 83.93% remain existing customers.
# 
# 2.
# Lower transaction activity is associated with attrition: Attrited customers recorded a lower average transaction count (44.93) than existing customers (68.67). Their average total transaction amount was also lower (3,095.03 compared with 4,654.66).
# 
# 3.
# Customer inactivity may be an important warning indicator: Attrited customers were inactive for an average of 2.69 months during the previous 12 months, compared with 2.27 months among existing customers.
# 
# 4.
# Attrited customers tend to have fewer relationships with the bank: The average total relationship count was 3.28 for attrited customers compared with 3.91 for existing customers, suggesting that customers with fewer banking relationships may require greater retention attention.
# 
# 5.
# Credit usage behaviour differs between the two groups: Attrited customers had a lower average revolving balance (672.82) than existing customers (1,256.60) and a lower average utilization ratio (0.16 compared with 0.30).
# 
# 6.
# Customer contact patterns also differ: Attrited customers had a higher average number of contacts during the previous 12 months (2.97) than existing customers (2.36), which may help identify customers requiring retention attention.

# %% [markdown]
# Overall Business Implication: The descriptive analysis indicates that lower transaction activity, greater inactivity, fewer banking relationships, lower credit utilization, and higher contact frequency are associated with customer attrition. These characteristics may help the bank identify potentially vulnerable customer groups and guide retention strategies. However, these findings are descriptive associations and should be validated through inferential statistical analysis and predictive modelling before making stronger conclusions.

# %%



