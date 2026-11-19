# Task 3
# Credit Card Customer Churn Analysis | Descriptive Analysis

# Step 1 - Load and Inspect the Dataset
# 1. Load the data
# Type in R console - setwd(choose.dir())
raw <- read.csv("data/BankChurners.csv", na.strings = c("","NA"))

head(raw) # Shows the first 6 rows
dim(raw) # Shows the number of rows and columns
names(raw) # Shows the column names

# Step 2 - Check missing values and duplicates
# 2.1
colSums(is.na(raw)) #Missing values in each column
sum(is.na(raw)) #Missing values in the entire dataset

# 2.2 Check for duplicate rows and customer IDs
sum(duplicated(raw)) # Number of repeated rows

customer_ids <- raw$CLIENTNUM
sum(duplicated(customer_ids[!is.na(customer_ids)])) # Repeated non-missing IDs

# Step 2.3 - Count "Unknown" in each categorical column
categorical_cols <- c(
  "Attrition_Flag", "Gender", "Education_Level",
  "Marital_Status", "Income_Category", "Card_Category"
)

unknown_counts <- sapply(raw[categorical_cols], function(x){
  sum(x == "Unknown", na.rm= TRUE)
})

print(unknown_counts)

# Step 3 - Check for possible outliers using the IQR rule
# Step 3.1
numeric_cols <- c(
  "Customer_Age", "Dependent_count", "Months_on_book",
  "Total_Relationship_Count", "Months_Inactive_12_mon",
  "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
  "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
  "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"
)

count_outliers <- function(x){
  if(all(is.na(x))) return(NA_integer_) #Nothing to check
  
  q1 <- quantile(x, 0.25, na.rm = TRUE, names = FALSE)
  q3 <- quantile(x, 0.75, na.rm = TRUE, names = FALSE)
  iqr_value <- q3 - q1
  
  lower_limit <- q1 - 1.5*iqr_value
  upper_limit <- q3 + 1.5*iqr_value
  
  sum(x < lower_limit | x > upper_limit, na.rm = TRUE)
}

# Apply the function to each selected column
outlier_counts <- sapply(raw[numeric_cols], count_outliers)
print(outlier_counts)

# Step 4
# Step 4.1 - Count customers by churn status
status_counts <- table(raw$Attrition_Flag, useNA = "ifany")
print(status_counts)

# If the table shows only the two expected statuses, calculate percentages
status_percentages <- round(100* prop.table(status_counts), 2)
print(status_percentages)

#Show only the percentage of customers who left
print(status_percentages["Attrited Customer"])

# Step 4.2 - Compare average transaction counts by churn status
transaction_summary <- aggregate(
  Total_Trans_Ct ~ Attrition_Flag,
  data = raw,
  FUN = mean
)

transaction_summary$Total_Trans_Ct <- round(
  transaction_summary$Total_Trans_Ct, 2
)

print(transaction_summary)

# Step 4.3 - Compare transaction amounts and inactivity by churn status
behavior_summary <- aggregate(
  cbind(Total_Trans_Amt, Months_Inactive_12_mon) ~ Attrition_Flag,
  data = raw,
  FUN = mean
)

behavior_summary$Total_Trans_Amt <- round(behavior_summary$Total_Trans_Amt, 2)
behavior_summary$Months_Inactive_12_mon <- round(
  behavior_summary$Months_Inactive_12_mon, 2
)

print(behavior_summary)

# Step 4.4 - Compare transaction counts visually
boxplot(
  Total_Trans_Ct ~ Attrition_Flag,
  data = raw,
  main = "Transaction counts by churn status",
  xlab = "Customer status",
  ylab = "Transactions in the last 12 months",
  col = c("salmon", "lightblue")
)

# Step 4.5 - Compare churn rates by card category
card_counts <- table(raw$Card_Category, raw$Attrition_Flag)

card_totals <- rowSums(card_counts)
card_churned <- as.integer(card_counts[, "Attrited Customer"])

card_summary <- data.frame(
  Card_Category = rownames(card_counts),
  Total_Customers = card_totals,
  Churned_Customers = card_churned,
  Churn_Rate_Pct = round(100*card_churned/card_totals, 2)
)

print(card_summary, row.names = FALSE)

# Step 4.6 - Compare churn rates by other customer categories
churn_by_category <- function(category, status){
  counts <- table(category, status)
  totals <- rowSums(counts)
  churned <- as.integer(counts[, "Attrited Customer"])
  
  data.frame(
    Category = rownames(counts),
    Total_Customers = totals,
    Churned_Customers = churned,
    Churn_Rate_Pct = round(100*churned/totals, 2)
  )
}

# Try gender first
gender_summary <- churn_by_category(raw$Gender, raw$Attrition_Flag)
print(gender_summary, row.names = FALSE)

print(churn_by_category(raw$Income_Category, raw$Attrition_Flag), row.names = FALSE)
print(churn_by_category(raw$Education_Level, raw$Attrition_Flag), row.names = FALSE)
print(churn_by_category(raw$Marital_Status, raw$Attrition_Flag), row.names = FALSE)

# Step 4.7 - Correlation analysis for the 14 selected numeric variables
correlation_matrix <- cor(
  raw[numeric_cols],
  method = "pearson",
  use = "pairwise.complete.obs"
)

# Show the correlation between transaction count and transaction amount
transaction_correlation <- correlation_matrix["Total_Trans_Ct", "Total_Trans_Amt"]
print(round(transaction_correlation, 2))

# Step 5 - Key findings
#
# 1. Overall churn: 1,627 of 10,127 customers left (16.07%).
#
# 2. Transaction activity: Customers who left made 44.93 transactions
#    on average, compared with 68.67 for customers who stayed.
#
# 3. Inactivity: Customers who left averaged 2.69 inactive months,
#    compared with 2.27 months for customers who stayed.
#
# 4. Correlation: Transaction counts and total transaction amount had
#    a Pearson correlation of 0.81. Customers who made more transactions
#    tended to have higher total transaction amounts.
#
# Limitation: These results describe patterns in the dataset.
# They do not tell us what caused customers to leave.