import pandas as pd

transaction_path = "../../data/raw/train_transaction.csv"
identity_path = "../../data/raw/train_identity.csv"

transaction_df = pd.read_csv(transaction_path)
identity_df = pd.read_csv(identity_path)
print(f"Transaction Shape: {transaction_df.shape}")
print(f"identity Shape: {identity_df.shape}")


# no of total rows in transaction df
total_count = transaction_df['isFraud'].count()
print(f"Total Rows in Transaction : {total_count}")

#Count how many rows have isFraud = 1
fraud_count = transaction_df['isFraud'].value_counts()[1]
print(f"Fraud Count : {fraud_count}")

#Finding fraud rate in entire transaction_df. That means identifying total no of fraud cases in entire transaction datset
fraud_rate = fraud_count/total_count * 100
# print(f"Fraud Rate : {fraud_rate}")

#Finding out missing values in each column of transaction df
missing_values = transaction_df.isnull().sum()
#print(missing_values)

#Finding out missing value % in each column
missing_percentage = missing_values/total_count * 100
#print(f"Total Transaction Columns {transaction_df.shape[1]}")
#print(f"Missing Percentage : {missing_percentage}")

#Total columns with any no of missing value
# Diagnostic only:
# We are NOT automatically dropping columns based on missing percentage.
# High missingness may itself contain useful fraud-related information.

columns_with_missing_values =  (missing_percentage > 0).sum()
print(f"No of columns with missing values in Transaction : {columns_with_missing_values} / {transaction_df.shape[1]}")
columns_with_missing_values_gt_50 =  (missing_percentage > 50).sum()
print(f"No of columns with missing more than 50% values in Transaction: {columns_with_missing_values_gt_50} / {transaction_df.shape[1]}")
columns_with_missing_values_gt_80 =  (missing_percentage > 80).sum()
print(f"No of columns with missing more than 80% values in Transaction : {columns_with_missing_values_gt_80} /  {transaction_df.shape[1]}")
print(f"Total Duplicate TransactionId in Transaction {transaction_df['TransactionID'].duplicated().sum()} ")


print(f"Identity Shape: {identity_df.shape}")
total_count = identity_df.shape[0]
print(f"Total Rows in Identity : {total_count}")
print(f"Total Columns in Identity : {identity_df.shape[1]}")
missing_values = identity_df.isnull().sum()
#print(missing_values)
missing_percentage = missing_values/total_count * 100
#print(f"Missing Percentage : {missing_percentage}")

# Diagnostic only:
# We are NOT automatically dropping columns based on missing percentage.
# High missingness may itself contain useful fraud-related information.

columns_with_missing_values =  (missing_percentage > 0).sum()
print(f"No of columns with missing values in Identity : {columns_with_missing_values} / {identity_df.shape[1]}")
columns_with_missing_values_gt_50 =  (missing_percentage > 50).sum()
print(f"No of columns with missing more than 50% values in Identity: {columns_with_missing_values_gt_50} / {identity_df.shape[1]}")
columns_with_missing_values_gt_80 =  (missing_percentage > 80).sum()
print(f"No of columns with missing more than 80% values in Identity : {columns_with_missing_values_gt_80} /  {identity_df.shape[1]}")

#Finding out if transaction id is duplicated in identity
print(f"Total Duplicate TransactionId in Identity {identity_df['TransactionID'].duplicated().sum()} ")

#finding out common transaction id's in both dataset
common_transaction_id = set(transaction_df['TransactionID']).intersection( identity_df['TransactionID'])
print(f"Common TransactionId : {len(common_transaction_id)}")
print(f"Transactions without Identity : {transaction_df.shape[0]-len(common_transaction_id)}")
#result printed that all the identity transaction id's are present in transaction but not all transactions have identity

numeric_columns = transaction_df.select_dtypes(include="number")
number_of_numeric_columns_transaction = numeric_columns.shape[1]
#print(f"Number of Numeric Transaction Columns : {number_of_numeric_columns_transaction}")

categorical_columns = transaction_df.select_dtypes(include="object")
no_categorical_columns_transaction = categorical_columns.shape[1]
#print(f"Number of Categorical Transaction Columns : {no_categorical_columns_transaction}")

numeric_columns = identity_df.select_dtypes(include="number")
number_of_numeric_columns_identity = numeric_columns.shape[1]
#print(f"Number of Numeric Identity Columns : {number_of_numeric_columns_identity}")

categorical_columns = identity_df.select_dtypes(include="object")
no_categorical_columns_identity = categorical_columns.shape[1]
#print(f"Number of Categorical Identity Columns : {no_categorical_columns_identity}")