import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data():

    transaction_path = "../../data/raw/train_transaction.csv"
    identity_path = "../../data/raw/train_identity.csv"
    transaction_df = pd.read_csv(transaction_path)
    identity_df = pd.read_csv(identity_path)
    merged_dataset = pd.merge(transaction_df, identity_df, on="TransactionID", how="left")
    # print(f"Merge Dataset Columns {merged_dataset.shape[1]}")
    # print(f"Merge Dataset Rows {merged_dataset.shape[0]}")
    X = merged_dataset.drop(columns=['isFraud','TransactionID'])
    y = merged_dataset['isFraud']
    #Finding/Sorting the column of type categorical(other than numerical)
    categorical_columns = X.select_dtypes(include="object").columns
    #print(categorical_columns.tolist())

    #finding the missing values from categorical columns
    #categorical_missing = X[categorical_columns].isnull().sum()
    #print(categorical_missing)

    #Replacing NaN with MISSING in all categorical columns
    X[categorical_columns] = X[categorical_columns].fillna("MISSING")

    #Making the datatype as Category instead of object for Categorical Columns
    X[categorical_columns] = X[categorical_columns].astype("category")
    

    #splitting X and y into train and temp(temp because it will be further split into validation and testing)
    #30% for temp remaining all train. stratify so that distribution of isFraud is uniform.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )
    #further splitting X_temp and y_temp in testing and validation.
    # 0.5 means splitting y_temp in 2 equal parts of 50% each for test and validate.
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    # print(X_train.columns[422])
    return X_train, X_validation, X_test, y_train, y_validation, y_test

if __name__ == "__main__":
    preprocess_data()