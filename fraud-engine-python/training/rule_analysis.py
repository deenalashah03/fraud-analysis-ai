import json

import pandas as pd

from models.xgboost.xgboost_model import XGBoostModel
from training.preprocess import preprocess_data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "models" / "xgboost" / "fraud_xgboost_config.json"
MODEL_PATH = BASE_DIR / "models"/ "xgboost"/"fraud_xgboost.json"



AMOUNT_BINS = [
    0,
    50,
    100,
    200,
    500,
    1000,
    float("inf")
]

AMOUNT_LABELS = [
    "0-50",
    "50-100",
    "100-200",
    "200-500",
    "500-1000",
    "1000+"
]


# ==========================================================
# RULE-SELECTION HISTORY / DECISION NOTES
# ==========================================================

"""
Rule-selection history (exploratory analysis; not runtime rule logic):

1. Product + amount candidates were compared using transaction counts,
   observed fraud rates, and XGBoost probability summaries. Product H +
   $200-$500 was retained as a candidate/runtime signal. Other explored
   combinations included Product H + $100-$200, Product R + $200-$500,
   Product C + $100-$200, Product C + $200-$500, Product S + $50-$100,
   and Product W + $500-$1,000; do not infer a rejection rationale beyond
   the measured outputs unless those outputs are reviewed.

2. Product C and missing addr1 were examined jointly because they overlap
   heavily. On the validation split, 9,716 transactions matched both; this
   group had a 12.10% observed fraud rate. Product C with addr1 present had
   513 transactions and a 7.41% rate; non-Product-C with missing addr1 had
   96 transactions and a 2.08% rate. The combined Product C + missing-address
   condition was selected instead of using either broad condition alone.

3. Email-domain analysis compared purchaser (P_emaildomain) and recipient
   (R_emaildomain) domains. On validation, P_emaildomain=outlook.com matched
   749 transactions (10.55% fraud); R_emaildomain=outlook.com matched 360
   (18.06%); either field being outlook.com matched 788 (10.03%). The combined
   Outlook condition was retained. R_emaildomain=gmail.com remains a candidate
   to evaluate for incremental coverage rather than being assumed useful.

4. Device/identity candidates included Android Chrome, Android OS, and iOS.
   Validation summaries: Android Chrome 3,090 transactions (13.53% fraud),
   Android OS 981 (9.17%), and iOS 2,913 (6.28%). Their incremental value must
   be checked against the current rule set and against one another.

5. card6=credit matched 22,259 validation transactions (6.62% fraud). It is
   retained as an analysis candidate only; broad coverage alone does not prove
   that it adds enough distinct value to become a runtime rule.

6. Rule overlap is measured explicitly before selecting additional rules.
   Candidate coverage and fraud rates are evidence for investigation signals,
   not proof that any individual transaction is fraudulent.

These notes preserve the currently known rationale. Refresh numerical results
by running the analysis against the current validation split; do not use the
validation split to tune the final reported test performance.
"""


def add_amount_bucket(df):
    """
    Add transaction amount bucket used by
    candidate rule analysis.
    """

    df = df.copy()

    df["amount_bucket"] = pd.cut(
        df["TransactionAmt"],
        bins=AMOUNT_BINS,
        labels=AMOUNT_LABELS
    )

    return df


# ==========================================================
# PRODUCT + AMOUNT ANALYSIS
# ==========================================================

def analyze_rules_against_model(
        X,
        y,
        model,
        dataset_name
):
    """
    Compare candidate deterministic risk patterns
    against actual fraud labels and XGBoost predictions.

    Exploratory analysis only.
    Does not create runtime rules.
    """

    print(
        f"\n========== PRODUCT + AMOUNT ANALYSIS - "
        f"{dataset_name} =========="
    )

    probabilities = model.predict_proba(X)

    analysis = X.copy()

    analysis["isFraud"] = y.values

    analysis["xgb_probability"] = probabilities

    analysis = add_amount_bucket(analysis)

    rules = {

        "Product C": (
                analysis["ProductCD"] == "C"
        ),

        "Product H + 100-200": (
                (analysis["ProductCD"] == "H") &
                (analysis["amount_bucket"] == "100-200")
        ),

        "Product H + 200-500": (
                (analysis["ProductCD"] == "H") &
                (analysis["amount_bucket"] == "200-500")
        ),

        "Product R + 200-500": (
                (analysis["ProductCD"] == "R") &
                (analysis["amount_bucket"] == "200-500")
        ),

        "Product C + 100-200": (
                (analysis["ProductCD"] == "C") &
                (analysis["amount_bucket"] == "100-200")
        ),

        "Product C + 200-500": (
                (analysis["ProductCD"] == "C") &
                (analysis["amount_bucket"] == "200-500")
        ),

        "Product S + 50-100": (
                (analysis["ProductCD"] == "S") &
                (analysis["amount_bucket"] == "50-100")
        ),

        "Product W + 500-1000": (
                (analysis["ProductCD"] == "W") &
                (analysis["amount_bucket"] == "500-1000")
        )
    }

    results = []

    for rule_name, condition in rules.items():

        rule_data = analysis[condition]

        if len(rule_data) == 0:
            continue

        results.append({

            "rule": rule_name,

            "transactions": len(rule_data),

            "fraud_transactions": (
                rule_data["isFraud"].sum()
            ),

            "actual_fraud_rate": (
                    rule_data["isFraud"].mean() * 100
            ),

            "avg_xgb_probability": (
                    rule_data["xgb_probability"].mean() * 100
            ),

            "median_xgb_probability": (
                    rule_data["xgb_probability"].median() * 100
            ),

            "xgb_predicted_fraud_rate": (
                    (
                            rule_data["xgb_probability"] >= 0.4
                    ).mean() * 100
            )
        })

    result_df = pd.DataFrame(results)

    print("\nCandidate rule comparison:")

    print(
        result_df.to_string(index=False)
    )

    return result_df


# ==========================================================
# EMAIL DOMAIN ANALYSIS
# ==========================================================

def analyze_email_domains(
        X,
        y,
        model,
        dataset_name
):
    """
    Analyze email-domain patterns.

    Exploratory analysis only.
    Does not create runtime rules.
    """

    print(
        f"\n========== EMAIL DOMAIN ANALYSIS - "
        f"{dataset_name} =========="
    )

    analysis = X.copy()

    analysis["isFraud"] = y.values

    analysis["xgb_probability"] = (
        model.predict_proba(X)
    )

    email_columns = [
        "P_emaildomain",
        "R_emaildomain"
    ]

    for column in email_columns:

        print(f"\n---------- {column} ----------")

        result = (
            analysis
            .groupby(
                column,
                dropna=False,
                observed=False
            )
            .agg(
                transactions=("isFraud", "count"),
                fraud_transactions=("isFraud", "sum"),
                fraud_rate=("isFraud", "mean"),
                avg_xgb_probability=(
                    "xgb_probability",
                    "mean"
                )
            )
            .reset_index()
        )

        result["fraud_rate"] *= 100

        result["avg_xgb_probability"] *= 100

        result = result[
            result["transactions"] >= 100
            ]

        result = result.sort_values(
            by="fraud_rate",
            ascending=False
        )

        print(
            "\nFraud rate by email domain "
            "(minimum 100 transactions):"
        )

        print(
            result.to_string(index=False)
        )


# ==========================================================
# CARD4 ANALYSIS
# ==========================================================

def analyze_card4(
        X,
        y,
        model,
        dataset_name
):
    """
    Analyze fraud patterns by card network/type.

    Exploratory analysis only.
    Does not create runtime rules.
    """

    print(
        f"\n========== CARD4 ANALYSIS - "
        f"{dataset_name} =========="
    )

    analysis = X.copy()

    analysis["isFraud"] = y.values

    analysis["xgb_probability"] = (
        model.predict_proba(X)
    )

    result = (
        analysis
        .groupby(
            "card4",
            dropna=False,
            observed=False
        )
        .agg(
            transactions=("isFraud", "count"),
            fraud_transactions=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            avg_xgb_probability=(
                "xgb_probability",
                "mean"
            )
        )
        .reset_index()
    )

    result["fraud_rate"] *= 100

    result["avg_xgb_probability"] *= 100

    result = result[
        result["transactions"] >= 100
        ]

    result = result.sort_values(
        by="fraud_rate",
        ascending=False
    )

    print(
        "\nFraud rate by card4 "
        "(minimum 100 transactions):"
    )

    print(
        result.to_string(index=False)
    )


# ==========================================================
# CARD6 ANALYSIS
# ==========================================================

def analyze_card6(
        X,
        y,
        model,
        dataset_name
):
    """
    Analyze fraud patterns by card type.

    Exploratory analysis only.
    Does not create runtime rules.
    """

    print(
        f"\n========== CARD6 ANALYSIS - "
        f"{dataset_name} =========="
    )

    analysis = X.copy()

    analysis["isFraud"] = y.values

    analysis["xgb_probability"] = (
        model.predict_proba(X)
    )

    result = (
        analysis
        .groupby(
            "card6",
            dropna=False,
            observed=False
        )
        .agg(
            transactions=("isFraud", "count"),
            fraud_transactions=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            avg_xgb_probability=(
                "xgb_probability",
                "mean"
            )
        )
        .reset_index()
    )

    result["fraud_rate"] *= 100

    result["avg_xgb_probability"] *= 100

    result = result[
        result["transactions"] >= 100
        ]

    result = result.sort_values(
        by="fraud_rate",
        ascending=False
    )

    print(
        "\nFraud rate by card6 "
        "(minimum 100 transactions):"
    )

    print(
        result.to_string(index=False)
    )


# ==========================================================
# FUTURE ANALYSES
# ==========================================================

# ----------------------------------------------------------
# ADDR1 ANALYSIS
# ----------------------------------------------------------

def analyze_addr1(X, y, model, dataset_name):
    print(f"\n========== ADDR1 ANALYSIS - {dataset_name} ==========")

    analysis = X.copy()
    analysis["isFraud"] = y.values
    analysis["xgb_probability"] = model.predict_proba(X)

    result = (
        analysis.groupby("addr1", dropna=False, observed=False)
        .agg(
            transactions=("isFraud", "count"),
            fraud_transactions=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            avg_xgb_probability=("xgb_probability", "mean")
        )
        .reset_index()
    )

    result["fraud_rate"] *= 100
    result["avg_xgb_probability"] *= 100

    result = result[result["transactions"] >= 100]

    result = result.sort_values(
        by="fraud_rate",
        ascending=False
    )

    print("\nFraud rate by addr1 (minimum 100 transactions):")
    print(result.to_string(index=False))


# ----------------------------------------------------------
# DISTANCE ANALYSIS
# ----------------------------------------------------------

def analyze_dist1(X, y, model, dataset_name):
    print(f"\n========== DIST1 ANALYSIS - {dataset_name} ==========")

    analysis = X.copy()
    analysis["isFraud"] = y.values
    analysis["xgb_probability"] = model.predict_proba(X)

    result = (
        analysis.groupby("dist1", dropna=False, observed=False)
        .agg(
            transactions=("isFraud", "count"),
            fraud_transactions=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            avg_xgb_probability=("xgb_probability", "mean")
        )
        .reset_index()
    )

    result["fraud_rate"] *= 100
    result["avg_xgb_probability"] *= 100

    result = result[result["transactions"] >= 100]

    result = result.sort_values(
        by="fraud_rate",
        ascending=False
    )

    print("\nFraud rate by dist1 (minimum 100 transactions):")
    print(result.to_string(index=False))

def analyze_dist2(X, y, model, dataset_name):
    print(f"\n========== DIST2 ANALYSIS - {dataset_name} ==========")

    analysis = X.copy()
    analysis["isFraud"] = y.values
    analysis["xgb_probability"] = model.predict_proba(X)

    result = (
        analysis.groupby("dist2", dropna=False, observed=False)
        .agg(
            transactions=("isFraud", "count"),
            fraud_transactions=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            avg_xgb_probability=("xgb_probability", "mean")
        )
        .reset_index()
    )

    result["fraud_rate"] *= 100
    result["avg_xgb_probability"] *= 100

    result = result[result["transactions"] >= 100]

    result = result.sort_values(
        by="fraud_rate",
        ascending=False
    )

    print("\nFraud rate by dist2 (minimum 100 transactions):")
    print(result.to_string(index=False))

# ----------------------------------------------------------
# DEVICE ANALYSIS
# ----------------------------------------------------------

def analyze_device_info(X, y, model, dataset_name):
    print(f"\n========== DEVICEINFO ANALYSIS - {dataset_name} ==========")

    analysis = X.copy()
    analysis["isFraud"] = y.values
    analysis["xgb_probability"] = model.predict_proba(X)

    result = (
        analysis.groupby("DeviceInfo", dropna=False, observed=False)
        .agg(
            transactions=("isFraud", "count"),
            fraud_transactions=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            avg_xgb_probability=("xgb_probability", "mean")
        )
        .reset_index()
    )

    result["fraud_rate"] *= 100
    result["avg_xgb_probability"] *= 100

    result = result[result["transactions"] >= 100]

    result = result.sort_values(
        by="fraud_rate",
        ascending=False
    )

    print("\nFraud rate by DeviceInfo (minimum 100 transactions):")
    print(result.to_string(index=False))

# ----------------------------------------------------------
# IDENTITY / BROWSER / OS ANALYSIS
# ----------------------------------------------------------

def analyze_id31(X, y, model, dataset_name):
    print(f"\n========== ID_31 ANALYSIS - {dataset_name} ==========")

    analysis = X.copy()
    analysis["isFraud"] = y.values
    analysis["xgb_probability"] = model.predict_proba(X)

    result = (
        analysis.groupby("id_31", dropna=False, observed=False)
        .agg(
            transactions=("isFraud", "count"),
            fraud_transactions=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            avg_xgb_probability=("xgb_probability", "mean")
        )
        .reset_index()
    )

    result["fraud_rate"] *= 100
    result["avg_xgb_probability"] *= 100

    result = result[result["transactions"] >= 100]

    result = result.sort_values(
        by="fraud_rate",
        ascending=False
    )

    print("\nFraud rate by id_31 (minimum 100 transactions):")
    print(result.to_string(index=False))

def analyze_id31_grouped(X, y, model, dataset_name):
    print(f"\n========== ID_31 GROUPED ANALYSIS - {dataset_name} ==========")

    analysis = X.copy()
    analysis["isFraud"] = y.values
    analysis["xgb_probability"] = model.predict_proba(X)

    def group_browser(browser):
        if pd.isna(browser):
            return "MISSING"

        browser = str(browser).lower()

        if "android" in browser and "chrome" in browser:
            return "ANDROID_CHROME"

        if "chrome" in browser:
            return "CHROME"

        if "firefox" in browser:
            return "FIREFOX"

        if "safari" in browser:
            return "SAFARI"

        if "edge" in browser:
            return "EDGE"

        if "ie " in browser or browser.startswith("ie"):
            return "INTERNET_EXPLORER"

        return "OTHER"

    analysis["browser_group"] = analysis["id_31"].apply(group_browser)

    result = (
        analysis.groupby("browser_group", observed=False)
        .agg(
            transactions=("isFraud", "count"),
            fraud_transactions=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            avg_xgb_probability=("xgb_probability", "mean")
        )
        .reset_index()
    )

    result["fraud_rate"] *= 100
    result["avg_xgb_probability"] *= 100

    result = result[result["transactions"] >= 100]

    result = result.sort_values(
        by="fraud_rate",
        ascending=False
    )

    print("\nFraud rate by browser group (minimum 100 transactions):")
    print(result.to_string(index=False))

def analyze_id30(X, y, model, dataset_name):
    print(f"\n========== ID_30 ANALYSIS - {dataset_name} ==========")

    analysis = X.copy()
    analysis["isFraud"] = y.values
    analysis["xgb_probability"] = model.predict_proba(X)

    result = (
        analysis.groupby("id_30", dropna=False, observed=False)
        .agg(
            transactions=("isFraud", "count"),
            fraud_transactions=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            avg_xgb_probability=("xgb_probability", "mean")
        )
        .reset_index()
    )

    result["fraud_rate"] *= 100
    result["avg_xgb_probability"] *= 100

    result = result[result["transactions"] >= 100]

    result = result.sort_values(
        by="fraud_rate",
        ascending=False
    )

    print("\nFraud rate by id_30 (minimum 100 transactions):")
    print(result.to_string(index=False))

def analyze_id30_grouped(X, y, model, dataset_name):
    print(f"\n========== ID_30 GROUPED ANALYSIS - {dataset_name} ==========")

    analysis = X.copy()
    analysis["isFraud"] = y.values
    analysis["xgb_probability"] = model.predict_proba(X)

    def group_os(os):
        if pd.isna(os):
            return "MISSING"

        os = str(os).lower()

        if "android" in os:
            return "ANDROID"

        if "ios" in os:
            return "IOS"

        if "windows" in os:
            return "WINDOWS"

        if "mac os" in os or os == "mac":
            return "MACOS"

        if "linux" in os:
            return "LINUX"

        return "OTHER"

    analysis["os_group"] = analysis["id_30"].apply(group_os)

    result = (
        analysis.groupby("os_group", observed=False)
        .agg(
            transactions=("isFraud", "count"),
            fraud_transactions=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            avg_xgb_probability=("xgb_probability", "mean")
        )
        .reset_index()
    )

    result["fraud_rate"] *= 100
    result["avg_xgb_probability"] *= 100

    result = result[result["transactions"] >= 100]

    result = result.sort_values(
        by="fraud_rate",
        ascending=False
    )

    print("\nFraud rate by OS group (minimum 100 transactions):")
    print(result.to_string(index=False))

def analyze_rule_overlap(X, y, dataset_name):
    print(
        f"\n========== RULE OVERLAP ANALYSIS - "
        f"{dataset_name} =========="
    )

    analysis = X.copy()
    analysis["isFraud"] = y.values

    analysis = add_amount_bucket(analysis)

    rules = {
        "PRODUCT_C": (
                analysis["ProductCD"] == "C"
        ),

        "PRODUCT_H_200_500": (
                (analysis["ProductCD"] == "H")
                &
                (analysis["amount_bucket"] == "200-500")
        ),

        "CREDIT_CARD": (
                analysis["card6"] == "credit"
        ),

        "MISSING_ADDRESS": (
            analysis["addr1"].isna()
        ),

        "P_EMAIL_OUTLOOK": (
                analysis["P_emaildomain"] == "outlook.com"
        ),

        "R_EMAIL_OUTLOOK": (
                analysis["R_emaildomain"] == "outlook.com"
        ),

        "R_EMAIL_ICLOUD": (
                analysis["R_emaildomain"] == "icloud.com"
        ),

        "R_EMAIL_GMAIL": (
                analysis["R_emaildomain"] == "gmail.com"
        ),

        "ANDROID_CHROME": (
                analysis["id_31"]
                .astype("string")
                .str.lower()
                .str.contains("chrome", na=False)
                &
                analysis["id_31"]
                .astype("string")
                .str.lower()
                .str.contains("android", na=False)
        ),

        "ANDROID_OS": (
            analysis["id_30"]
            .astype("string")
            .str.lower()
            .str.contains(
                "android",
                na=False
            )
        ),

        "IOS": (
            analysis["id_30"]
            .astype("string")
            .str.lower()
            .str.contains(
                "ios",
                na=False
            )
        )
    }

    rule_names = list(rules.keys())

    print("\n---------- Rule Coverage ----------")

    for rule_name, condition in rules.items():

        matching = analysis[condition]

        if len(matching) == 0:
            continue

        fraud_rate = (
                matching["isFraud"].mean() * 100
        )

        print(
            f"{rule_name:<25}"
            f" transactions={len(matching):<7}"
            f" fraud_rate={fraud_rate:.2f}%"
        )

    print("\n---------- Rule Overlap ----------")

    for i in range(len(rule_names)):

        for j in range(i + 1, len(rule_names)):

            rule_a = rule_names[i]
            rule_b = rule_names[j]

            overlap = (
                    rules[rule_a]
                    & rules[rule_b]
            )

            overlap_count = overlap.sum()

            if overlap_count == 0:
                continue

            print(
                f"{rule_a:<25}"
                f" + {rule_b:<25}"
                f" overlap={overlap_count}"

            )
    print("\n---------- PRODUCT C vs MISSING ADDRESS ----------")
    product_c = rules["PRODUCT_C"]
    missing_address = rules["MISSING_ADDRESS"]
    groups = {
        "PRODUCT_C_AND_MISSING_ADDRESS": ( product_c & missing_address ),
        "PRODUCT_C_AND_ADDRESS_PRESENT": ( product_c & ~missing_address ),
        "NON_PRODUCT_C_AND_MISSING_ADDRESS": ( ~product_c & missing_address ),
        "NON_PRODUCT_C_AND_ADDRESS_PRESENT": ( ~product_c & ~missing_address ) }
    for group_name, condition in groups.items():
        matching = analysis[condition]
        transaction_count = len(matching)
        fraud_count = int(matching["isFraud"].sum())
        fraud_rate = ( matching["isFraud"].mean() * 100
                       if transaction_count > 0
                       else 0.0 )
    print(
        f"{group_name:<40}"
        f" transactions={transaction_count:<7}"
        f" fraud_count={fraud_count:<6}"
        f" fraud_rate={fraud_rate:.2f}%"
    )


def analyze_incremental_coverage(X, y, dataset_name):
    """
    Compare candidate rules with the combined coverage of the three current
    rules. Each candidate is compared independently with the same baseline;
    candidate-to-candidate overlap is not removed from these individual rows.

    This is exploratory analysis only and does not add runtime rules.
    Pass the full, unfiltered feature set so rule-only columns remain available.
    """
    print(
        f"\n========== INCREMENTAL RULE COVERAGE - "
        f"{dataset_name} =========="
    )

    analysis = X.copy()
    analysis["isFraud"] = pd.Series(y, index=X.index).to_numpy()
    analysis = add_amount_bucket(analysis)

    current_rules = {
        "PRODUCT_C_AND_MISSING_ADDRESS": (
                (analysis["ProductCD"] == "C") & analysis["addr1"].isna()
        ),
        "PRODUCT_H_200_500": (
                (analysis["ProductCD"] == "H")
                & (analysis["amount_bucket"] == "200-500")
        ),
        "OUTLOOK_EMAIL_DOMAIN": (
                (analysis["P_emaildomain"] == "outlook.com")
                | (analysis["R_emaildomain"] == "outlook.com")
        ),
    }
    baseline_covered = pd.Series(False, index=analysis.index)
    for condition in current_rules.values():
        baseline_covered |= condition.fillna(False)

    candidates = {
        "R_EMAIL_GMAIL": (analysis["R_emaildomain"] == "gmail.com"),
        "ANDROID_CHROME": (
                analysis["id_31"].astype("string").str.lower().str.contains("chrome", na=False)
                & analysis["id_31"].astype("string").str.lower().str.contains("android", na=False)
        ),
        "ANDROID_OS": (
            analysis["id_30"].astype("string").str.lower().str.contains("android", na=False)
        ),
        "IOS": (
            analysis["id_30"].astype("string").str.lower().str.contains("ios", na=False)
        ),
        "CREDIT_CARD": (analysis["card6"] == "credit"),
    }

    print("\nCurrent rule baseline:")
    baseline = analysis[baseline_covered]
    baseline_count = len(baseline)
    baseline_fraud_count = int(baseline["isFraud"].sum())
    baseline_rate = baseline["isFraud"].mean() * 100 if baseline_count else 0.0
    print(
        f"transactions={baseline_count}, "
        f"fraud_transactions={baseline_fraud_count}, "
        f"fraud_rate={baseline_rate:.2f}%"
    )

    results = []
    for rule_name, condition in candidates.items():
        condition = condition.fillna(False)
        matching = condition
        additional = condition & ~baseline_covered
        matched_data = analysis[matching]
        additional_data = analysis[additional]
        total_count = int(matching.sum())
        additional_count = int(additional.sum())
        additional_fraud_count = int(additional_data["isFraud"].sum())
        additional_fraud_rate = (
            additional_data["isFraud"].mean() * 100 if additional_count else 0.0
        )
        total_fraud_rate = (
            matched_data["isFraud"].mean() * 100 if total_count else 0.0
        )
        results.append({
            "candidate": rule_name,
            "total_transactions": total_count,
            "total_fraud_rate_pct": round(total_fraud_rate, 2),
            "already_covered": int((condition & baseline_covered).sum()),
            "additional_transactions": additional_count,
            "additional_fraud_transactions": additional_fraud_count,
            "additional_fraud_rate_pct": round(additional_fraud_rate, 2),
        })

    result_df = pd.DataFrame(results).sort_values(
        by=["additional_fraud_rate_pct", "additional_transactions"],
        ascending=[False, False],
    )
    print("\nCandidate incremental coverage (each compared independently):")
    print(result_df.to_string(index=False))
    return result_df

# ==========================================================
# MAIN
# ==========================================================

def main():

    # ======================================================
    # 1. Get preprocessed data
    # ======================================================

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = preprocess_data()


    # ======================================================
    # 2. Load final XGBoost configuration
    # ======================================================

    with open(
            CONFIG_PATH,
            "r"
    ) as file:

        config = json.load(file)


    selected_features = config[
        "selected_features"
    ]


    # ======================================================
    # 3. Keep full features for rule analysis
    #    and selected features for XGBoost
    # ======================================================
    X_train_full = X_train.copy()
    X_validation_full = X_validation.copy()

    X_train = X_train[
        selected_features
    ]

    X_validation = X_validation[
        selected_features
    ]
    print("\n========== DEVICE VALUE CHECK ==========")

    print("\nMost common id_30 values:")
    print(X_validation["id_30"].value_counts(dropna=False).head(30))

    print("\nValues containing 'Android':")
    print(
        X_validation.loc[
            X_validation["id_30"].astype("string")
            .str.contains("Android", case=False, na=False),
            "id_30"
        ].value_counts()
    )

    print("\nValues containing 'iOS':")
    print(
        X_validation.loc[
            X_validation["id_30"].astype("string")
            .str.contains("iOS", case=False, na=False),
            "id_30"
        ].value_counts()
    )
    android_mask = (
        X_validation["id_30"].astype("string")
        .str.contains("Android", case=False, na=False)
    )

    ios_mask = (
        X_validation["id_30"].astype("string")
        .str.contains(r"\biOS\b", case=False, na=False)
    )

    print("Android matches:", int(android_mask.sum()))
    print("iOS matches:", int(ios_mask.sum()))



# ======================================================
    # 4. Load locked XGBoost model
    # ======================================================

    model = XGBoostModel(

        max_depth=10,

        learning_rate=0.3,

        min_child_weight=1,

        n_estimators=4000,

        early_stopping_rounds=100,

        random_state=42
    )


    model.load(
        MODEL_PATH
    )


    # ======================================================
    # 5. TRAINING ANALYSIS
    # ======================================================

    print(
        "\n\n"
        "##################################################\n"
        "#         RULE CANDIDATE ANALYSIS (VALIDATION)    #\n"
        "##################################################"
    )

    # analyze_rules_against_model(
    #     X_train,
    #     y_train,
    #     model,
    #     "TRAINING"
    # )

    # analyze_email_domains(
    #     X_train,
    #     y_train,
    #     model,
    #     "TRAINING"
    # )

    # analyze_card4(
    #     X_train,
    #     y_train,
    #     model,
    #     "TRAINING"
    # )
    #
    # analyze_card6(
    #     X_train,
    #     y_train,
    #     model,
    #     "TRAINING"
    # )
    # analyze_addr1(
    #     X_train,
    #     y_train,
    #     model,
    #     "TRAINING"
    # )

    # analyze_dist1(X_train,y_train, model, "TRAINING")
    # analyze_dist2(X_train, y_train, model, "TRAINING")
    #analyze_id31(X_train, y_train, model, "TRAINING")
    # analyze_id31_grouped(X_train, y_train, model, "TRAINING")
    # analyze_id30(X_train, y_train, model, "TRAINING")
    # analyze_id30_grouped(X_train, y_train, model, "TRAINING")
    # analyze_device_info(X_train, y_train, model, "TRAINING")
    analyze_incremental_coverage(
        X_validation_full,
        y_validation,
        "VALIDATION"
    )


    # ======================================================
    # 6. VALIDATION ANALYSIS
    # ======================================================

    print(
        "\n\n"
        "##################################################\n"
        "#              VALIDATION ANALYSIS               #\n"
        "##################################################"
    )
    analyze_candidate_overlap(X_validation, y_validation, "VALIDATION")
    # analyze_rules_against_model(
    #     X_validation,
    #     y_validation,
    #     model,
    #     "VALIDATION"
    # )

    # analyze_email_domains(
    #     X_validation,
    #     y_validation,
    #     model,
    #     "VALIDATION"
    # )
    #
    # analyze_card4(
    #     X_validation,
    #     y_validation,
    #     model,
    #     "VALIDATION"
    # )
    #
    # analyze_card6(
    #     X_validation,
    #     y_validation,
    #     model,
    #     "VALIDATION"
    # )
    # analyze_addr1(
    #     X_validation,
    #     y_validation,
    #     model,
    #     "VALIDATION"
    # )
    # analyze_dist1(X_validation,y_validation, model, "VALIDATION")
    # analyze_dist2(X_validation, y_validation, model, "VALIDATION")
    # analyze_id31(X_validation ,y_validation,model, "VALIDATION")
    # analyze_id31_grouped(X_validation, y_validation, model, "VALIDATION")
    # analyze_id30(X_validation, y_validation, model, "VALIDATION")
    # analyze_id30_grouped(X_validation, y_validation, model, "VALIDATION")
# analyze_device_info(X_validation, y_validation, model, "VALIDATION")

def analyze_candidate_overlap(X, y, dataset_name="VALIDATION"):
    """
    Analyze overlap among candidate rules after excluding transactions
    already covered by the current three-rule baseline.
    """

    # Existing rule baseline
    product_c_missing_address = (
            X["ProductCD"].eq("C") & X["addr1"].isna()
    )

    product_h_amount = (
            X["ProductCD"].eq("H")
            & X["TransactionAmt"].gt(200)
            & X["TransactionAmt"].le(500)
    )

    outlook_email = (
            X["P_emaildomain"].eq("outlook.com")
            | X["R_emaildomain"].eq("outlook.com")
    )

    existing_rules = (
            product_c_missing_address
            | product_h_amount
            | outlook_email
    )

    # Candidate rule masks
    candidates = {
        "R_EMAIL_GMAIL": X["R_emaildomain"].eq("gmail.com"),
        "ANDROID_OS": (
            X["id_30"].astype("string")
            .str.contains("Android", case=False, na=False)
        ),
        "IOS": (
            X["id_30"].astype("string")
            .str.contains(r"\biOS\b", case=False, na=False)
        ),
    }

    # Exclude transactions already covered by existing rules
    new_candidates = {
        name: mask & ~existing_rules
        for name, mask in candidates.items()
    }

    # Compare candidates with one another
    print(f"\n========== CANDIDATE OVERLAP - {dataset_name} ==========")

    for name, mask in new_candidates.items():
        other_candidates = [
            other_mask
            for other_name, other_mask in new_candidates.items()
            if other_name != name
        ]

        overlaps = mask & pd.concat(other_candidates, axis=1).any(axis=1)

        print(
            f"{name}: "
            f"{int(mask.sum())} new transactions; "
            f"{int(overlaps.sum())} also covered by another candidate"
        )

    # Combined coverage of all three candidates
    combined_candidates = pd.concat(
        list(new_candidates.values()), axis=1
    ).any(axis=1)

    y_series = pd.Series(y, index=X.index).reset_index(drop=True)
    combined_candidates = combined_candidates.reset_index(drop=True)

    new_fraud = int(y_series[combined_candidates].sum())
    new_transactions = int(combined_candidates.sum())

    print("\nCombined incremental coverage:")
    print(f"New transactions: {new_transactions}")
    print(f"Fraud transactions: {new_fraud}")

    if new_transactions:
        print(f"Fraud rate: {new_fraud / new_transactions * 100:.2f}%")


if __name__ == "__main__":
    main()
