import pandas as pd


def rule_product_c_missing_address(transaction):
    product = transaction.get("ProductCD")
    address = transaction.get("addr1")

    if product == "C" and pd.isna(address):
        return {
            "rule": "PRODUCT_C_AND_MISSING_ADDRESS",
            "description": (
                "Product C transaction with a missing billing address"
            ),
            "evidence": {
                "ProductCD": product,
                "addr1": None
            }
        }

    return None


def rule_product_h_amount_200_500(transaction):
    product = transaction.get("ProductCD")
    amount = transaction.get("TransactionAmt")

    if (
            product == "H"
            and pd.notna(amount)
            and 200 < float(amount) <= 500
    ):
        return {
            "rule": "PRODUCT_H_200_500",
            "description": (
                "Product H transaction with amount between $200 and $500"
            ),
            "evidence": {
                "ProductCD": product,
                "TransactionAmt": amount
            }
        }

    return None


def rule_outlook_email(transaction):
    p_email = transaction.get("P_emaildomain")
    r_email = transaction.get("R_emaildomain")

    if p_email == "outlook.com" or r_email == "outlook.com":
        return {
            "rule": "OUTLOOK_EMAIL_DOMAIN",
            "description": (
                "Outlook.com appears as the purchaser or recipient "
                "email domain"
            ),
            "evidence": {
                "P_emaildomain": p_email,
                "R_emaildomain": r_email
            }
        }

    return None

def rule_recipient_gmail(transaction):
    recipient_email = transaction.get("R_emaildomain")
    if recipient_email == "gmail.com":
        return {
            "rule": "R_EMAIL_GMAIL",
            "description": "Recipient email domain is Gmail",
            "evidence": { "R_emaildomain": recipient_email }
        }
    return None

def rule_android_os(transaction):
    device_info = transaction.get("id_30")
    if pd.notna(device_info) and "android" in str(device_info).lower():
        return {
            "rule": "ANDROID_OS",
            "description": "Transaction is associated with an Android operating system",
            "evidence": { "id_30": device_info }
        }
    return None

def rule_ios(transaction):
    device_info = transaction.get("id_30")

    if (
            pd.notna(device_info)
            and pd.Series([device_info])
            .astype("string")
            .str.contains(r"\biOS\b", case=False, na=False)
            .iloc[0]
    ):
        return {
            "rule": "IOS",
            "description": "Transaction is associated with an iOS operating system",
            "evidence": {"id_30": device_info}
        }

    return None

class RuleEngine:

    def evaluate_rules(self, transaction):
        """
        Evaluate deterministic risk signals for one transaction.

        Rules provide investigation evidence; they do not independently
        determine whether a transaction is fraudulent.
        """

        matched_rules = []

        rules = [
            rule_product_c_missing_address,
            rule_product_h_amount_200_500,
            rule_outlook_email,
            rule_recipient_gmail,
            rule_android_os,
            rule_ios ]

        for rule in rules:
            result = rule(transaction)

            if result is not None:
                matched_rules.append(result)

        return matched_rules

