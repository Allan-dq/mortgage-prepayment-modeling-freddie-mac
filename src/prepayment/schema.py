"""Freddie Mac Release 47 (July 2026) pipe-delimited file schemas."""

ORIGINATION_COLUMNS = [
    "fico", "first_payment_date", "first_time_homebuyer", "maturity_date",
    "msa", "mi_pct", "units", "occupancy", "original_cltv", "original_dti",
    "original_upb", "original_ltv", "original_rate", "channel",
    "prepay_penalty", "amortization_type", "state", "property_type",
    "postal_code", "loan_id", "loan_purpose", "original_term", "borrowers",
    "seller", "super_conforming", "pre_harp_loan_id", "special_program",
    "harp_indicator", "valuation_method", "interest_only", "vantage_score",
]
PERFORMANCE_COLUMNS = [
    "loan_id", "period", "current_upb", "delinquency_status", "loan_age",
    "remaining_term", "defect_settlement_date", "modification_flag",
    "zero_balance_code", "zero_balance_date", "current_rate",
    "non_interest_bearing_upb", "ddlpi", "mi_recoveries", "net_sales_proceeds",
    "non_mi_recoveries", "total_expenses", "legal_costs", "maintenance_costs",
    "taxes_insurance", "misc_expenses", "actual_loss", "cumulative_mod_cost",
    "rate_step_indicator", "payment_deferral_flag", "estimated_ltv",
    "zero_balance_removal_upb", "delinquent_accrued_interest", "disaster_flag",
    "borrower_assistance_plan", "current_period_mod_cost",
    "interest_bearing_upb", "mi_cancellation_indicator", "servicer",
    "bankruptcy_cramdown_costs",
]

NUMERIC_ORIGINATION = [
    "fico", "mi_pct", "units", "original_cltv", "original_dti", "original_upb",
    "original_ltv", "original_rate", "original_term", "borrowers", "vantage_score",
]

NUMERIC_PERFORMANCE = [
    "current_upb", "loan_age", "remaining_term", "current_rate",
    "non_interest_bearing_upb", "estimated_ltv", "interest_bearing_upb",
]
