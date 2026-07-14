# Modeling data dictionary

| Field | Role | Description |
|---|---|---|
| `loan_id` | key | Freddie Mac masked loan identifier; never published by this project |
| `period` | key / split | Monthly performance date (`YYYYMM` in raw files) |
| `response` | target | 1 when the next contiguous record has Zero Balance Code `01` |
| `current_upb` | exposure | Current unpaid principal balance at the decision month |
| `refi_incentive` | feature | Current note rate less contemporaneous market mortgage rate |
| `positive_incentive` | feature | Positive portion of refinance incentive |
| `loan_age` / `age_squared` | feature | Seasoning and nonlinear seasoning term |
| `balance_factor` | feature | Current UPB divided by original UPB |
| `months_in_money_prior` | feature | Prior months with incentive above 50 bp; shifted one month |
| `max_incentive_prior` | feature | Maximum prior refinance incentive; shifted one month |
| `prior_delinquent_months` | feature | Count of prior non-current months; shifted one month |
| `fico`, `original_dti`, `original_ltv` | features | Origination borrower/collateral characteristics |
| categorical fields | features | Occupancy, purpose, channel, property type, state, FTHB, modification, delinquency bucket |

See Freddie Mac's official File Layout for authoritative raw-field definitions.
