# Universal Bank - Credit Risk Policy (2026)

## 1. Loan Amount and Exposure Rules
- **Rule CP101 (High Value Loans):** Any loan application exceeding **DM 10,000** must be subjected to a senior auditor review, even if the model predicts 'Good'. The maximum automated approval limit is **DM 10,000**.
- **Rule CP102 (Small Loans):** Loans under **DM 1,000** are generally considered low exposure unless the applicant has 'unknown' checking account status.

## 2. Demographic and Employment Constraints
- **Rule CP201 (Young Applicants):** Applicants aged **25 or younger** must have at least 'moderate' savings or checking account status to be considered for automated approval. If savings are 'little' or 'unknown', a 'Bad' risk prediction is reinforced.
- **Rule CP202 (High Skilled Labor):** Applicants with a Job Skill Level of **3 (Highly Skilled)** are given preferential consideration. Even with moderate risk scores, they may be approved if the loan duration is under **12 months**.
- **Rule CP203 (Unskilled Labor):** Applicants with Job Level **0 (Unskilled)** should be primarily evaluated on their existing savings rather than future income potential.

## 3. Financial Wellness Indicators
- **Rule CP301 (Savings Buffer):** A 'rich' or 'quite rich' savings account status is a strong indicator of repayment capacity. This can override 'rent' housing status or 'business' loan purpose volatility.
- **Rule CP302 (Liquidity Check):** An 'unknown' checking account status combined with 'rent' housing status is considered a high-risk combination, regardless of age.

## 4. Loan Purpose and Duration
- **Rule CP401 (Education and Business):** Loans for 'education' or 'business' purposes are considered investment-class. These allow for longer durations (up to **48 months**) provided the applicant is at least 'Skilled' (Level 2).
- **Rule CP402 (Luxury and Others):** Loans for 'vacation' or 'others' are non-essential. These are strictly limited to **24 months** and require an 'own' housing status for approval.
- **Rule CP403 (Appliances and Repairs):** These are typically short-term needs. Durations exceeding **18 months** for these categories trigger a warning on 'Bad' risk probability.

## 5. Risk Assessment Summary Guidelines
- Decisions must be explained based on a combination of the statistical model (probabilistic) and these manual policy override rules (deterministic).
- The "GenAI Auditor" must highlight which policy rule triggered an override or a reinforced decision.
