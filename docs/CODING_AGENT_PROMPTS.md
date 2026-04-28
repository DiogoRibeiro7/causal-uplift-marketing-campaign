# Coding Agent Prompts

## Prompt 1 — Strengthen data adapters

Improve the data-loading layer so the repository can read both the synthetic dataset and Criteo-style uplift datasets. Add schema validation, column mapping, and helpful errors when treatment or outcome columns are missing.

## Prompt 2 — Add robust uplift metrics

Implement AUUC, normalized Qini, uplift-at-k, and bootstrap confidence intervals. Add unit tests using small synthetic fixtures.

## Prompt 3 — Add cross-fitting

Implement cross-fitted S-learner and T-learner estimates to reduce overfitting in uplift scores. Keep the API simple and documented.

## Prompt 4 — Add budget simulation

Build a module that simulates campaign contact budgets, contact costs, conversion value, expected net value, and stockholder-ready output tables.

## Prompt 5 — Add reporting

Create a script that generates `reports/campaign_policy_report.md` from pipeline outputs. Include treatment summary, Qini metrics, top segment actions, assumptions, and limitations.
