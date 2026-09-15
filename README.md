# Buy or Wait? — AI Financial Agent

My solution for the **HackerRank Orchestrate** hackathon (September 2026 edition) — challenge: *"Buy or Wait?"*

## The problem

Given a user's account balances, spending history, and a requested purchase, decide whether they can safely afford it right now, later, or with a payment plan — accounting for recurring expenses, pending payments, essential spending, confirmed income, and available payment options.

## My approach

A 5-stage deterministic pipeline rather than handing the whole decision to an LLM:

```
code/
├── main.py             # Entry point
├── data_loader.py       # Loads and joins the account + transaction data
├── currency.py           # Dated exchange-rate conversion
├── forecast.py           # 90-day balance simulation
├── decision_engine.py     # Affordability rules, payment eligibility, tie-breaks
├── formatter.py           # Writes the final output.csv in the required schema
└── resolve_images.py     # OCR for the few requests with amounts embedded in images
```

I kept the core financial logic deterministic and rule-based — an LLM guessing at someone's balance is a bug, not a feature. Vision/OCR was used only where a request's amount genuinely lived inside an image.

## Evaluating my own work

`code/evaluation/evaluate.py` scores my predictions against the labelled sample set, field by field. That surfaced a known limitation: recurring-expense detection currently relies on exact description matching, so it misses variable/category-level spend (groceries, transport). That's the next thing to improve.

## Result

Finished **#1357 of 3,062** (top 45%) on my first Orchestrate.

## Data

The `dataset/` folder here is HackerRank's provided sample data for the challenge. Full task spec is in `problem_statement.md`.