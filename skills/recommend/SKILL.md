---
name: recommend
description: Given one active listing the user likes (by listing id), recommend the top 5 similar active listings and say whether each is priced in line with recent sold comps. Use for "find homes like this one", "similar to listing 368235", "is this a fair price compared to recent sales". Do NOT use to search from scratch by city, price, beds or description; those go to search-listings or semantic-search.
---

# Recommend similar listings

Run the CLI with the listing id. It uses no Gemini or other paid API calls; the embedding index is already on disk:

```bash
~/Desktop/developer/idx/IDX-Exchange-Agentic-AI/.venv/bin/python ~/Desktop/developer/idx/IDX-Exchange-Agentic-AI/recommend/recommend.py 368235 --json
```

Options:

- `--json`: raw results. Always use this when answering for a user.
- `--top-k N`: number of recommendations (default 5).
- `--mode semantic`: rank by remark similarity alone, with no type filter. Only for showing what the hybrid score adds. `--mode both` prints the two.

The listing id is the `id` field from search-listings or semantic-search results. If the user only describes a listing, find it with search-listings first.

## What the score means

Each recommendation has a score out of 100 with its parts: price (20), beds (15), city (15), sqft (10) and remark similarity (40). Price and sqft are compared as percentages of the liked listing's, and beds are graded (1 apart is better than 3 apart). Candidates are limited to the same property type as the liked listing. The score is a ranking, not a percent match: don't call a listing "92% similar".

## Comp validation

Each listing (the liked one and every recommendation) is checked against sold homes in the same city and property type, with living area within 20%, over the last 6 months of the sold data. The result gives the comp-implied price, the percent difference, a verdict (in line / above / below / well above / well below) and the comp count. The `comp_window` field gives the real date range; quote it, because the sold data does not run up to today.

- Status `insufficient` means fewer than 5 comps, or no square footage. Say "not enough comparable sales", never a price opinion or a delta. Do not widen the search yourself.
- Status `no_price` means the list price looks like a placeholder (under $10,000).
- A verdict is a comparison with past sales, not an appraisal. Condition, lot, view and updates are not in the comps.

## How to answer

- Report the `flags` field. Notable ones: "type unknown: not filtered by type" (the liked listing has no property type, so recommendations may mix types), "no remarks vector" (similarity of remarks was not used, so ranking is by price, beds, city and sqft only) and placeholder price.
- Listings with the same remarks and a different unit number in the same building are kept on purpose. Relists of the same unit are removed.
- Show the address, price, beds/baths/sqft, score and the comp line for each. Don't invent anything else.
- If fewer than 5 recommendations come back (rare property types), say so rather than loosening the filter.
