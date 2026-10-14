---
name: search-listings
description: Search active real estate listings in the MySQL database and show property cards, or show recent sold comps for a city. Use whenever the user asks to find, search for, browse, or compare homes, or asks what homes recently sold in a city.
---

# Search listings

Use the user's search text as one quoted argument. This parses it into filters (one Gemini call) and then queries the database:

```bash
~/Desktop/developer/idx/IDX-Exchange-Agentic-AI/.venv/bin/python ~/Desktop/developer/idx/IDX-Exchange-Agentic-AI/search/search_listings.py "<user's search text>"
```

If you already have filters (for example from the parse-property-query skill), pass them as JSON instead. This uses no Gemini call:

```bash
~/Desktop/developer/idx/IDX-Exchange-Agentic-AI/.venv/bin/python ~/Desktop/developer/idx/IDX-Exchange-Agentic-AI/search/search_listings.py --filters '{"city": "Irvine", "max_price": 1500000, "pool": false}'
```

Filter keys: `city`, `max_price`, `min_beds`, `min_baths`, `min_sqft`, `property_type`, `pool`, `view`, `max_hoa`. Use `null` for anything the user didn't ask for. `pool: false` means "no pool" and still filters.

Options (add after the search text or `--filters`):

- `--page N`: page number, starting at 1. Use it when the user asks for "more" results.
- `--limit N`: results per page (default 10, never more than 50).
- `--comps`: instead of active listings, show recent sold comps for the filter's city (needs a city).
- `--months N`: how many months of sold comps to show (default 6).

The output is ready-to-read property cards (address, price, beds/baths, sqft, type, HOA, days on market, pool, photo count) or comp cards (sale price, close date, sale-to-list ratio). Show them to the user as printed. Don't invent listings or details that aren't in the output. If there are no results, say so and suggest loosening a filter. If the search text contains double quotes, remove them before passing it in.
