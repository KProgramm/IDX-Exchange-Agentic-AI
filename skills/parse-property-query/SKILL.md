---
name: parse-property-query
description: Turn a free-text real estate search (like "3 bed condos in Irvine under $1.5M with a pool") into a structured JSON filter object for the rets_property listings table. Use whenever the user asks to find, search for, or filter homes or property listings.
---

# Parse property query

Run the parser with the user's search text as one quoted argument:

```bash
~/Desktop/developer/idx/IDX-Exchange-Agentic-AI/.venv/bin/python ~/Desktop/developer/idx/IDX-Exchange-Agentic-AI/parser/parse_query.py "<user's search text>"
```

It prints a JSON object with these keys. A null value means the user didn't ask for that filter.

- `city`: maps to `L_City`
- `max_price`: maps to `L_SystemPrice`
- `min_beds`: maps to `L_Keyword2`
- `min_baths`: maps to `LM_Dec_3`
- `min_sqft`: maps to `LM_Int2_3`
- `property_type`: maps to `L_Type_`
- `pool`: maps to `PoolPrivateYN` (true = wants a pool, false = no pool)
- `view`: maps to `ViewYN`
- `max_hoa`: maps to `AssociationFee`

Show the user the JSON exactly as printed. Don't add filters the parser didn't return. If the search text contains double quotes, remove them before passing it in.

Database search is not wired up yet (that's the next skill), so stop after returning the filters.
