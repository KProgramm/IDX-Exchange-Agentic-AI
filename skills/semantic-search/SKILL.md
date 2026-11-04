---
name: semantic-search
description: Find active listings by description or "vibe" using semantic search over listing remarks: style, character, feel, setting, features described in words (e.g. "charming craftsman with mountain views", "modern open concept with a big backyard for entertaining"). Use for descriptive free-text property queries. Do NOT use for structured filters like city, price, beds, baths, sqft or property type; those go to search-listings.
---

# Semantic search

Run the CLI with the description as one quoted argument. This uses no Gemini or other paid API calls; the model runs locally:

```bash
~/Desktop/developer/idx/IDX-Exchange-Agentic-AI/.venv/bin/python ~/Desktop/developer/idx/IDX-Exchange-Agentic-AI/embeddings/cli.py "charming craftsman with mountain views" --json
```

It returns the closest listings by meaning of their agent-written remarks, each with a similarity score (higher is closer), address, city, price, beds, baths, sqft, type and the first 150 characters of the remarks.

Options:

- `--json`: raw results as JSON. Always use this when answering for a user.
- `--top-k N`: number of results (default 5).

When to use it, and when not to:

- Use for vibe and description queries: architectural style, character, atmosphere, views, outdoor space, "good for entertaining", "quiet", "move-in ready".
- Do not use it to enforce structured filters. Semantic search does not guarantee a city, price cap, bed count or property type. If the user gives hard filters ("3 bed under $1.5M in Irvine"), use the search-listings skill (`search_active_listings`) instead.
- Mixed requests ("modern home in Irvine under $1.5M with a big backyard"): run the structured search for the hard filters, then run this skill for the description and show only listings that satisfy the filters. Say that the description part is a similarity ranking, not an exact match.

How to answer:

- Report only what is returned. Don't invent details beyond the fields and the 150-character snippet.
- Similarity is a relative ranking within one result set. Scores are not comparable across different queries, and there is no threshold that means "good match". Don't present a score as a percent match or call a result a strong match because of its number.
- Results can include homes that don't truly fit (the snippet shows only the start of the remarks). Say so if a result looks off rather than overselling it.
- Only the first ~1,000 characters of each listing's remarks were embedded, so details late in long remarks may not influence the ranking.
- The index covers active listings with remarks of 80+ characters. If the CLI says the index is missing, tell the user to run `embeddings/build_index.py`.
- If the description contains double quotes, remove them before passing it in.
