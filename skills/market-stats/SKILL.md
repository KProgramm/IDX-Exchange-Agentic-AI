---
name: market-stats
description: Answer real estate market questions for a California city or ZIP from sold-home data: median and average sale price, price per square foot, days on market, list-to-close ratio, monthly price trend, and active-vs-sold inventory. Use whenever the user asks about prices, trends, "is now a good time to buy", how fast homes sell, or how competitive a market is.
---

# Market stats

Run the CLI with the city as one quoted argument. This uses no Gemini calls:

```bash
~/Desktop/developer/idx/IDX-Exchange-Agentic-AI/.venv/bin/python ~/Desktop/developer/idx/IDX-Exchange-Agentic-AI/market/cli.py "San Diego"
```

It prints three blocks: summary (sold count, average and median price, price per sqft, days on market, list-to-close %), a monthly trend (sales, average price, price per sqft, days on market, month-over-month and year-over-year %), and inventory (active listings, sales pace, months of supply).

Options:

- `--only summary|trend|inventory`: print just one block.
- `--zip 91107`: narrow the city to one 5-digit ZIP.
- `--type Condominium`: property subtype. Default is `SingleFamilyResidence`. Never mix subtypes; if the user doesn't say, use the default and tell them which type the numbers cover.
- `--months N` (summary and inventory window, default 12) and `--trend-months N` (default 24).
- `--json`: raw numbers instead of text.

How to answer:

- Report only what is printed. Don't invent figures.
- The sold data currently covers about 6 months, so say "over the last N months of data" using the printed month count, not "12-month". Year-over-year is not available until there are 13+ months; the output says so.
- The first and last months of the trend are partial, so don't read much into their sale counts.
- "Is now a good time to buy": give the facts (price direction, days on market, list-to-close %, months of supply) and let the user decide. Under about 100% list-to-close and rising months of supply favor buyers; the reverse favors sellers. Don't give a guarantee or financial advice.
- If the output says no sales were found, say so and suggest checking the city spelling or trying another property type. If the city has no data for the default type, retry with `--type Condominium` before giving up.
- If the city name contains double quotes, remove them before passing it in.
