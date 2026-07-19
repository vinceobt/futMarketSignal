# FUTNext price API (reference)

Reverse-engineered from the **FC26 Enhancer** browser extension (the `futsolver`
repo, `js/index.js` bundle) and **verified live on 2026-07-08**. This documents an
undocumented third-party endpoint we can use as a price source for the collector.

> **Status:** candidate price source, not yet wired in. See "Integration notes".

---

## TL;DR

- **Current player prices** are served from an **open, unauthenticated** endpoint:
  `GET https://enhancer-api.futnext.com/players/v2/prices`
- No account, no API key, no token required. The extension's axios interceptor
  only attaches a `Bearer` token *if one exists* (`t && (...)`) — the price
  endpoint accepts requests without it.
- IDs are EA **`definitionId`**s — the **numeric part of a fut.gg card URL**
  (`26-184788461` → `184788461`), which we already store in the watchlist.
- Cleaner than the current fut.gg headless-browser scraping: plain JSON GET,
  batched up to 50 ids, no Chromium / no Turnstile.

---

## Endpoints

Base host: `https://enhancer-api.futnext.com` (the extension's `ENHANCER_API`).
All are `GET`. Only the first is usable without auth.

| Endpoint | Auth | Params | Notes |
|---|---|---|---|
| `/players/v2/prices` | **none** ✅ | `ids`, `platform` | Current lowest-BIN prices. **This is the one we use.** |
| `/players/price-trend` | required ❌ | `ids`, `platform`, `range` | Price history. Returns `403 Forbidden` without the extension's anonymous Supabase token. |
| `/players/extended-price-trend` | required ❌ | `ids`, `platform`, `range` | Longer trend series. Also `403` without a token. |
| `/players/fodder-prices` | ? | `platform` | Not needed here. |
| `/club-items/prices`, `/consumables/prices`, `/managers/prices`, `/play-styles/prices` | ? | — | Non-player items; out of scope. |

We do **not** need the trend endpoints: fc-market-analytics is append-only and
builds its own history from spot-price snapshots, so the open `/players/v2/prices`
endpoint is a complete fit.

---

## `/players/v2/prices` contract

### Request

```
GET https://enhancer-api.futnext.com/players/v2/prices?ids=<ids>&platform=<platform>
```

| Param | Required | Format | Rules |
|---|---|---|---|
| `ids` | yes | `definitionId`(s) joined by **underscore** `_`, e.g. `231747_239085` | Batch up to **50** ids per call (`maxBatchSize: 50` in the extension). |
| `platform` | yes | `pc` **or** `ps` **only** | Any other value (`console`, `xbox`, empty, `web`) → `400 {"errors":["Invalid input"]}`. There is no combined "console" price — `ps` is the PlayStation/console market. |

No auth headers needed. Sending `Origin: https://www.ea.com` is harmless but not
required.

### Response

`200` with a JSON array, one object per **known** id:

```json
[
  {
    "definitionId": 184788461,
    "avg": 2700000,
    "price": 2700000,
    "top5Cheapest": [2700000],
    "updatedAt": 1783497942658
  }
]
```

| Field | Meaning |
|---|---|
| `definitionId` | EA definition id echoed back (join key). |
| `price` | Current **lowest BIN** (what we store as the quote). |
| `avg` | Average / reference price. |
| `top5Cheapest` | Up to 5 cheapest listings. |
| `updatedAt` | Last update, **ms epoch**. |

**Edge cases**
- Unknown / invalid id → omitted from the array (an all-unknown batch returns
  `[]` with `200`, *not* an error). Map responses back by `definitionId`; treat a
  missing id as "no price".
- Bad `platform` or malformed `ids` → `400 {"errors":["Invalid input"]}`.

### Verified examples (2026-07-08)

```bash
# single, PlayStation/console market
curl "https://enhancer-api.futnext.com/players/v2/prices?ids=184788461&platform=ps"
# → [{"definitionId":184788461,"avg":2700000,"price":2700000,"top5Cheapest":[2700000],"updatedAt":1783497942658}]

# same card, PC market (different price)
curl "https://enhancer-api.futnext.com/players/v2/prices?ids=231747&platform=pc"
# → [{"definitionId":231747,"avg":238000,"price":238000,"top5Cheapest":[238000],"updatedAt":1783497941137}]

# batch of two, underscore-joined
curl "https://enhancer-api.futnext.com/players/v2/prices?ids=231747_239085&platform=ps"
# → [{"definitionId":231747,...},{"definitionId":239085,...}]

# rejected platform
curl "https://enhancer-api.futnext.com/players/v2/prices?ids=231747&platform=console"
# → 400 {"errors":["Invalid input"]}
```

---

## ID mapping (fut.gg → definitionId)

The watchlist stores full fut.gg URLs. The **numeric part of the card segment is
the `definitionId`** — no lookup needed:

```
https://www.fut.gg/players/erling-haaland/26-184788461/
                                            ^^^^^^^^^   ← definitionId = 184788461
```

Verified: `184788461` returned a valid price above. Extraction:

```python
import re
DEF_ID = re.compile(r"-(\d+)/?$")          # last "-<digits>" in the URL
def_id = DEF_ID.search(player.url).group(1)  # "184788461"
```

(`config.parse_player_url` already isolates the `card_seg` `26-184788461`; take the
part after the last `-`.)

---

## Integration notes (for a future `futnext_source.py`)

Follows the existing `PriceSource` contract in
`src/futmarket/collectors/base.py`.

- **Platform map:** our config uses `{"console", "pc"}`; FUTNext wants
  `{"ps", "pc"}`. Map `console → ps`, `pc → pc`.
- **Batching:** unlike the per-player scrapers, this endpoint takes up to 50 ids
  at once. Ideally batch the whole watchlist into `_`-joined chunks of 50 rather
  than one HTTP call per player.
- **History:** leave `PriceQuote.history` empty; the pipeline builds history from
  snapshots. (Trend endpoints are gated — see above.)
- **Registration:** add `FutNextSource` to `collectors/__init__.py::get_source`
  and add `"futnext"` to `config.VALID_SOURCES`.
- **Dependency:** a plain `httpx`/`requests` GET — no `patchright`/browser needed.

Sketch:

```python
r = httpx.get(
    "https://enhancer-api.futnext.com/players/v2/prices",
    params={"ids": "_".join(def_ids), "platform": "pc" if platform == "pc" else "ps"},
    timeout=15,
).json()
by_id = {row["definitionId"]: row for row in r}
```

---

## Caveats

- **Undocumented private backend**, not a licensed/public feed. It can change or
  disappear without notice; validate responses defensively.
- **No stated rate limits.** Be polite — batch, and keep the existing
  inter-poll delays. Excessive traffic could get the endpoint locked down.
- **ToS:** this is FUTNext's own backend. Programmatic use may run against their
  terms. Keep it opt-in; keep the fut.gg source as the default.
- Only `pc` and `ps` markets exist; there is no Xbox-specific or unified price.
