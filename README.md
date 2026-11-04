# Arrangely

Originally built at e{hack}2024 (2nd place); rebuilt in 2026.

A wedding seating planner. Describe your guests in plain language, and Arrangely seats similar people together using **semantic text embeddings** and **spectral clustering**.

Guest descriptions are processed on your computer and are never sent anywhere. Arrangely is a single Flask app with its own web page. It has no database and no user accounts.

## How It Works

1. **Read the descriptions.** A language model running on your computer ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) turns each description into a vector that represents its meaning. This lets guests with similar backgrounds land close together even when they're described in different words.
2. **Link couples.** Guests connected by the partner field become a single unit, so they can't be separated. Names, on their own, have no effect on where a guest sits.
3. **Find the groups.** Spectral clustering treats the units as a graph, where stronger connections mean more similar descriptions, and splits it into one group per table.
4. **Fit the groups to the tables.** Spectral clustering knows nothing about seat limits. So Arrangely moves guests out of overfull groups, then keeps moving and swapping units for as long as that improves the plan. This step enforces seat limits, handles keep-apart requests, and evens out table sizes.

The same input always produces the same plan.

## Getting Started

You need Python 3.12 or newer. It has been tested on 3.12 and 3.14. From the project's root directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

On macOS or Linux, activate with `source venv/bin/activate` instead.

Then open <http://localhost:5000>. Select **Load example** to try it with sample guests.

The first seating plan needs internet access and takes longer, because it downloads the language model (about 90 MB) into `.model_cache`. After that, it works offline.

## Using the Page

1. Set **Seats per table**. **Number of tables** is optional. If you leave it blank, Arrangely uses the fewest tables that fit everyone.
2. Enter each guest's name and description.
3. Optionally, fill in **Sits with (partner)** and **Keep apart from**. Names must match the guest list exactly. You only need to enter a link on one of the two guests.
4. Select **Create seating plan**.

The page saves your guest list as a draft in your browser, so it's still there after a reload.

### Reading the Results

- **Average similarity between tablemates** should be higher than **expected from random seating**. The larger the gap, the more alike the tablemates are compared to chance.
- **Keep-apart requests met** shows how many requests the plan honors.
- **Most like** names the tablemate whose description is closest to that guest's.

### Writing Good Descriptions

Describe how you know the guest, along with their age, interests, and personality.

Specific details matter. With short descriptions, wording like "Bride's" can outweigh what actually makes guests alike, so an 84-year-old grandmother could land with the college friends.

Use the **Keep apart from** box for conflicts, not the description. Embeddings can't reliably read negation: "seat me with Dana" and "do not seat me near Dana" look very similar to the model.

## Limitations

- **Review every plan.** Similar descriptions don't guarantee that guests will get along.
- **Partners are always kept together, but keep-apart requests can fail.** If there's no way to meet a request, such as when there's only one table, the plan still returns and reports the conflict.
- **Tables are kept fairly even.** A very large circle may be split, and a very small one may be seated with others.
- **Guests are grouped by similarity.** People with different backgrounds won't be mixed on purpose.
- **Plans aren't saved.** Creating a new plan replaces the one on the page.
- **Seats per table is the same for every table.**
- **It's built to run locally.** There's no login, so don't put it on a public address as it is.

## API

The page uses a single endpoint, which you can also call directly.

`POST /api/seating`

```json
{
  "seats_per_table": 3,
  "num_tables": 2,
  "guests": [
    {
      "name": "Maya Chen",
      "description": "Bride's college roommate, a lawyer who loves karaoke.",
      "partner": "Daniel Chen"
    },
    {
      "name": "Daniel Chen",
      "description": "Maya's husband, an outgoing startup founder who loves to dance."
    },
    {
      "name": "Priya Nair",
      "description": "Was in the bride's sorority and is always first on the dance floor.",
      "avoid": ["Tom Baker"]
    },
    {
      "name": "Tom Baker",
      "description": "Groom's manager at the accounting firm, a big golf fan."
    },
    {
      "name": "Aisha Khan",
      "description": "Colleague of the groom in the finance department."
    },
    {
      "name": "Marcus Hill",
      "description": "Sits next to the groom at the office and talks about spreadsheets."
    }
  ]
}
```

| Field | Required | Rules |
|---|---|---|
| `seats_per_table` | Yes | Whole number from 1 to 50 |
| `num_tables` | No | Whole number from 1 to the number of guests |
| `guests` | Yes | 2 to 500 guests |
| `name` | Yes | Up to 100 characters, and different from every other guest's name, ignoring capitalization |
| `description` | Yes | Up to 2,000 characters |
| `partner` | No | Name of another guest |
| `avoid` | No | List of other guests' names |

A successful response looks like this. The second table is left out here to save space.

```json
{
  "tables": [
    {
      "table": 1,
      "average_similarity": 0.545,
      "guests": [
        {
          "name": "Tom Baker",
          "closest_tablemate": { "name": "Aisha Khan", "similarity": 0.631 }
        },
        {
          "name": "Aisha Khan",
          "closest_tablemate": { "name": "Tom Baker", "similarity": 0.631 }
        },
        {
          "name": "Marcus Hill",
          "closest_tablemate": { "name": "Aisha Khan", "similarity": 0.559 }
        }
      ]
    }
  ],
  "summary": {
    "guest_count": 6,
    "tables_used": 2,
    "seats_per_table": 3,
    "average_tablemate_similarity": 0.478,
    "random_seating_similarity": 0.344,
    "silhouette_score": 0.297,
    "avoid_requests": 1,
    "avoid_conflicts": 0
  }
}
```

`silhouette_score` runs from -1 to 1, and higher means better-separated tables. It's `null` when it can't be calculated, such as when there's only one table.

Errors return JSON with a `message` field:

| Status | Meaning |
|---|---|
| `400` | The request is invalid or can't be seated, such as when there are too few seats |
| `503` | The language model couldn't be loaded, usually because the first run has no internet access |

## Project Structure

```text
├── app.py              # Web server, routes, and request validation
├── embeddings.py       # Loads and runs the language model
├── seating.py          # Spectral clustering and table fitting
├── static/index.html   # The web page
├── tests/test_api.py   # Automated tests
├── requirements.txt    # Packages needed to run the app
└── requirements-dev.txt  # Those packages plus the test tools
```

## Running the Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest
python -m pytest -m model
```

The first test command uses a small fake model, so it runs in seconds. The second uses the real language model, which is downloaded if you don't already have it.

## Server Settings

By default, the server runs on port 5000 with debug mode off, and only accepts connections from your own computer. You can change this with environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | The port to listen on |
| `HOST` | `127.0.0.1` | The network address to listen on |
| `FLASK_DEBUG` | Off | Set to `1` to turn on debug mode |

Debug mode lets anyone who can reach the server run code on your computer, so never combine it with `HOST=0.0.0.0` or a tunnel such as ngrok.
