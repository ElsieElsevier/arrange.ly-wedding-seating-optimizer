# Arrangely

A wedding seating planner. Describe your guests in plain language, and Arrangely seats similar people together using **semantic text embeddings** and **spectral clustering**.

Everything runs on your computer. Guest descriptions are never sent anywhere.

## How It Works

1. **Understand each guest.** A local language model ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) turns each description into a vector, so that "pediatric nurse" and "orthopedic surgeon" land close together even though they share no words.
2. **Keep couples together.** Partners are treated as a single unit.
3. **Find the groups.** Spectral clustering builds a similarity graph of the guests and splits it into one group per table.
4. **Fit the tables.** Groups are adjusted to fit the seats, even out table sizes, and honor "keep apart" requests.

## Getting Started

You need Python 3.12 or newer. From the project's root directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

On macOS or Linux, activate with `source venv/bin/activate` instead.

Then open <http://localhost:5000>. Select **Load example** to try it with sample guests.

The first seating plan needs internet access and takes longer, because it downloads the language model (about 90 MB). After that, it works offline.

## Writing Good Descriptions

Describe how you know the guest, along with their age, interests, and personality.

Specific details matter. With short descriptions, wording like "Bride's" can outweigh what actually makes guests alike, so an 84-year-old grandmother could land with the college friends. Always review the plan before using it.

Use the **Keep apart from** box for conflicts, not the description. Embeddings can't reliably read negation: "seat me with Dana" and "do not seat me near Dana" look very similar to the model.

## Limitations

- Seating plans aren't saved on the server. The page keeps your guest list as a draft in your browser.
- A keep-apart request that can't be met, such as when there's only one table, is reported on the results page.
- Guests are grouped by similarity, so people with different backgrounds won't be mixed on purpose.
- Tables are kept fairly even, so a very large circle may be split, and a very small one may be seated with others.

## API

`POST /api/seating`

```json
{
  "seats_per_table": 8,
  "num_tables": 3,
  "guests": [
    {
      "name": "Maya Chen",
      "description": "Bride's college roommate, a lawyer who loves karaoke.",
      "partner": "Daniel Chen",
      "avoid": ["Tom Baker"]
    }
  ]
}
```

`num_tables`, `partner`, and `avoid` are optional. If `num_tables` is left out, Arrangely uses the fewest tables that fit everyone. The response lists each table's guests, plus a summary of the plan's quality.

## Running the Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest
python -m pytest -m model
```

The second test command uses the real language model.

## Server Settings

By default, the server only accepts connections from your own computer, with debug mode off. To change this, set the `HOST`, `PORT`, and `FLASK_DEBUG=1` environment variables.

Debug mode lets anyone who can reach the server run code on your computer, so never combine it with `HOST=0.0.0.0` or a tunnel such as ngrok.
