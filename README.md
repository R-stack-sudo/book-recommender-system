# Leafwise book recommender

A Flask web app based book recommender project. Pick a book from the catalogue and the app recommends titles that were liked by readers with similar preferences.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
flask --app app run --debug
```

Open `http://127.0.0.1:5000` in a browser.

## What is included

- A popular-books landing page, based on rating volume and average reader score.
- A case-insensitive title search over 706 collaborative-filtering model titles.
- Six nearest-neighbour recommendations and an API title-search endpoint at `/api/titles?q=...`.
- A compatibility loader for the supplied legacy pandas model artifacts, so the project works with current pandas versions.

The `*.pkl` files are the trained tutorial artifacts. They are required to run the application.
