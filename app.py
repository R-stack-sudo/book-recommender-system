"""Flask application for the collaborative-filtering book recommender."""

from __future__ import annotations

import inspect
import pickle
import sys
import types
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILES = ("popular.pkl", "pt.pkl", "books.pkl", "similarity_scores.pkl")


def enable_legacy_pandas_pickle_support() -> None:
    """Load the tutorial's pandas-1.x artifacts with modern pandas."""
    import pandas as pd
    import pandas._libs.internals as pandas_internals
    import pandas.core.internals.blocks as pandas_blocks

    if "pandas.core.indexes.numeric" not in sys.modules:
        legacy_numeric_index = types.ModuleType("pandas.core.indexes.numeric")
        legacy_numeric_index.Int64Index = pd.Index
        legacy_numeric_index.UInt64Index = pd.Index
        legacy_numeric_index.Float64Index = pd.Index
        sys.modules["pandas.core.indexes.numeric"] = legacy_numeric_index

    original_new_block = pandas_blocks.new_block

    def compatible_new_block(values, placement, ndim, refs=None):
        if isinstance(placement, slice):
            placement = pandas_internals.BlockPlacement(placement)
        try:
            return original_new_block(values, placement, ndim=ndim, refs=refs)
        except TypeError:
            return original_new_block(values, placement, ndim, refs)

    if "ndim" in inspect.signature(original_new_block).parameters:
        pandas_blocks.new_block = compatible_new_block


def load_model():
    missing = [name for name in MODEL_FILES if not (BASE_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"Missing model artifact(s): {', '.join(missing)}")
    enable_legacy_pandas_pickle_support()
    artifacts = []
    for filename in MODEL_FILES:
        with (BASE_DIR / filename).open("rb") as model_file:
            artifacts.append(pickle.load(model_file))
    return artifacts


popular_books, pivot_table, books, similarity_scores = load_model()
book_metadata = books.drop_duplicates("Book-Title").set_index("Book-Title")[["Book-Author", "Image-URL-M"]]
available_titles = pivot_table.index.astype(str).tolist()
title_lookup = {title.casefold(): title for title in available_titles}

app = Flask(__name__)


def get_recommendations(title: str, limit: int = 6) -> list[dict[str, object]]:
    """Return the nearest collaborative-filtering neighbours for a title."""
    model_index = available_titles.index(title)
    ranked_indices = np.argsort(similarity_scores[model_index])[::-1]
    recommendations = []
    for candidate_index in ranked_indices:
        if candidate_index == model_index:
            continue
        candidate_title = available_titles[candidate_index]
        if candidate_title not in book_metadata.index:
            continue
        metadata = book_metadata.loc[candidate_title]
        recommendations.append({
            "title": candidate_title,
            "author": metadata["Book-Author"],
            "image": metadata["Image-URL-M"],
            "score": round(float(similarity_scores[model_index][candidate_index]) * 100),
        })
        if len(recommendations) == limit:
            break
    return recommendations


@app.get("/")
def index():
    popular = [{
        "title": row["Book-Title"], "author": row["Book-Author"], "image": row["Image-URL-M"],
        "ratings": int(row["num_ratings"]), "rating": round(float(row["avg_rating"]), 1),
    } for _, row in popular_books.iterrows()]
    return render_template("index.html", popular_books=popular)


@app.get("/recommend")
def recommend_page():
    return render_template("recommend.html", titles=available_titles)


@app.post("/recommend_books")
def recommend_books():
    submitted_title = request.form.get("user_input", "").strip()
    title = title_lookup.get(submitted_title.casefold())
    if not title:
        return render_template(
            "recommend.html", titles=available_titles, submitted_title=submitted_title,
            error="We could not find that book in the trained catalogue. Choose a title from the suggestions.",
        ), 404
    return render_template(
        "recommend.html", titles=available_titles, submitted_title=title,
        recommendations=get_recommendations(title),
    )


@app.get("/api/titles")
def title_search():
    query = request.args.get("q", "").strip().casefold()
    return jsonify([title for title in available_titles if query in title.casefold()][:12])


if __name__ == "__main__":
    app.run(debug=True)
