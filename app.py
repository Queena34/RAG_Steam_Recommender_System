import threading
from typing import Any

from flask import Flask, jsonify, render_template, request
from recommender import create_search_engine


def create_app() -> Flask:
    app = Flask(__name__)
    # Lazy initialization prevents Flask's debug reloader parent process from
    # loading FAISS, BM25, and Ollama before the serving child starts.
    search_engine = None
    search_engine_lock = threading.Lock()

    def get_search_engine():
        nonlocal search_engine
        if search_engine is None:
            with search_engine_lock:
                if search_engine is None:
                    search_engine = create_search_engine()
        return search_engine

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/api/search")
    def search() -> Any:
        payload = request.get_json(silent=True) or {}
        raw_query = payload.get("query")
        query = raw_query.strip() if isinstance(raw_query, str) else ""
        if not query:
            return jsonify({"error": "A game description is required."}), 400
        try:
            return jsonify(get_search_engine().search(query))
        except Exception:
            app.logger.exception("Search failed")
            return jsonify({"error": "Search failed."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
