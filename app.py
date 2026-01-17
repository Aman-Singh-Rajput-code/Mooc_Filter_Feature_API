from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import nltk

from recommendation import recommend_courses
from data_processor import process_input
from sentiment_analyzer import analyze_sentiment

# ----------------------------------
# NLTK SETUP
# ----------------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ----------------------------------
# App setup
# ----------------------------------
app = Flask(__name__)
CORS(app)

DATASET_PATH = "output.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("output.csv not found")

print("MOOC Filter Feature API ready!")

# ----------------------------------
# Health check
# ----------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "mooc-filter-feature-api"
    }), 200

# ----------------------------------
# Filter Courses API
# ----------------------------------
@app.route("/filter-courses", methods=["POST"])
def filter_courses():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    try:
        processed = process_input(data)
        sentiment = analyze_sentiment(processed.get("query", ""))

        results = recommend_courses(
            dataset_path=DATASET_PATH,
            filters=processed,
            sentiment=sentiment
        )

        return jsonify({
            "success": True,
            "sentiment": sentiment,
            "results": results,
            "total": len(results)
        }), 200

    except Exception as e:
        print("Filter error:", e)
        return jsonify({"error": "Failed to filter courses"}), 500

# ----------------------------------
# Error handlers
# ----------------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

# ----------------------------------
# Local run
# ----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
