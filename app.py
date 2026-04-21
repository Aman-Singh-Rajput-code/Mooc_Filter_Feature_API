from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import nltk

from recommendation import recommend_courses
from data_processor import process_input
from sentiment_analyzer import analyze_sentiment

# NLTK
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

app = Flask(__name__)
CORS(app)

DATASET_PATH = "output.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("output.csv not found")

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "mooc-filter-feature-api"
    })

@app.route("/filter-courses", methods=["POST"])
def filter_courses():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    processed = process_input(data)
    sentiment = analyze_sentiment(processed["course_name"])

    results = recommend_courses(
        dataset_path=DATASET_PATH,
        filters=processed,
        sentiment=sentiment,
        top_n=data.get("top_n", 10)
    )

    return jsonify({
        "success": True,
        "sentiment": sentiment,
        "results": results,
        "total": len(results)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
