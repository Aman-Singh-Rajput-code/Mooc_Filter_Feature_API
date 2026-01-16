from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from recommendation import recommend_courses
from data_processor import process_input
from sentiment_analyzer import analyze_sentiment
import nltk

#-------------------------------------
#for nltk libraries
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ----------------------------------
# App setup
# ----------------------------------
app = Flask(__name__)

# ✅ Enable CORS for MERN / React
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
    """
    Request JSON:
    {
        "query": "machine learning",
        "platform": "Coursera",
        "min_rating": 4.5,
        "is_paid": "Free",
        "top_n": 10
    }
    """

    data = request.get_json()

    if not data:
        return jsonify({"error": "JSON body required"}), 400

    try:
        # 1️⃣ Process user input
        processed = process_input(data)

        # 2️⃣ Sentiment analysis (if query exists)
        sentiment = analyze_sentiment(processed.get("query", ""))

        # 3️⃣ Recommendation logic
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
        return jsonify({
            "error": "Failed to filter courses"
        }), 500

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
# Local run (Render uses gunicorn)
# ----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
