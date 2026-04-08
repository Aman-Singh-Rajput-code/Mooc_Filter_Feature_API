from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import nltk

from recommendation import recommend_courses
from data_processor import process_input
from sentiment_analyzer import analyze_sentiment

# NLTK downloads
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

app = Flask(__name__)

# ✅ FIXED CORS (VERY IMPORTANT)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ HANDLE PREFLIGHT REQUESTS (CRITICAL FOR RENDER)
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


DATASET_PATH = "output.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("output.csv not found")


# ------------------------------
# HEALTH CHECK
# ------------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "mooc-filter-feature-api"
    })


# ------------------------------
# MAIN API
# ------------------------------
@app.route("/filter-courses", methods=["POST", "OPTIONS"])
def filter_courses():

    # ✅ Handle preflight request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    # Process input
    processed = process_input(data)

    # Sentiment
    sentiment = analyze_sentiment(processed.get("course_name", ""))

    # Recommendation
    results = recommend_courses(
        dataset_path=DATASET_PATH,
        filters=processed,
        sentiment=sentiment,
        top_n=data.get("top_n", 10)
    )

    # ------------------------------
    # 🔥 TRANSFORM RESULTS
    # ------------------------------
    enhanced_results = []

    for course in results:
        enhanced_results.append({
            "course_name": course.get("title", ""),
            "similarity": float(course.get("similarity", 0)),
            "rating": float(course.get("rating", 0)),
            "platform": course.get("platform", "N/A"),
            "is_paid": course.get("is_paid", "Unknown"),
            "skills": course.get("skills", []),

            # Explainability
            "why_recommended": f"Matches your interest with similarity {round(course.get('similarity', 0), 2)}"
        })

    # ------------------------------
    # 🔥 GRAPH DATA
    # ------------------------------
    graph_data = {
        "labels": [c["course_name"] for c in enhanced_results],
        "similarity_scores": [c["similarity"] for c in enhanced_results],
        "ratings": [c["rating"] for c in enhanced_results]
    }

    # ------------------------------
    # RESPONSE
    # ------------------------------
    return jsonify({
        "success": True,
        "sentiment": sentiment,
        "total": len(enhanced_results),
        "courses": enhanced_results,
        "graph_data": graph_data
    })


# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
