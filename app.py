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
    print("DATA:", data)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    processed = process_input(data)
    print("PROCESSED:", processed)
    sentiment = analyze_sentiment(processed.get("course_name", ""))

    results = recommend_courses(
        dataset_path=DATASET_PATH,
        filters=processed,
        sentiment=sentiment,
        top_n=data.get("top_n", 10)
    )
    print("RESULTS:", results) 

    # 🔥 NEW: Transform results for visualization
    enhanced_results = []

    for course in results:
        print("COURSE:", course)
        enhanced_results.append({
            "course_name": course.get("title", ""),
            "similarity": float(course.get("similarity", 0)),  # IMPORTANT
            "rating": float(course.get("rating", 0)),
            "platform": course.get("platform", "N/A"),
            "is_paid": course.get("is_paid", "Unknown"),
            "duration": course.get("duration", "N/A"),
            "level": course.get("level", "N/A"),

            # 🔥 Optional: extract skills (simple version)
            "skills": course.get("skills", []),
            "course_url": course.get("course_url", ""),

            "final_score": course.get("final_score", 0),
            

            # 🔥 Explainability (VERY IMPORTANT)
            "why_recommended": f"Matches your interest with similarity score {round(course.get('similarity', 0), 2)}"
        })

    # 🔥 Graph Data (ready for frontend)
    graph_data = {
        "labels": [c["course_name"] for c in enhanced_results],
        "similarity_scores": [c["similarity"] for c in enhanced_results],
        "ratings": [c["rating"] for c in enhanced_results]
    }

    return jsonify({
        "success": True,
        "sentiment": sentiment,
        "total": len(enhanced_results),

        # 🔥 Main Data
        "courses": enhanced_results,

        # 🔥 Visualization Ready
        "graph_data": graph_data
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
