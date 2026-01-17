import pandas as pd
import ast
import os
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentiment_analyzer import SentimentAnalyzer

# ======================================================
# ✅ REQUIRED FOR RENDER (TEMP FILESYSTEM SAFE)
# ======================================================
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"


# ======================================================
# 🔒 JSON SANITIZER (CRITICAL)
# ======================================================
def sanitize_json(obj):
    """Remove NaN / Inf so Flask returns valid JSON"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(i) for i in obj]
    return obj


# ======================================================
# 🔗 COURSE LINK RESOLVER (FIXES YOUR ISSUE)
# ======================================================
def resolve_course_url(row):
    """
    Resolve course URL from multiple possible dataset columns
    """
    for col in ["course_url", "course_link", "url", "course_href", "source"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            return str(row[col]).strip()
    return ""


# ======================================================
# 🧠 COURSE RECOMMENDER
# ======================================================
class CourseRecommender:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.sentiment_analyzer = SentimentAnalyzer(use_distilroberta=False)
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english"
        )
        self.df = data_processor.get_course_data()
        self.prepare_features()

    def prepare_features(self):
        """Prepare TF-IDF features"""
        if self.df.empty:
            self.tfidf_matrix = None
            return

        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(
                self.df["combined_features"]
            )
        except Exception:
            self.tfidf_matrix = None

    def parse_comments(self, comments_str):
        """Safely parse comments column"""
        if pd.isna(comments_str) or comments_str == "":
            return []

        try:
            parsed = ast.literal_eval(comments_str)
            if isinstance(parsed, list):
                return parsed
            return [str(comments_str)]
        except Exception:
            return [str(comments_str)]

    def get_recommendations(self, user_input, top_n=10):
        """
        user_input keys:
        - course_name
        - is_paid (paid / free / any)
        - min_rating
        - user_comments
        """

        course_query = user_input.get("course_name", "")
        is_paid = user_input.get("is_paid", "any")
        min_rating = float(user_input.get("min_rating", 0.0))
        user_comments = user_input.get("user_comments", "")

        filtered_df = self.df.copy()

        # -----------------------------
        # Filters
        # -----------------------------
        if is_paid.lower() == "paid":
            filtered_df = filtered_df[
                filtered_df["is_paid"].str.lower() == "paid"
            ]
        elif is_paid.lower() == "free":
            filtered_df = filtered_df[
                filtered_df["is_paid"].str.lower() == "free"
            ]

        filtered_df = filtered_df[
            filtered_df["course_rating"] >= min_rating
        ]

        if filtered_df.empty:
            return []

        results = []

        # -----------------------------
        # Scoring
        # -----------------------------
        for _, row in filtered_df.iterrows():
            score = 0.0

            # 1️⃣ Rating (30%)
            rating = row.get("course_rating", 0)
            rating = 0 if pd.isna(rating) else float(rating)
            score += (rating / 5.0) * 0.30

            # 2️⃣ Text similarity (40%)
            if course_query:
                user_query = f"{course_query} {user_comments}"
                course_text = f"{row.get('course_name', '')} {row.get('instructor', '')}"

                try:
                    q_vec = self.vectorizer.transform([user_query])
                    c_vec = self.vectorizer.transform([course_text])
                    similarity = cosine_similarity(q_vec, c_vec)[0][0]
                    score += similarity * 0.40
                except Exception:
                    score += 0.20
            else:
                score += 0.20

            # 3️⃣ Sentiment (30%)
            comments_list = self.parse_comments(row.get("user_comments", ""))

            if comments_list:
                sentiment_score = self.sentiment_analyzer.get_sentiment_score(
                    " ".join(comments_list[:5])
                )
                score += sentiment_score * 0.30
            else:
                sentiment_score = 0.5
                score += 0.15

            # Enrollment
            enrollment = row.get("Number_of_student_enrolled", 0)
            enrollment = 0 if pd.isna(enrollment) else int(enrollment)

            results.append({
                "course_id": row.get("course_id", ""),
                "course_name": row.get("course_name", "Unknown"),
                "course_rating": round(rating, 2),
                "course_url": resolve_course_url(row),
                "instructor": row.get("instructor", "N/A"),
                "platform": row.get("platform", "N/A"),
                "is_paid": row.get("is_paid", "Unknown"),
                "enrollment": enrollment,
                "sentiment_score": round(sentiment_score, 3),
                "final_score": round(score, 3)
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_n]


# ======================================================
# 🚀 FLASK-SAFE WRAPPER (USED BY app.py)
# ======================================================
_recommender_instance = None

def recommend_courses(dataset_path, filters, sentiment=None, top_n=10):
    """
    Flask entry-point
    app.py imports THIS function
    """

    global _recommender_instance

    if _recommender_instance is None:
        from data_processor import DataProcessor
        processor = DataProcessor(dataset_path)
        _recommender_instance = CourseRecommender(processor)

    raw_results = _recommender_instance.get_recommendations(
        user_input=filters,
        top_n=top_n
    )

    return sanitize_json(raw_results)
