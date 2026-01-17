import pandas as pd
import ast
import os
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentiment_analyzer import SentimentAnalyzer

# ======================================================
# ✅ RENDER SAFE ENV
# ======================================================
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"


# ======================================================
# 🔒 JSON SANITIZER
# ======================================================
def sanitize_json(obj):
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
# 🔗 COURSE LINK NORMALIZER
# ======================================================
def extract_course_url(row):
    for key in ["course_url", "sources", "course_link", "url", "course_href"]:
        val = row.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val
    return ""


# ======================================================
# 🧠 COURSE RECOMMENDER
# ======================================================
class CourseRecommender:
    def __init__(self, data_processor):
        self.df = data_processor.get_course_data()
        self.sentiment_analyzer = SentimentAnalyzer(use_distilroberta=False)
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english"
        )
        self._prepare_features()

    def _prepare_features(self):
        if self.df.empty:
            self.tfidf_matrix = None
            return
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["combined_features"]
        )

    def _parse_comments(self, comments):
        if pd.isna(comments) or comments == "":
            return []
        try:
            parsed = ast.literal_eval(comments)
            return parsed if isinstance(parsed, list) else [str(comments)]
        except Exception:
            return [str(comments)]

    def get_recommendations(self, user_input, top_n=10):
        course_query = user_input.get("course_name", "")
        is_paid = user_input.get("is_paid", "any").lower()
        min_rating = float(user_input.get("min_rating", 0))
        user_comments = user_input.get("user_comments", "")

        df = self.df.copy()

        # -----------------------------
        # Filters
        # -----------------------------
        if is_paid in ["paid", "free"]:
            df = df[df["is_paid"].str.lower() == is_paid]

        df = df[df["course_rating"] >= min_rating]

        if df.empty:
            return []

        results = []

        for _, row in df.iterrows():
            rating = row.get("course_rating", 0)
            rating = 0 if pd.isna(rating) else float(rating)

            enrollment = row.get("Number_of_student_enrolled", 0)
            enrollment = 0 if pd.isna(enrollment) else int(enrollment)

            # -----------------------------
            # Similarity
            # -----------------------------
            similarity = 0.0
            if course_query:
                try:
                    user_vec = self.vectorizer.transform(
                        [course_query + " " + user_comments]
                    )
                    course_vec = self.vectorizer.transform(
                        [row.get("course_name", "") + " " + str(row.get("instructor", ""))]
                    )
                    similarity = cosine_similarity(user_vec, course_vec)[0][0]
                except Exception:
                    similarity = 0.0

            # -----------------------------
            # Sentiment
            # -----------------------------
            comments = self._parse_comments(row.get("user_comments", ""))
            sentiment_score = (
                self.sentiment_analyzer.get_sentiment_score(" ".join(comments[:5]))
                if comments else 0.5
            )

            # -----------------------------
            # Final Score
            # -----------------------------
            final_score = (
                similarity * 0.40 +
                (rating / 5.0) * 0.35 +
                sentiment_score * 0.25
            )

            results.append({
                "course_id": row.get("course_id", ""),
                "course_name": row.get("course_name", "Unknown"),
                "course_rating": round(rating, 2),
                "instructor": row.get("instructor", "N/A"),
                "platform": row.get("platform", "N/A"),
                "is_paid": row.get("is_paid", "Unknown"),
                "enrollment": enrollment,
                "sentiment_score": round(sentiment_score, 3),
                "final_score": round(final_score, 3),

                # ✅ LINK FIXED
                "course_url": extract_course_url(row)
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_n]


# ======================================================
# 🚀 FLASK ENTRY POINT
# ======================================================
_recommender_instance = None

def recommend_courses(dataset_path, filters, sentiment=None, top_n=10):
    global _recommender_instance

    if _recommender_instance is None:
        from data_processor import DataProcessor
        processor = DataProcessor(dataset_path)
        _recommender_instance = CourseRecommender(processor)

    raw = _recommender_instance.get_recommendations(filters, top_n)
    return sanitize_json(raw)
