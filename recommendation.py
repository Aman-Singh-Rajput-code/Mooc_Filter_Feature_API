import pandas as pd
import ast
import os
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentiment_analyzer import SentimentAnalyzer

# Render-safe cache
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"


# ------------------------------
# Utility Functions
# ------------------------------

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


def extract_course_url(sources):
    if sources is None or sources == "":
        return ""

    try:
        if isinstance(sources, str):
            parsed = ast.literal_eval(sources)
        else:
            parsed = sources

        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], str) and parsed[0].startswith("http"):
                return parsed[0]
    except Exception:
        pass

    return ""


# 🔥 NEW: Skill Extraction
def extract_skills(text):
    skills_list = [
        "python", "machine learning", "data science", "deep learning",
        "sql", "excel", "ai", "nlp", "statistics", "power bi"
    ]

    text = str(text).lower()
    return [skill for skill in skills_list if skill in text]


# ------------------------------
# Main Recommender Class
# ------------------------------

class CourseRecommender:
    def __init__(self, data_processor):
        self.df = data_processor.get_course_data()
        self.sentiment_analyzer = SentimentAnalyzer(use_distilroberta=False)

        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words="english"
        )

        self.prepare_features()

    def prepare_features(self):
        if self.df.empty:
            self.tfidf_matrix = None
            return

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["combined_features"]
        )

    def parse_comments(self, comments):
        if pd.isna(comments) or comments == "":
            return []

        try:
            parsed = ast.literal_eval(comments)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        return [str(comments)]

    def get_recommendations(self, user_input, top_n=10):
        query = user_input.get("course_name", "")
        is_paid = user_input.get("is_paid", "any").lower()
        min_rating = float(user_input.get("min_rating", 0.0))
        user_comments = user_input.get("user_comments", "")

        df = self.df.copy()

        # 🔹 Apply Filters
        if is_paid in ["paid", "free"]:
            df = df[df["is_paid"].str.lower() == is_paid]

        df = df[df["course_rating"] >= min_rating]

        results = []

        # 🔥 OPTIMIZATION: compute query vector once
        try:
            q_vec = self.vectorizer.transform([f"{query} {user_comments}"])
        except Exception:
            q_vec = None

        for _, row in df.iterrows():
            rating = float(row["course_rating"])

            # 🔥 Similarity Calculation
            try:
                if q_vec is not None:
                    c_vec = self.vectorizer.transform(
                        [f"{row['course_name']} {row.get('instructor', '')}"]
                    )
                    similarity = cosine_similarity(q_vec, c_vec)[0][0]
                else:
                    similarity = 0.2
            except Exception:
                similarity = 0.2

            # 🔥 Sentiment Calculation
            comments = self.parse_comments(row.get("user_comments", ""))
            sentiment = (
                self.sentiment_analyzer.get_sentiment_score(
                    " ".join(comments[:5])
                )
                if comments else 0.5
            )

            # 🔥 Final Hybrid Score
            score = (
                (rating / 5.0) * 0.30 +
                similarity * 0.40 +
                sentiment * 0.30
            )

            # 🔥 Skill Extraction
            skills = extract_skills(row["course_name"])

            # 🔥 Result Object (Frontend Ready)
            results.append({
                "course_id": row["course_id"],
                "title": row["course_name"],
                "rating": round(rating, 2),
                "instructor": row.get("instructor", "N/A"),
                "platform": row.get("platform", "N/A"),
                "is_paid": row.get("is_paid", "Unknown"),
                "enrollment": int(row.get("Number_of_student_enrolled", 0)),
                "course_url": extract_course_url(row.get("sources")),

                # 🔥 KEY FIELDS FOR VISUALIZATION
                "similarity": round(float(similarity), 3),
                "sentiment_score": round(sentiment, 3),
                "final_score": round(score, 3),
                "skills": skills,

                # 🔥 Explainability (VIVA GOLD)
                "why_recommended": f"Recommended because it matches your input with similarity {round(similarity,2)} and has rating {rating}"
            })

        # 🔥 Sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)

        return sanitize_json(results[:top_n])


# ------------------------------
# Singleton Wrapper
# ------------------------------

_recommender = None


def recommend_courses(dataset_path, filters, sentiment=None, top_n=10):
    global _recommender

    if _recommender is None:
        from data_processor import DataProcessor
        _recommender = CourseRecommender(
            DataProcessor(dataset_path)
        )

    return _recommender.get_recommendations(filters, top_n)
