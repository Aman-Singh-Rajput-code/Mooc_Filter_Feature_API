import pandas as pd
import ast
import math
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentiment_analyzer import SentimentAnalyzer

# ============================
# Render safe cache
# ============================
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"


# ============================
# JSON SANITIZER
# ============================
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


# ============================
# COURSE URL EXTRACTOR 🔥
# ============================
def extract_course_url(sources):
    if pd.isna(sources) or sources == "":
        return ""

    if isinstance(sources, str):
        if sources.startswith("http"):
            return sources
        try:
            parsed = ast.literal_eval(sources)
            return extract_course_url(parsed)
        except:
            return ""

    if isinstance(sources, list) and sources:
        first = sources[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("url", "")

    if isinstance(sources, dict):
        return sources.get("url", "")

    return ""


# ============================
# RECOMMENDER CLASS
# ============================
class CourseRecommender:
    def __init__(self, data_processor):
        self.df = data_processor.get_course_data()
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english"
        )
        self.sentiment_analyzer = SentimentAnalyzer(use_distilroberta=False)
        self.prepare_features()

    def prepare_features(self):
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
        except:
            pass
        return [str(comments)]

    def get_recommendations(self, user_input, top_n=10):
        filtered = self.df.copy()

        # Filters
        if user_input["is_paid"].lower() != "any":
            filtered = filtered[
                filtered["is_paid"].str.lower() ==
                user_input["is_paid"].lower()
            ]

        filtered = filtered[
            filtered["course_rating"] >= user_input["min_rating"]
        ]

        results = []

        for _, row in filtered.iterrows():
            score = 0

            # Rating
            score += (row["course_rating"] / 5) * 0.3

            # Similarity
            if user_input["course_name"]:
                try:
                    q_vec = self.vectorizer.transform(
                        [user_input["course_name"]]
                    )
                    c_vec = self.vectorizer.transform(
                        [row["course_name"]]
                    )
                    sim = cosine_similarity(q_vec, c_vec)[0][0]
                    score += sim * 0.4
                except:
                    score += 0.2
            else:
                score += 0.2

            # Sentiment
            comments = self.parse_comments(row["user_comments"])
            sentiment = (
                self.sentiment_analyzer.get_sentiment_score(
                    " ".join(comments[:5])
                ) if comments else 0.5
            )
            score += sentiment * 0.3

            results.append({
                "course_id": row["course_id"],
                "course_name": row["course_name"],
                "course_rating": round(row["course_rating"], 2),
                "instructor": row.get("instructor", ""),
                "platform": row.get("platform", ""),
                "is_paid": row["is_paid"],
                "enrollment": int(row["Number_of_student_enrolled"]),
                "course_url": extract_course_url(row.get("sources")),
                "sentiment_score": round(sentiment, 3),
                "final_score": round(score, 3)
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:top_n]


# ============================
# FLASK ENTRY FUNCTION
# ============================
_recommender = None

def recommend_courses(dataset_path, filters, sentiment=None, top_n=10):
    global _recommender

    if _recommender is None:
        from data_processor import DataProcessor
        processor = DataProcessor(dataset_path)
        _recommender = CourseRecommender(processor)

    return sanitize_json(
        _recommender.get_recommendations(filters, top_n)
    )
