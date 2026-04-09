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
        parsed = ast.literal_eval(sources) if isinstance(sources, str) else sources

        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, str) and item.startswith("http"):
                    return item
    except:
        pass

    return ""


# 🔥 Improved Skill Extraction
def extract_skills(text):
    skills_list = [
        "python", "machine learning", "data science", "deep learning",
        "html", "css", "javascript", "react", "node",
        "web development", "frontend", "backend",
        "cyber security", "network security", "sql", "excel"
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
            max_features=1500,
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
        except:
            pass

        return [str(comments)]

    def get_recommendations(self, user_input, top_n=10):
        query = user_input.get("course_name", "").lower()
        is_paid = user_input.get("is_paid", "any").lower()
        min_rating = float(user_input.get("min_rating", 0.0))
        user_comments = user_input.get("user_comments", "")

        df = self.df.copy()

        # ------------------------------
        # FILTERS
        # ------------------------------
        if is_paid in ["paid", "free"]:
            df = df[df["is_paid"].str.lower() == is_paid]

        df = df[df["course_rating"] >= min_rating]

        # 🔥 SMART QUERY FILTER
        if query:
            keywords = query.split()

            mask = df["combined_features"].apply(
                lambda x: any(word in str(x).lower() for word in keywords)
            )

            df = df[mask]

        # fallback
        if df.empty:
            df = self.df.copy()

        results = []

        # ------------------------------
        # QUERY VECTOR
        # ------------------------------
        try:
            q_vec = self.vectorizer.transform(
                [f"{query} {query} {query} {user_comments}"]
            )
        except:
            q_vec = None

        # ------------------------------
        # MAIN LOOP
        # ------------------------------
        for _, row in df.iterrows():
            rating = float(row.get("course_rating", 0))

            # 🔥 SIMILARITY
            try:
                if q_vec is not None:
                    c_vec = self.vectorizer.transform(
                        [str(row["combined_features"])]
                    )
                    similarity = cosine_similarity(q_vec, c_vec)[0][0]
                else:
                    similarity = 0.2
            except:
                similarity = 0.2

            # 🔥 KEYWORD BOOST
            if any(word in str(row["combined_features"]).lower() for word in query.split()):
                similarity += 0.2

            similarity = min(similarity, 1.0)

            # 🔥 SENTIMENT (IMPROVED)
            comments = self.parse_comments(row.get("user_comments", ""))

            text_for_sentiment = (
                " ".join(comments[:5]) + " " +
                str(row.get("course_name", "")) + " " +
                str(row.get("combined_features", ""))[:200]
            )

            try:
                sentiment = self.sentiment_analyzer.get_sentiment_score(text_for_sentiment)
                print("FINAL SENTIMENT:", sentiment)
                if sentiment is None:
                    sentiment = 0.4
            except:
                sentiment = 0.4

            # ------------------------------
            # FINAL SCORE
            # ------------------------------
            score = (
                (rating / 5.0) * 0.30 +
                similarity * 0.40 +
                sentiment * 0.30
            )

            # 🔥 SKILLS
            skills = extract_skills(row.get("combined_features", ""))

            # ------------------------------
            # RESULT OBJECT
            # ------------------------------
            results.append({
                "course_id": row.get("course_id", ""),
                "title": row.get("course_name", ""),
                "rating": round(rating, 2),
                "instructor": row.get("instructor", "N/A"),
                "platform": row.get("platform", "N/A"),
                "is_paid": row.get("is_paid", "Unknown"),
                "enrollment": int(row.get("Number_of_student_enrolled", 0)),
                "course_url": extract_course_url(row.get("sources")),

                "similarity": round(float(similarity), 3),
                "sentiment_score": round(float(sentiment), 3),
                "final_score": round(float(score), 3),
                "skills": skills,

                "why_recommended": f"Recommended because it matches your query with similarity {round(similarity,2)} and has rating {rating}"
            })

        # ------------------------------
        # SORT
        # ------------------------------
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
