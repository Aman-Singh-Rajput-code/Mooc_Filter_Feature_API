import pandas as pd
import numpy as np
import re


class DataProcessor:
    def __init__(self, csv_path="output.csv"):
        self.csv_path = csv_path
        self.df = pd.DataFrame()
        self.load_data()

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    def load_data(self):
        try:
            self.df = pd.read_csv(self.csv_path)
            self.preprocess_data()
        except FileNotFoundError:
            print(f"❌ Error: {self.csv_path} not found")
            self.df = pd.DataFrame()

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    def preprocess_data(self):
        if self.df.empty:
            return

        # Normalize column names
        self.df.columns = [c.strip() for c in self.df.columns]

        # Course rating
        self.df["course_rating"] = pd.to_numeric(
            self.df.get("course_rating", 0), errors="coerce"
        )
        self.df["course_rating"] = self.df["course_rating"].fillna(
            self.df["course_rating"].mean()
        )

        # Payment type
        self.df["is_paid"] = (
            self.df.get("is_paid", "Unknown")
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.title()
        )

        # Comments
        self.df["user_comments"] = self.df.get("user_comments", "").fillna("")

        # Instructor
        self.df["instructor"] = self.df.get("instructor", "").fillna("")

        # Course name
        self.df["course_name"] = self.df.get("course_name", "").fillna("")

        # Platform
        self.df["platform"] = self.df.get("platform", "Unknown").fillna("Unknown")

        # Combined text features (🔥 REQUIRED BY RECOMMENDER)
        self.df["combined_features"] = (
            self.df["course_name"] + " " +
            self.df["instructor"] + " " +
            self.df["user_comments"]
        )

    # --------------------------------------------------
    # Accessors
    # --------------------------------------------------
    def get_course_data(self):
        return self.df

    def get_course_by_id(self, course_id):
        if self.df.empty:
            return None

        course = self.df[self.df["course_id"] == course_id]
        if course.empty:
            return None

        return course.iloc[0].to_dict()

    # --------------------------------------------------
    # Filters
    # --------------------------------------------------
    def filter_courses(self, is_paid=None, min_rating=None):
        filtered_df = self.df.copy()

        if is_paid:
            if is_paid.lower() == "paid":
                filtered_df = filtered_df[filtered_df["is_paid"] == "Paid"]
            elif is_paid.lower() == "free":
                filtered_df = filtered_df[filtered_df["is_paid"] == "Free"]

        if min_rating is not None:
            filtered_df = filtered_df[filtered_df["course_rating"] >= float(min_rating)]

        return filtered_df

    def search_courses(self, query):
        if not query:
            return self.df

        query = query.lower()
        return self.df[
            self.df["course_name"].str.lower().str.contains(query, na=False)
        ]


# ====================================================
# ✅ REQUIRED BY app.py (THIS FIXES YOUR CRASH)
# ====================================================
def process_input(data: dict) -> dict:
    """
    Normalize incoming API JSON for recommendation engine
    """

    return {
        "course_name": data.get("query", ""),
        "platform": data.get("platform", "any"),
        "min_rating": float(data.get("min_rating", 0)),
        "is_paid": data.get("is_paid", "any"),
        "user_comments": data.get("user_comments", ""),
    }
