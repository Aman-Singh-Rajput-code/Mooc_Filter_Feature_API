import pandas as pd
import ast


class DataProcessor:
    def __init__(self, csv_path="output.csv"):
        self.csv_path = csv_path
        self.df = None
        self.load_data()

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.preprocess_data()

    def preprocess_data(self):
        # Ratings
        self.df["course_rating"] = pd.to_numeric(
            self.df["course_rating"], errors="coerce"
        ).fillna(0)

        # Enrollment
        self.df["Number_of_student_enrolled"] = pd.to_numeric(
            self.df["Number_of_student_enrolled"], errors="coerce"
        ).fillna(0)

        # Paid / Free
        self.df["is_paid"] = self.df["is_paid"].fillna("Unknown").astype(str)

        # Comments
        self.df["user_comments"] = self.df["user_comments"].fillna("")

        # Combined text for NLP
        self.df["combined_features"] = (
            self.df["course_name"].fillna("") + " " +
            self.df["instructor"].fillna("") + " " +
            self.df["user_comments"].fillna("")
        )

    def get_course_data(self):
        return self.df


# ================================
# INPUT NORMALIZER FOR API
# ================================
def process_input(data: dict):
    return {
        "course_name": data.get("query", ""),
        "is_paid": data.get("is_paid", "any"),
        "min_rating": float(data.get("min_rating", 0)),
        "user_comments": data.get("query", "")
    }
