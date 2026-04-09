#safe file
'''
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
'''

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
        df = self.df

        # ------------------------------
        # NUMERIC CLEANING
        # ------------------------------
        df["course_rating"] = pd.to_numeric(
            df.get("course_rating", 0), errors="coerce"
        ).fillna(0)

        df["Number_of_student_enrolled"] = pd.to_numeric(
            df.get("Number_of_student_enrolled", 0), errors="coerce"
        ).fillna(0)

        # ------------------------------
        # BASIC TEXT CLEANING
        # ------------------------------
        df["is_paid"] = df.get("is_paid", "Unknown").fillna("Unknown").astype(str)

        df["course_name"] = df.get("course_name", "").fillna("")
        df["instructor"] = df.get("instructor", "").fillna("")
        df["user_comments"] = df.get("user_comments", "").fillna("")

        # 🔥 IMPORTANT FIELDS (CHECK EXISTENCE)
        df["description"] = df.get("description", "").fillna("")
        df["category"] = df.get("category", "").fillna("")
        df["skills"] = df.get("skills", "").fillna("")

        # ------------------------------
        # OPTIONAL: PARSE SKILLS (if stored as list string)
        # ------------------------------
        def clean_skills(x):
            try:
                if isinstance(x, str) and x.startswith("["):
                    parsed = ast.literal_eval(x)
                    if isinstance(parsed, list):
                        return " ".join(parsed)
            except:
                pass
            return str(x)

        df["skills"] = df["skills"].apply(clean_skills)

        # ------------------------------
        # 🔥 FINAL COMBINED FEATURES (KEY FIX)
        # ------------------------------
        df["combined_features"] = (
            df["course_name"] + " " +
            df["description"] + " " +
            df["skills"] + " " +
            df["category"] + " " +
            df["instructor"] + " " +
            df["user_comments"]
        )

        # ------------------------------
        # LOWERCASE (IMPORTANT FOR NLP)
        # ------------------------------
        df["combined_features"] = df["combined_features"].str.lower()

        self.df = df

    def get_course_data(self):
        return self.df


# ================================
# INPUT NORMALIZER FOR API
# ================================
def process_input(data: dict):
    query = data.get("course_name") or data.get("query") or ""

    return {
        "course_name": query.lower(),
        "is_paid": data.get("is_paid", "any"),
        "min_rating": float(data.get("min_rating", 0)),
        "user_comments": query.lower()
    }
