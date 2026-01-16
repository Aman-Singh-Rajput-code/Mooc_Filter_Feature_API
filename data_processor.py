import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DataProcessor:
    def __init__(self, csv_path='output.csv'):
        self.df = None
        self.csv_path = csv_path
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the dataset"""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.preprocess_data()
        except FileNotFoundError:
            print(f"Error: {self.csv_path} not found!")
            self.df = pd.DataFrame()
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        # Handle missing values
        self.df['course_rating'] = pd.to_numeric(self.df['course_rating'], errors='coerce')
        self.df['course_rating'].fillna(self.df['course_rating'].mean(), inplace=True)
        
        # Clean is_paid column
        self.df['is_paid'] = self.df['is_paid'].fillna('Unknown')
        self.df['is_paid'] = self.df['is_paid'].str.strip().str.title()
        
        # Handle user_comments
        self.df['user_comments'] = self.df['user_comments'].fillna('')
        
        # Create a combined feature for text analysis
        self.df['combined_features'] = (
            self.df['course_name'].fillna('') + ' ' +
            self.df['instructor'].fillna('') + ' ' +
            self.df['user_comments'].fillna('')
        )
        
    def get_course_data(self):
        """Return processed dataframe"""
        return self.df
    
    def filter_courses(self, is_paid=None, min_rating=None):
        """Filter courses based on criteria"""
        filtered_df = self.df.copy()
        
        if is_paid is not None:
            if is_paid.lower() == 'paid':
                filtered_df = filtered_df[filtered_df['is_paid'] == 'Paid']
            elif is_paid.lower() == 'free':
                filtered_df = filtered_df[filtered_df['is_paid'] == 'Free']
        
        if min_rating is not None:
            filtered_df = filtered_df[filtered_df['course_rating'] >= min_rating]
        
        return filtered_df
    
    def search_courses(self, query):
        """Search courses by name or keywords"""
        if not query:
            return self.df
        
        query = query.lower()
        mask = self.df['course_name'].str.lower().str.contains(query, na=False)
        return self.df[mask]
    
    def get_course_by_id(self, course_id):
        """Get a specific course by ID"""
        course = self.df[self.df['course_id'] == course_id]
        if not course.empty:
            return course.iloc[0].to_dict()
        return None