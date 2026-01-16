import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentiment_analyzer import SentimentAnalyzer
import ast
import os

# ✅ REQUIRED FOR RENDER
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


class CourseRecommender:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.sentiment_analyzer = SentimentAnalyzer(use_distilroberta=False)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.df = data_processor.get_course_data()
        self.prepare_features()
    
    def prepare_features(self):
        """Prepare features for recommendation"""
        if self.df.empty:
            return
        
        # Create TF-IDF matrix for course features
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df['combined_features'])
        except:
            self.tfidf_matrix = None
    
    def parse_comments(self, comments_str):
        """Parse comments string to list"""
        if pd.isna(comments_str) or comments_str == '':
            return []
        
        try:
            # Try to parse as list
            if isinstance(comments_str, str):
                comments = ast.literal_eval(comments_str)
                if isinstance(comments, list):
                    return comments
            return [str(comments_str)]
        except:
            return [str(comments_str)]
    
    def get_recommendations(self, user_input, top_n=10):
        """
        Get course recommendations based on user input
        
        user_input: dict with keys:
            - course_name: preferred course topics (string)
            - is_paid: 'paid', 'free', or 'any' (string)
            - min_rating: minimum rating (float)
            - user_comments: user's additional preferences (string)
        """
        # Extract user preferences
        course_query = user_input.get('course_name', '')
        is_paid = user_input.get('is_paid', 'any')
        min_rating = float(user_input.get('min_rating', 0.0))
        user_comments = user_input.get('user_comments', '')
        
        # Filter courses based on criteria
        filtered_df = self.df.copy()
        
        # Filter by payment type
        if is_paid.lower() == 'paid':
            filtered_df = filtered_df[filtered_df['is_paid'].str.lower() == 'paid']
        elif is_paid.lower() == 'free':
            filtered_df = filtered_df[filtered_df['is_paid'].str.lower() == 'free']
        
        # Filter by minimum rating
        filtered_df = filtered_df[filtered_df['course_rating'] >= min_rating]
        
        if filtered_df.empty:
            return []
        
        # Calculate scores for each course
        scores = []
        
        for idx, row in filtered_df.iterrows():
            score = 0.0
            
            # 1. Rating score (30% weight)
            rating_score = row['course_rating'] / 5.0
            score += rating_score * 0.30
            
            # 2. Text similarity score (40% weight)
            if course_query:
                user_query = course_query + ' ' + user_comments
                course_text = row['course_name'] + ' ' + row.get('instructor', '')
                
                try:
                    query_vec = self.vectorizer.transform([user_query])
                    course_vec = self.vectorizer.transform([course_text])
                    similarity = cosine_similarity(query_vec, course_vec)[0][0]
                    score += similarity * 0.40
                except:
                    score += 0.20  # neutral score if vectorization fails
            else:
                score += 0.20
            
            # 3. Sentiment score from user comments (30% weight)
            comments_list = self.parse_comments(row['user_comments'])
            if comments_list:
                sentiment_analysis = self.sentiment_analyzer.analyze_comments(comments_list)
                sentiment_score = self.sentiment_analyzer.get_sentiment_score(
                    ' '.join(comments_list[:5])  # Use first 5 comments
                )
                score += sentiment_score * 0.30
            else:
                score += 0.15  # neutral score if no comments
            
            scores.append({
                'index': idx,
                'score': score,
                'course_id': row['course_id'],
                'course_name': row['course_name'],
                'course_rating': row['course_rating'],
                'instructor': row.get('instructor', 'N/A'),
                'is_paid': row['is_paid'],
                'platform': row.get('platform', 'N/A'),
                'sentiment_score': sentiment_score if comments_list else 0.5,
                'enrollment': row.get('Number_of_student_enrolled', 'N/A')
            })
        
        # Sort by score and return top N
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_n]
    
    def get_similar_courses(self, course_id, top_n=5):
        """Get courses similar to a given course"""
        if self.tfidf_matrix is None:
            return []
        
        # Find course index
        course_indices = self.df[self.df['course_id'] == course_id].index
        if len(course_indices) == 0:
            return []
        
        idx = course_indices[0]
        
        # Calculate similarity scores
        cosine_similarities = cosine_similarity(
            self.tfidf_matrix[idx:idx+1], 
            self.tfidf_matrix
        ).flatten()
        
        # Get top similar courses (excluding the course itself)
        similar_indices = cosine_similarities.argsort()[::-1][1:top_n+1]
        
        similar_courses = []
        for i in similar_indices:
            course = self.df.iloc[i]
            similar_courses.append({
                'course_id': course['course_id'],
                'course_name': course['course_name'],
                'course_rating': course['course_rating'],
                'is_paid': course['is_paid'],
                'similarity_score': round(cosine_similarities[i], 3)
            })
        
        return similar_courses
    
    def get_course_details_with_sentiment(self, course_id):
        """Get detailed course information with sentiment analysis"""
        course = self.data_processor.get_course_by_id(course_id)
        if not course:
            return None
        
        # Analyze sentiment of comments
        comments_list = self.parse_comments(course.get('user_comments', ''))
        sentiment_analysis = self.sentiment_analyzer.analyze_comments(comments_list)
        
        course['sentiment_analysis'] = sentiment_analysis
        course['sample_comments'] = comments_list[:5] if comments_list else []
        
        return course
