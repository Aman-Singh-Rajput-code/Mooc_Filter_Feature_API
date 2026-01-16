import re
from textblob import TextBlob
import numpy as np

# For future DistilRoBERTa implementation (currently using TextBlob for simplicity)
# Uncomment these when you want to use DistilRoBERTa:
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

class SentimentAnalyzer:
    def __init__(self, use_distilroberta=False):
        """
        Initialize sentiment analyzer
        use_distilroberta: Set to True to use DistilRoBERTa model
        """
        self.use_distilroberta = use_distilroberta
        
        if self.use_distilroberta:
            # Uncomment when ready to use DistilRoBERTa
            # self.tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
            # self.model = AutoModelForSequenceClassification.from_pretrained(
            #     "distilbert-base-uncased-finetuned-sst-2-english"
            # )
            print("DistilRoBERTa mode enabled (using TextBlob as placeholder)")
        
    def clean_text(self, text):
        """Clean text for sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def analyze_sentiment_textblob(self, text):
        """
        Analyze sentiment using TextBlob (lightweight, no model download needed)
        Returns: sentiment score (-1 to 1) and label
        """
        if not text:
            return 0.0, "neutral"
        
        cleaned_text = self.clean_text(text)
        blob = TextBlob(cleaned_text)
        polarity = blob.sentiment.polarity
        
        # Classify sentiment
        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"
        
        return polarity, label
    
    def analyze_sentiment_distilroberta(self, text):
        """
        Analyze sentiment using DistilRoBERTa (for future implementation)
        This will provide more accurate sentiment analysis
        """
        # FUTURE IMPLEMENTATION:
        # Uncomment this code when you're ready to use DistilRoBERTa
        
        # if not text:
        #     return 0.0, "neutral"
        # 
        # cleaned_text = self.clean_text(text)
        # inputs = self.tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)
        # 
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        #     predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        #     score = predictions[0][1].item() - predictions[0][0].item()  # positive - negative
        #     
        #     if score > 0.3:
        #         label = "positive"
        #     elif score < -0.3:
        #         label = "negative"
        #     else:
        #         label = "neutral"
        # 
        # return score, label
        
        # Placeholder: use TextBlob for now
        return self.analyze_sentiment_textblob(text)
    
    def analyze_comments(self, comments_list):
        """
        Analyze a list of comments and return aggregate sentiment
        """
        if not comments_list or len(comments_list) == 0:
            return {
                'avg_sentiment': 0.0,
                'sentiment_label': 'neutral',
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0
            }
        
        sentiments = []
        labels = []
        
        for comment in comments_list:
            if self.use_distilroberta:
                score, label = self.analyze_sentiment_distilroberta(comment)
            else:
                score, label = self.analyze_sentiment_textblob(comment)
            
            sentiments.append(score)
            labels.append(label)
        
        # Calculate statistics
        avg_sentiment = np.mean(sentiments)
        positive_count = labels.count('positive')
        negative_count = labels.count('negative')
        neutral_count = labels.count('neutral')
        total = len(labels)
        
        # Determine overall label
        if avg_sentiment > 0.1:
            overall_label = 'positive'
        elif avg_sentiment < -0.1:
            overall_label = 'negative'
        else:
            overall_label = 'neutral'
        
        return {
            'avg_sentiment': round(avg_sentiment, 3),
            'sentiment_label': overall_label,
            'positive_ratio': round(positive_count / total, 3),
            'negative_ratio': round(negative_count / total, 3),
            'neutral_ratio': round(neutral_count / total, 3),
            'total_comments': total
        }
    
    def get_sentiment_score(self, text):
        """
        Get sentiment score for a single text
        Returns normalized score between 0 and 1 (higher is more positive)
        """
        if self.use_distilroberta:
            score, _ = self.analyze_sentiment_distilroberta(text)
        else:
            score, _ = self.analyze_sentiment_textblob(text)
        
        # Normalize to 0-1 range
        normalized = (score + 1) / 2
        return normalized