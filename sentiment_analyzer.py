from textblob import TextBlob


class SentimentAnalyzer:
    def __init__(self, use_distilroberta=False):
        """
        Lightweight sentiment analyzer.
        Keeping transformer-based models disabled for Render stability.
        """
        self.use_distilroberta = use_distilroberta

    def analyze_comments(self, comments):
        """
        Analyze list of comments
        """
        if not comments:
            return {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            }

        results = {"positive": 0, "neutral": 0, "negative": 0}

        for comment in comments:
            polarity = TextBlob(str(comment)).sentiment.polarity
            if polarity > 0.1:
                results["positive"] += 1
            elif polarity < -0.1:
                results["negative"] += 1
            else:
                results["neutral"] += 1

        return results

    def get_sentiment_score(self, text):
        """
        Return normalized sentiment score [0–1]
        """
        if not text:
            return 0.5

        polarity = TextBlob(str(text)).sentiment.polarity
        return round((polarity + 1) / 2, 3)


# ====================================================
# ✅ REQUIRED BY app.py
# ====================================================
from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    label = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"

    return {
        "label": label,
        "polarity": polarity
    }
