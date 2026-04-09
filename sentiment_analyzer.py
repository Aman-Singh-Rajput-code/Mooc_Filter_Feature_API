from textblob import TextBlob


class SentimentAnalyzer:
    def __init__(self, use_distilroberta=False):
        self.use_distilroberta = use_distilroberta

    def analyze_comments(self, comments):
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
        Improved sentiment scoring for technical content
        """
        if not text or text.strip() == "":
            return 0.5

        text = str(text).lower()

        # 🔥 Step 1: TextBlob polarity
        try:
            polarity = TextBlob(text).sentiment.polarity
            score = (polarity + 1) / 2  # normalize 0–1
        except:
            score = 0.5

        # 🔥 Step 2: Keyword-based boost (VERY IMPORTANT)
        positive_keywords = [
            "best", "good", "excellent", "great", "amazing",
            "learn", "master", "complete", "professional"
        ]

        tech_positive_keywords = [
            "python", "machine learning", "web development",
            "react", "ai", "data science", "deep learning"
        ]

        boost = 0

        # sentiment keywords
        for word in positive_keywords:
            if word in text:
                boost += 0.1

        # tech relevance boost
        for word in tech_positive_keywords:
            if word in text:
                boost += 0.05

        final_score = min(score + boost, 1.0)

        return round(final_score, 3)


# ====================================================
# API HELPER (USED IN app.py)
# ====================================================
def analyze_sentiment(text: str) -> dict:
    if not text:
        return {
            "polarity": 0.0,
            "label": "neutral"
        }

    text = str(text)

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.1:
        label = "positive"
    elif polarity < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return {
        "polarity": round(polarity, 3),
        "label": label
    }
