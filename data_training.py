import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import re

# Download required NLTK data
def download_nltk_data():
    required_data = {
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'punkt': 'tokenizers/punkt'
    }
    
    for data_name, data_path in required_data.items():
        try:
            nltk.data.find(data_path)
            print(f"✅ {data_name} already downloaded")
        except LookupError:
            print(f"📥 Downloading {data_name}...")
            try:
                if data_name == 'punkt':
                    nltk.download('punkt')
                elif data_name == 'stopwords':
                    nltk.download('stopwords')
                elif data_name == 'wordnet':
                    nltk.download('wordnet')
                print(f"✅ {data_name} downloaded successfully")
            except Exception as e:
                print(f"❌ Error downloading {data_name}: {e}")

# Download required data
download_nltk_data()

class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
        self.svm_model = SVC(kernel='linear', probability=True, random_state=42)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False
        
        # Rule-based sentiment words
        self.negative_words = ['sad', 'unhappy', 'depressed', 'miserable', 'terrible', 'awful', 'bad', 'horrible', 'hate', 'angry', 'upset']
        self.positive_words = ['happy', 'joyful', 'excited', 'great', 'wonderful', 'amazing', 'good', 'excellent', 'love', 'fantastic', 'perfect']
    
    def rule_based_sentiment(self, text):
        """Rule-based sentiment as fallback for clear cases"""
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        if negative_count > positive_count:
            return 'negative', 0.9
        elif positive_count > negative_count:
            return 'positive', 0.9
        else:
            return 'neutral', 0.7
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits but keep basic punctuation for sentiment
        text = re.sub(r'[^a-zA-Z\s!?]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Tokenize and lemmatize
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words and len(word) > 2]
        return ' '.join(words)
    
    def prepare_training_data(self):
        """Create comprehensive training data for sentiment analysis - FIXED COUNT"""
        texts = [
            # Enhanced Positive sentiments (18 examples)
            "I love this product! It's amazing and works perfectly.",
            "I'm so happy with my new job! Everything is going great.",
            "The food was delicious and the service was excellent!",
            "This is fantastic news! I'm thrilled to hear it.",
            "I'm excited about the upcoming vacation! Can't wait!",
            "I am very satisfied with the purchase. Good quality product.",
            "I'm delighted with the outcome! Perfect results!",
            "The event was well organized and informative.",
            "Outstanding performance! Exceeded all expectations!",
            "Wonderful experience! Will definitely come back again.",
            "This is absolutely brilliant! I'm so impressed.",
            "Great service and amazing quality! Highly recommended.",
            "I love how easy this is to use. Perfect design!",
            "Excellent customer support! Very helpful and friendly.",
            "Beautiful day today! Everything feels perfect.",
            "I'm so happy and joyful today!",
            "This makes me feel wonderful and excited!",
            "I love this so much! It's fantastic!",
            
            # Enhanced Negative sentiments (19 examples)
            "This is the worst experience I've ever had. Terrible service!",
            "This movie was boring and disappointing. Waste of time.",
            "I hate waiting in long lines. Very frustrating experience.",
            "The customer support was rude and unhelpful.",
            "The product broke after one day. Very poor quality.",
            "The service was slow and the staff was unfriendly.",
            "This is horrible, I want my money back!",
            "Poor customer experience, will not recommend.",
            "Very disappointed with the quality and service.",
            "Awful product! Doesn't work as advertised at all.",
            "I'm really upset about this situation. Very unfair!",
            "Terrible quality and bad customer service.",
            "This made me so angry! Complete waste of money.",
            "I regret buying this product. Total disappointment.",
            "The worst decision I've ever made. So frustrating.",
            "I am sad and feeling down today.",
            "This makes me so unhappy and depressed.",
            "I feel miserable and sad about this situation.",
            "This sad news has ruined my day.",
            
            # Neutral sentiments (18 examples)
            "The weather is nice today. Nothing special.",
            "The package arrived on time. Standard delivery service.",
            "It's an average day. Nothing particularly good or bad.",
            "The meeting went as planned. Regular updates were shared.",
            "Standard procedure, followed as expected.",
            "The product works as described. No issues found.",
            "Regular day at work. Everything is normal.",
            "The service was adequate. Met basic expectations.",
            "The movie was okay. Not great but not bad either.",
            "Average performance. Could be better, could be worse.",
            "The food was acceptable. Nothing extraordinary.",
            "Regular maintenance completed. System is functional.",
            "Standard features included. Works as intended.",
            "Normal day with routine activities.",
            "The product meets basic requirements.",
            "The report was submitted on schedule.",
            "Regular system check completed successfully.",
            "Standard operating procedures were followed."
        ]
        
        labels = [
            # Positive labels (18)
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive',
            
            # Negative labels (19)
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative',
            
            # Neutral labels (18)
            'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
            'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
            'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
            'neutral', 'neutral', 'neutral'
        ]
        
        # Verify counts match
        print(f"📊 Data Verification: {len(texts)} texts, {len(labels)} labels")
        if len(texts) != len(labels):
            raise ValueError(f"Data mismatch: {len(texts)} texts vs {len(labels)} labels")
        
        return texts, labels
    
    def train_model(self):
        """Train the SVM sentiment analysis model"""
        print("🔄 Training SVM sentiment model...")
        try:
            texts, labels = self.prepare_training_data()
            
            # Verify data integrity
            if len(texts) != len(labels):
                raise ValueError(f"Data mismatch detected: {len(texts)} texts vs {len(labels)} labels")
            
            print(f"📈 Training on {len(texts)} samples...")
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # Create TF-IDF features
            X = self.vectorizer.fit_transform(processed_texts)
            y = labels
            
            # Verify feature matrix shape
            print(f"📊 Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Train SVM model
            self.svm_model.fit(X, y)
            
            # Calculate accuracy
            y_pred = self.svm_model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            
            print(f"✅ Model training completed!")
            print(f"📊 Training Accuracy: {accuracy:.3f}")
            print(f"📈 Total training samples: {len(texts)}")
            print(f"🎯 Classes: {set(labels)}")
            
            self.is_trained = True
            return self
            
        except Exception as e:
            print(f"❌ Error during model training: {e}")
            raise
    
    def svm_predict(self, text):
        """SVM-only prediction"""
        processed_text = self.preprocess_text(text)
        features = self.vectorizer.transform([processed_text])
        prediction = self.svm_model.predict(features)[0]
        probability = np.max(self.svm_model.predict_proba(features))
        
        return prediction, round(probability, 3)
    
    def predict_sentiment(self, text):
        """Predict sentiment with rule-based fallback"""
        if not self.is_trained:
            raise ValueError("Model is not trained. Please call train_model() first.")
        
        if not isinstance(text, str) or not text.strip():
            return 'neutral', 0.33
        
        # First, try rule-based for clear cases
        rule_sentiment, rule_confidence = self.rule_based_sentiment(text)
        
        # If rule-based has high confidence, use it
        if rule_confidence >= 0.9:
            return rule_sentiment, rule_confidence
        
        # Otherwise, use SVM
        try:
            svm_sentiment, svm_confidence = self.svm_predict(text)
            
            # If SVM confidence is low, prefer rule-based
            if svm_confidence < 0.6:
                return rule_sentiment, max(rule_confidence, svm_confidence)
            
            return svm_sentiment, svm_confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return rule_sentiment, rule_confidence

# Test the model directly
if __name__ == "__main__":
    print("🧪 Testing Sentiment Model...")
    model = SentimentModel()
    model.train_model()
    
    # Test the model
    test_texts = [
        "I love this amazing product!",
        "This is terrible and awful.",
        "It's okay, nothing special.",
        "I'm so happy today!",
        "This makes me really angry!",
        "I am sad and unhappy"
    ]
    
    print("\n📋 Model Testing Results:")
    print("-" * 50)
    for text in test_texts:
        sentiment, confidence = model.predict_sentiment(text)
        print(f"Text: '{text}'")
        print(f"→ Sentiment: {sentiment.upper()} (Confidence: {confidence:.3f})")
        print()