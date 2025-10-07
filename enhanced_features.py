import re
import nltk
from nltk.tokenize import sent_tokenize

class EnhancedFeatures:
    def __init__(self):
        # Response templates based on sentiment
        self.response_templates = {
            'positive': [
                "That's great to hear! I'm happy for you.",
                "Wonderful! Thanks for sharing this positive news.",
                "Awesome! Keep up the positive energy.",
                "That sounds amazing! I'm glad things are going well.",
                "Fantastic! It's great to see you so enthusiastic."
            ],
            'negative': [
                "I'm sorry you're feeling this way. Would you like to talk about it?",
                "That sounds tough. Remember, I'm here to support you.",
                "I understand this is difficult. Things will get better.",
                "Thank you for sharing. It's okay to feel this way sometimes.",
                "I hear you. That sounds really challenging."
            ],
            'neutral': [
                "Thanks for sharing this information.",
                "I see. Is there anything specific you'd like to discuss?",
                "Understood. Let me know if you need any help.",
                "Thanks for the update. How can I assist you further?",
                "Noted. I appreciate you keeping me informed."
            ]
        }
        
        # Context detection keywords
        self.context_keywords = {
            'work': ['job', 'work', 'office', 'career', 'boss', 'colleague', 'meeting', 'promotion', 'salary', 'project'],
            'personal': ['family', 'friend', 'relationship', 'love', 'partner', 'parents', 'kids', 'marriage', 'dating'],
            'product': ['product', 'service', 'customer', 'buy', 'purchase', 'order', 'delivery', 'quality', 'price'],
            'health': ['feel', 'sick', 'health', 'doctor', 'hospital', 'pain', 'tired', 'energy', 'medical', 'illness'],
            'weather': ['weather', 'rain', 'sunny', 'cold', 'hot', 'temperature', 'forecast', 'climate', 'storm'],
            'food': ['food', 'restaurant', 'meal', 'delicious', 'tasty', 'cooking', 'recipe', 'dinner', 'lunch'],
            'entertainment': ['movie', 'music', 'game', 'show', 'concert', 'film', 'entertainment', 'fun']
        }
        
        # Emotion detection keywords
        self.emotion_keywords = {
            'anger': ['angry', 'mad', 'furious', 'hate', 'rage', 'annoyed', 'frustrated', 'upset', 'irritated'],
            'joy': ['happy', 'joy', 'excited', 'delighted', 'love', 'thrilled', 'ecstatic', 'pleased', 'glad'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'cry', 'heartbroken', 'gloomy', 'down', 'hopeless'],
            'fear': ['scared', 'afraid', 'fear', 'terrified', 'worried', 'anxious', 'nervous', 'panic', 'concerned'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow', 'astonishing']
        }
    
    def generate_summary(self, text):
        """Generate a simple summary of the text using NLTK"""
        if not isinstance(text, str) or not text.strip():
            return "No text provided for summary."
        
        try:
            # Use NLTK for sentence tokenization
            sentences = sent_tokenize(text)
            
            if not sentences:
                return text[:100] + "..." if len(text) > 100 else text
            
            # Simple extractive summarization - take first 2 meaningful sentences
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            if not meaningful_sentences:
                # If no sentences are long enough, take the first sentence
                summary = sentences[0]
            else:
                # Take up to 2 sentences for summary
                summary_sentences = meaningful_sentences[:2]
                summary = ' '.join(summary_sentences)
            
            # Ensure summary ends with proper punctuation
            if not summary.endswith(('.', '!', '?')):
                summary += '.'
                
            return summary
            
        except Exception as e:
            # Fallback: simple truncation
            if len(text) > 150:
                return text[:147] + "..."
            return text
    
    def detect_context(self, text):
        """Detect the context/topic of the text"""
        if not isinstance(text, str):
            return 'general'
        
        text_lower = text.lower()
        context_scores = {}
        
        for context, keywords in self.context_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                context_scores[context] = score
        
        if context_scores:
            return max(context_scores, key=context_scores.get)
        return 'general'
    
    def detect_emotion(self, text):
        """Detect specific emotions in the text"""
        if not isinstance(text, str):
            return 'neutral', {}
        
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            return primary_emotion, emotion_scores
        else:
            return 'neutral', {}
    
    def analyze_urgency(self, text):
        """Analyze if the text indicates urgency"""
        if not isinstance(text, str):
            return "low"
        
        urgent_keywords = ['emergency', 'urgent', 'asap', 'immediately', 'right now', 'help needed', 'critical', 'important']
        text_lower = text.lower()
        
        urgent_count = sum(1 for keyword in urgent_keywords if keyword in text_lower)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        urgency_score = urgent_count + (exclamation_count * 0.3) + (question_count * 0.2)
        
        if urgency_score >= 2:
            return "high"
        elif urgency_score >= 1:
            return "medium"
        else:
            return "low"
    
    def generate_context_aware_replies(self, sentiment, text, primary_emotion=None):
        """Generate context-aware reply suggestions"""
        base_replies = self.response_templates.get(sentiment, ["Thanks for sharing."])
        context = self.detect_context(text)
        urgency = self.analyze_urgency(text)
        
        customized_replies = []
        
        for reply in base_replies:
            customized_reply = reply
            
            # Add context-specific phrases
            context_phrases = {
                'work': {
                    'positive': " That's great for your career!",
                    'negative': " Work situations can be challenging.",
                    'neutral': " Work life has its routines."
                },
                'personal': {
                    'positive': " That's wonderful for your personal life!",
                    'negative': " Personal matters can be tough to navigate.",
                    'neutral': " Personal life has its ups and downs."
                },
                'health': {
                    'positive': " Great to hear about your health!",
                    'negative': " I hope you feel better soon.",
                    'neutral': " Health is important to monitor."
                },
                'food': {
                    'positive': " Sounds delicious!",
                    'negative': " Sorry to hear about the food experience.",
                    'neutral': " Food experiences vary greatly."
                },
                'weather': {
                    'positive': " Nice weather always helps!",
                    'negative': " Weather can affect our mood.",
                    'neutral': " Weather is what it is."
                }
            }
            
            if context in context_phrases and sentiment in context_phrases[context]:
                customized_reply += context_phrases[context][sentiment]
            
            # Add emotion-specific phrases
            emotion_phrases = {
                'anger': " It's understandable to feel frustrated.",
                'sadness': " Remember that feelings are temporary.",
                'fear': " It's okay to feel concerned sometimes.",
                'surprise': " Unexpected events can be quite impactful."
            }
            
            if primary_emotion in emotion_phrases:
                customized_reply += emotion_phrases[primary_emotion]
            
            # Add urgency handling
            if urgency == "high" and sentiment == 'negative':
                customized_reply += " This sounds urgent - please seek help if needed."
            elif urgency == "medium":
                customized_reply += " Let me know if you need immediate assistance."
            
            customized_replies.append(customized_reply)
        
        return customized_replies[:4]  # Return max 4 suggestions
    
    def get_detailed_analysis(self, text, sentiment, confidence):
        """Generate comprehensive analysis of the input text"""
        if not isinstance(text, str):
            text = ""
        
        # Override sentiment for clear emotion words if confidence is low
        text_lower = text.lower()
        if confidence < 0.6:
            if any(word in text_lower for word in ['sad', 'unhappy', 'depressed', 'miserable', 'hate', 'angry']):
                sentiment = 'negative'
                confidence = 0.9
            elif any(word in text_lower for word in ['happy', 'joyful', 'excited', 'love', 'great', 'wonderful']):
                sentiment = 'positive'
                confidence = 0.9
        
        context = self.detect_context(text)
        primary_emotion, emotion_scores = self.detect_emotion(text)
        urgency = self.analyze_urgency(text)
        summary = self.generate_summary(text)
        reply_suggestions = self.generate_context_aware_replies(sentiment, text, primary_emotion)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'context': context,
            'primary_emotion': primary_emotion,
            'emotion_scores': emotion_scores,
            'urgency': urgency,
            'summary': summary,
            'suggestions': reply_suggestions,
            'word_count': len(text.split()),
            'character_count': len(text)
        }