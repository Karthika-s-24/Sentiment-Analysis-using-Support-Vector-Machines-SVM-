import streamlit as st
import pandas as pd
import time
from data_training import SentimentModel
from enhanced_features import EnhancedFeatures

# Page configuration
st.set_page_config(
    page_title="Smart Reply Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .sentiment-neutral {
        background-color: #e2e3e5;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #6c757d;
        margin: 10px 0;
    }
    .suggestion-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #b3d9ff;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'enhanced' not in st.session_state:
    st.session_state.enhanced = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

@st.cache_resource
def initialize_models():
    """Initialize and cache the models"""
    with st.spinner('🔄 Loading AI models... This may take a few seconds.'):
        model = SentimentModel()
        model.train_model()
        enhanced = EnhancedFeatures()
        return model, enhanced

def get_sentiment_color(sentiment):
    """Return color based on sentiment"""
    colors = {
        'positive': 'green',
        'negative': 'red',
        'neutral': 'gray'
    }
    return colors.get(sentiment, 'gray')

def get_sentiment_emoji(sentiment):
    """Return emoji based on sentiment"""
    emojis = {
        'positive': '😊',
        'negative': '😔',
        'neutral': '😐'
    }
    return emojis.get(sentiment, '🤔')

def get_context_emoji(context):
    """Return emoji based on context"""
    emojis = {
        'work': '💼',
        'personal': '👨‍👩‍👧‍👦',
        'health': '🏥',
        'food': '🍕',
        'weather': '🌤️',
        'entertainment': '🎬',
        'general': '📝'
    }
    return emojis.get(context, '📝')

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Smart Reply Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze messages and get smart reply suggestions using AI!")
    
    # Initialize models
    if not st.session_state.initialized:
        with st.spinner("🔄 Initializing AI models... Please wait."):
            try:
                st.session_state.model, st.session_state.enhanced = initialize_models()
                st.session_state.initialized = True
                st.success("✅ Models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing models: {e}")
                return
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This AI-powered assistant:
        - 🔍 Analyzes text sentiment
        - 🎯 Detects context and emotions
        - 💡 Suggests smart replies
        - 📊 Provides detailed analysis
        """)
        
        st.header("Quick Examples")
        example_texts = [
            "I love my new job! The work is exciting!",
            "This product is terrible and doesn't work properly.",
            "The meeting is scheduled for 3 PM tomorrow.",
            "I'm feeling sick today with a headache.",
            "I am sad and feeling down today.",
            "The weather is beautiful today!"
        ]
        
        for example in example_texts:
            if st.button(example, key=example):
                st.session_state.input_text = example
                st.rerun()
        
        st.markdown("---")
        st.header("Analysis History")
        if st.session_state.analysis_history:
            history_df = pd.DataFrame(st.session_state.analysis_history)
            st.dataframe(history_df[['sentiment', 'confidence', 'context']].tail(5), use_container_width=True)
            if st.button("Clear History"):
                st.session_state.analysis_history = []
                st.rerun()
        else:
            st.write("No analysis yet")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Enter Your Message")
        
        # Text input
        input_text = st.text_area(
            "Type your message, paragraph, or any text below:",
            height=150,
            placeholder="Example: I'm so excited about my promotion at work! The new role comes with great responsibilities...",
            key="input_text",
            label_visibility="collapsed"
        )
        
        # Analyze button
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Clear Text", use_container_width=True):
                st.session_state.input_text = ""
                st.rerun()
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        if st.session_state.analysis_history:
            latest = st.session_state.analysis_history[-1]
            st.metric("Total Analyses", len(st.session_state.analysis_history))
            st.metric("Latest Sentiment", f"{latest['sentiment'].title()} {get_sentiment_emoji(latest['sentiment'])}")
            st.metric("Confidence", f"{latest['confidence']:.3f}")
        else:
            st.metric("Total Analyses", 0)
            st.metric("Latest Sentiment", "N/A")
            st.metric("Confidence", "N/A")
        
        st.markdown("---")
        st.caption("💡 **Tip**: Try the example messages in the sidebar!")
    
    # Analysis results
    if analyze_btn and input_text.strip():
        with st.spinner('🔄 Analyzing your message...'):
            # Perform analysis
            sentiment, confidence = st.session_state.model.predict_sentiment(input_text)
            analysis = st.session_state.enhanced.get_detailed_analysis(input_text, sentiment, confidence)
            
            # Store in history
            st.session_state.analysis_history.append(analysis)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            # Sentiment with colored box
            sentiment_class = f"sentiment-{sentiment}"
            sentiment_emoji = get_sentiment_emoji(sentiment)
            st.markdown(
                f'<div class="{sentiment_class}">'
                f'<h3>{sentiment_emoji} Sentiment: {sentiment.upper()} (Confidence: {confidence:.3f})</h3>'
                f'</div>', 
                unsafe_allow_html=True
            )
            
            # Key metrics
            col3, col4, col5, col6 = st.columns(4)
            
            with col3:
                context_emoji = get_context_emoji(analysis['context'])
                st.metric("Context", f"{context_emoji} {analysis['context'].title()}")
            
            with col4:
                emotion_emoji = "😐" if analysis['primary_emotion'] == 'neutral' else "🎭"
                st.metric("Primary Emotion", f"{emotion_emoji} {analysis['primary_emotion'].title()}")
            
            with col5:
                urgency_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                st.metric("Urgency", f"{urgency_color[analysis['urgency']]} {analysis['urgency'].title()}")
            
            with col6:
                st.metric("Word Count", analysis['word_count'])
            
            # Summary
            st.subheader("📝 Summary")
            st.info(analysis['summary'])
            
            # Detailed analysis
            col7, col8 = st.columns(2)
            
            with col7:
                st.subheader("📈 Text Statistics")
                st.metric("Character Count", analysis['character_count'])
                
                if analysis['emotion_scores']:
                    st.write("**Emotion Breakdown:**")
                    for emotion, score in analysis['emotion_scores'].items():
                        st.progress(min(score/10, 1.0), text=f"{emotion.title()}: {score}")
            
            with col8:
                st.subheader("🎯 Context Details")
                st.write(f"**Detected Context:** {analysis['context'].title()}")
                st.write(f"**Primary Emotion:** {analysis['primary_emotion'].title()}")
                st.write(f"**Urgency Level:** {analysis['urgency'].title()}")
                
                # Emotion scores visualization
                if analysis['emotion_scores']:
                    emotion_df = pd.DataFrame({
                        'Emotion': list(analysis['emotion_scores'].keys()),
                        'Score': list(analysis['emotion_scores'].values())
                    })
                    st.bar_chart(emotion_df.set_index('Emotion'))
            
            # Suggested Replies
            st.subheader("💡 Suggested Replies")
            st.info("Choose one of these context-aware replies for your conversation:")
            
            selected_reply = None
            for i, reply in enumerate(analysis['suggestions'], 1):
                with st.container():
                    col9, col10 = st.columns([4, 1])
                    with col9:
                        st.markdown(f'<div class="suggestion-box"><b>Option {i}:</b> {reply}</div>', unsafe_allow_html=True)
                    with col10:
                        if st.button(f"Use", key=f"use_{i}"):
                            selected_reply = reply
            
            if selected_reply:
                st.success("✅ Reply selected! You can copy it below:")
                st.code(selected_reply, language=None)
    
    elif analyze_btn and not input_text.strip():
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with ❤️ using Streamlit, Scikit-learn, and NLTK | "
        "Smart Reply Assistant v2.0"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()