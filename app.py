import openai
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Correctly access the OpenAI API key from the secrets file
openai.api_key = st.secrets["general"]["OPENAI_API_KEY"]

# --- Function to extract themes ---
def extract_themes(reviews, label):
    joined_reviews = "\n".join(reviews)
    prompt = f"""
    Here are some {label} customer reviews:\n{joined_reviews}\n
    Please extract the top 5 common themes or topics mentioned in these reviews. 
    Return them as a bullet list with brief descriptions.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=300
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# --- Streamlit UI ---
st.title("🧠 Google Reviews Analyzer (AI-powered)")

uploaded_file = st.file_uploader("Upload your reviews CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if "review" in df.columns and "class" in df.columns:
        st.success("CSV loaded successfully!")

        # Show sentiment distribution
        counts = df["class"].value_counts()
        st.subheader("📊 Sentiment Overview")
        fig, ax = plt.subplots()
        ax.bar(["Positive", "Negative"], [counts.get(1, 0), counts.get(0, 0)], color=["green", "red"])
        st.pyplot(fig)

        # Split positive and negative reviews
        positive_reviews = df[df["class"] == 1]["review"].tolist()
        negative_reviews = df[df["class"] == 0]["review"].tolist()

        # Limit for fast results
        short_pos = positive_reviews[:30]
        short_neg = negative_reviews[:30]

        with st.spinner("Analyzing positive reviews..."):
            pos_themes = extract_themes(short_pos, "positive")
        with st.spinner("Analyzing negative reviews..."):
            neg_themes = extract_themes(short_neg, "negative")

        st.subheader("✅ Top Themes in Positive Reviews")
        st.markdown(pos_themes)

        st.subheader("⚠️ Top Themes in Negative Reviews")
        st.markdown(neg_themes)
    
    else:
        st.error("CSV must have 'review' and 'class' columns.")
