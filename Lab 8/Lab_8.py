"""
TFB3133 Data Visualization - Lab 8: Qualitative Data and Sentiment Analysis
Run:  streamlit run lab8_sentiment.py
Edited version - covers Lesson 3 Assignment tasks 1-4 (marked "TASK x.y" below)
"""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# ======================================================================
# Lesson 1 - sample data + sentiment labels (unchanged from lab sheet)
# ======================================================================
reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "Good value, but cleaning the narrow opening is difficult.",
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality.",
]
df = pd.DataFrame({"id": [f"R{i:02}" for i in range(1, 19)], "text": reviews})
analyzer = SentimentIntensityAnalyzer()


def label_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    if score <= -0.05:
        return "Negative", score
    return "Neutral", score


df[["sentiment", "compound"]] = df["text"].apply(lambda text: pd.Series(label_sentiment(text)))
df["compound"] = df["compound"].round(3)

# TASK 4.1 - main topic for each comment (assigned by reading each comment)
TOPICS = {
    "R01": "Insulation", "R02": "Portability", "R03": "Design", "R04": "Lid & leaks",
    "R05": "Lid & leaks", "R06": "Cleaning", "R07": "Insulation", "R08": "General",
    "R09": "Insulation", "R10": "Portability", "R11": "Portability", "R12": "General",
    "R13": "Durability", "R14": "General", "R15": "Insulation", "R16": "Price",
    "R17": "Design", "R18": "Durability",
}
df["topic"] = df["id"].map(TOPICS)

# TASK 1.3 - comments that praise AND criticise (contrast words "but", "though", ";")
df["mixed_tone"] = df["text"].str.contains(r"\bbut\b|\bthough\b|;", case=False)

ORDER = ["Positive", "Neutral", "Negative"]
COLORS = {"Positive": "#2a9d8f", "Neutral": "#9aa5b1", "Negative": "#e76f51"}

# ======================================================================
# Lesson 2 - dashboard
# ======================================================================
st.set_page_config(page_title="Lab 8 | Feedback", layout="wide")
st.title("Product Feedback: A Text Summary")
st.caption("Practice data only (18 synthetic water-bottle comments). "
           "Automated labels need human review.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Comments", len(df))
for col, s in zip([k2, k3, k4], ORDER):
    n = (df["sentiment"] == s).sum()
    col.metric(s, f"{n}  ({n / len(df):.0%})")

left, right = st.columns(2)
with left:
    st.subheader("Comment counts by sentiment")
    counts = df["sentiment"].value_counts().reindex(ORDER, fill_value=0)
    st.bar_chart(counts, horizontal=True, x_label="Number of comments", y_label="VADER label")
    st.caption("Labels use VADER compound score: >= 0.05 Positive, <= -0.05 Negative, "
               "otherwise Neutral. Counts hide mixed opinions inside one comment.")

with right:
    st.subheader("Words used in the comments")
    # TASK 2.1 - remove an uninformative frequent term: add "day" to stop words
    # TASK 2.2 - retain a useful term: take "not" OUT of the stop-word list
    stop_words = (set(STOPWORDS) | {"bottle", "drinks", "drink", "day"}) - {"not"}
    cloud = WordCloud(width=900, height=500, background_color="white",
                      colormap="viridis", stopwords=stop_words,
                      collocations=False, random_state=1).generate(" ".join(df["text"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    # TASK 2.2 - caption explaining word size and its limits
    st.caption("Word size = how often a word appears after filler words are removed "
               "(\"cold\" appears 3 times). Position and colour mean nothing. Size does NOT "
               "show sentiment or context: \"cold\" comes from both praise (R01) and complaints "
               "(R15). \"day\" was removed (it only appears in \"all day\" / \"every day\"); "
               "\"not\" was kept so negations stay visible.")

st.subheader("Review the original comments")
st.dataframe(df[["id", "text", "topic", "sentiment", "compound", "mixed_tone"]],
             width="stretch", hide_index=True)
st.caption("A compound score is a model output, not a confidence rating.")

# ======================================================================
# Lesson 3 - assignment
# ======================================================================
st.header("Assignment")

# ---------------------------------------------------------------- Task 1
st.subheader("1. Run and describe")
st.markdown("""
- The app shows **18 comments**: 9 Positive (50%), 3 Neutral (17%), 6 Negative (33%).
- **Largest group:** Positive (9 of 18).
- **Recurring theme:** keeping drinks cold. "cold" is the largest word (3 mentions: R01, R07, R15),
  and insulation is the most common topic (4 comments).
- **Mixed-tone example - R07** *"Keeps drinks cold, though the paint scratches easily."*
  is labelled Positive (0.340). The single label hides the complaint about the paint,
  so a product team would miss a durability issue. 4 comments have a mixed tone
  (R06, R07, R11, R15 contain "but" or "though"; R17 uses ";").
""")

# ---------------------------------------------------------------- Task 3
st.subheader("3. Audit of sentiment labels (manual review)")
audit = pd.DataFrame([
    ["R02", "Positive", "Positive", "Yes", "Clear praise for size and ease of carrying."],
    ["R09", "Positive", "Negative", "No - incorrect", "\"expected better\" is a complaint; VADER only sees the word \"better\"."],
    ["R14", "Neutral", "Neutral / mildly positive", "Yes", "Factual statement; no strong opinion."],
    ["R01", "Neutral", "Positive", "No - borderline", "Praises the main function, but has no sentiment words (\"cold\" is not in the lexicon)."],
    ["R13", "Negative", "Negative", "Yes", "Broken lid and poor quality - clear complaint."],
    ["R08", "Negative", "Neutral", "No - borderline", "\"fine ... nothing special\" is neutral; \"nothing\" pulls the score below -0.05."],
], columns=["id", "VADER label", "My label", "Reasonable?", "Note"])
audit = audit.merge(df[["id", "text", "compound"]], on="id")
st.dataframe(audit[["id", "text", "compound", "VADER label", "My label", "Reasonable?", "Note"]],
             width="stretch", hide_index=True)
st.markdown("""
- **3 of 6 labels look reasonable.** Other incorrect results outside the sample:
  R15 *"does not keep drinks cold"* is Positive (0.226) because only "Nice" is scored,
  and R16 *"Affordable and practical"* is Neutral (0.000) even though it is praise.
""")

st.markdown("**Negation test**")
test_text = st.text_input("Type a comment to score", "The bottle is not easy to clean.")
t_label, t_score = label_sentiment(test_text)
base_label, base_score = label_sentiment("The bottle is easy to clean.")
c1, c2 = st.columns(2)
c1.metric("Original: \"The bottle is easy to clean.\"", base_label, f"{base_score:+.3f}")
c2.metric(f"Test: \"{test_text}\"", t_label, f"{t_score:+.3f}")
st.caption("VADER flips the score when \"not\" is directly before a sentiment word (easy). "
           "It fails in R15 because \"cold\" is not a sentiment word, so \"not ... cold\" is ignored.")

# ---------------------------------------------------------------- Task 4
st.subheader("4. Sentiment by topic")
by_topic = (pd.crosstab(df["topic"], df["sentiment"]).reindex(columns=ORDER, fill_value=0))
by_topic["Total"] = by_topic.sum(axis=1)
by_topic["Mean compound"] = df.groupby("topic")["compound"].mean().round(3)
by_topic = by_topic.sort_values("Total", ascending=False)

fig2, ax2 = plt.subplots(figsize=(8, 3.8))
by_topic[ORDER].iloc[::-1].plot.barh(stacked=True, ax=ax2, color=[COLORS[s] for s in ORDER],
                                     edgecolor="white")
ax2.set_xlabel("Number of comments")
ax2.set_ylabel("")
ax2.set_title("VADER sentiment by main topic (n = 18)")
ax2.legend(title="VADER label", loc="lower right", frameon=False)
ax2.spines[["top", "right"]].set_visible(False)
st.pyplot(fig2, clear_figure=True)
st.dataframe(by_topic, width="stretch")
st.markdown("""
- **Too small for a fair comparison:** each topic has only 1-4 comments, so one comment
  changes a topic's result by 25-100%.
- **Insulation** (4 comments) looks 75% positive by VADER, but on reading them 2 are praise
  (R01, R07) and 2 are complaints (R09, R15) - both complaints were mislabelled Positive.
- **Lid & leaks** (R04, R05) and **Cleaning** (R06) are negative only; **Portability** (R02, R10, R11) is positive only.
""")

# ---------------------------------------------------------------- Summary
st.subheader("Summary for the product team")
st.markdown("""
We analysed 18 customer comments on the water bottle using an automated sentiment model
(VADER) and reviewed the results by hand.

**Finding:** Half of the comments (9 of 18) were labelled positive, mostly about portability
and design. Keeping drinks cold is the most discussed topic, but opinion on it is split:
two customers praise the insulation and two complain about it. All negative comments
about the lid, leaks or cleaning point to the lid and opening as the main weakness.

**Limitation:** The sample is small and synthetic, with only one to four comments per topic.
The model also misreads several comments. "I expected better insulation" and "does not keep
drinks cold" were both labelled positive, so the automated counts overstate satisfaction
with insulation.

**Recommendation:** Collect at least 30 comments per topic from real buyers, and pair each
comment with a 1-5 star rating so text labels can be checked against scores. Ask one
question per topic, such as lid, insulation and cleaning, so that mixed opinions are not
merged into one label.
""")

# ---------------------------------------------------------------- Reflection
with st.expander("Reflection"):
    st.markdown("""
- **What the visuals revealed:** mostly positive feedback, with insulation ("cold") as the
  main theme and the lid/opening as the main complaint.
- **What could mislead:** the bar chart presents model labels as facts, and a large word
  does not mean a positive or negative opinion. The word cloud cannot show "not cold".
- **Validation:** have 2 people label the comments independently, compare with VADER,
  and report agreement; check labels against star ratings; collect a larger, real sample.
""")