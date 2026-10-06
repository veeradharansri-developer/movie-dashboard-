import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Movie Intelligence Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background: #0a0d12;
            color: #f4f4f4;
        }
        [data-testid="stSidebar"] {
            background: #0f131a;
            border-right: 1px solid #252b34;
        }
        .hero {
            padding: 26px 28px;
            border-radius: 18px;
            background: linear-gradient(135deg, #111722 0%, #0b0f15 100%);
            border: 1px solid #292f39;
            margin-bottom: 18px;
        }
        .hero h1 {
            color: #f2c66d;
            font-size: 40px;
            margin-bottom: 4px;
        }
        .hero p {
            color: #b9c0ca;
            font-size: 16px;
            margin: 0;
        }
        .section-title {
            color: #f2c66d;
            font-size: 25px;
            font-weight: 700;
            margin: 12px 0 8px 0;
        }
        .metric-card {
            background: #111722;
            border: 1px solid #292f39;
            border-radius: 14px;
            padding: 17px 18px;
            height: 100%;
        }
        .metric-label {
            color: #8e98a6;
            font-size: 13px;
            margin-bottom: 5px;
        }
        .metric-value {
            color: #f4f4f4;
            font-size: 28px;
            font-weight: 700;
        }
        .metric-note {
            color: #aeb6c1;
            font-size: 12px;
            margin-top: 3px;
        }
        .insight {
            background: #10151d;
            border-left: 4px solid #f2c66d;
            border-radius: 10px;
            padding: 13px 16px;
            margin-bottom: 10px;
            color: #d9dde4;
        }
        .small-note {
            color: #9099a6;
            font-size: 12px;
        }
        div[data-testid="stMetricValue"] {
            color: #f2c66d;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Data loading and preparation
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIE_PATH = os.path.join(BASE_DIR, "data", "movie_data.csv")
TOP_PATH = os.path.join(BASE_DIR, "data", "Top_rated_movies.csv")


@st.cache_data
def load_data():
    """Load the two supplied CSV files."""
    df = pd.read_csv(MOVIE_PATH)
    top_df = pd.read_csv(TOP_PATH)
    return df, top_df


@st.cache_data
def prepare_data(raw_df):
    """Apply the same simple cleaning and feature engineering used in the notebook."""
    df = raw_df.copy()

    # Clean text.
    df["title"] = df["title"].astype(str).str.strip()
    df["overview"] = df["overview"].fillna("")

    # Convert dates and create year.
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"] = df["release_date"].dt.year

    # Remove duplicates and rows without a release date.
    df = df.drop_duplicates()
    df = df.dropna(subset=["release_date"])

    # Convert numeric fields safely.
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce")
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")

    # Fill any numeric gaps.
    df["vote_average"] = df["vote_average"].fillna(df["vote_average"].median())
    df["vote_count"] = df["vote_count"].fillna(0)

    # Keep realistic rating and vote ranges.
    df = df[(df["vote_average"] >= 0) & (df["vote_average"] <= 10)]
    df = df[df["vote_count"] >= 0]

    # Feature engineering.
    df["title_length"] = df["title"].str.len()
    df["overview_length"] = df["overview"].str.len()
    df["decade"] = (df["year"] // 10 * 10).astype(int)
    df["log_votes"] = np.log1p(df["vote_count"])

    # ML target: 1 means highly rated (rating >= 8).
    df["highly_rated"] = (df["vote_average"] >= 8.0).astype(int)
    return df


@st.cache_data
def prepare_top_data(raw_top):
    """Remove repeated copies of the same movie from the reference table."""
    top_df = raw_top.copy()
    top_df["title"] = top_df["title"].astype(str).str.strip()
    top_df["release_date"] = pd.to_datetime(top_df["release_date"], errors="coerce")
    top_df = top_df.drop_duplicates(subset="id")
    return top_df


@st.cache_resource
def train_model(df):
    """Train the notebook's Random Forest model and return evaluation results."""
    X = df[[
        "year",
        "vote_count",
        "log_votes",
        "title_length",
        "overview_length",
        "original_language",
    ]].copy()

    # Convert language text into numeric dummy columns.
    X = pd.get_dummies(X, columns=["original_language"], drop_first=True, dtype=int)
    y = df["highly_rated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Not Highly Rated", "Highly Rated"],
        output_dict=True,
        zero_division=0,
    )

    return model, X.columns.tolist(), accuracy, matrix, report


raw_df, raw_top_df = load_data()
df = prepare_data(raw_df)
top_df = prepare_top_data(raw_top_df)
model, model_columns, accuracy, cm, report = train_model(df)


# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.markdown("## 🎬 Dashboard Controls")
st.sidebar.caption("Use the filters to explore the cleaned movie data.")

min_year = int(df["year"].min())
max_year = int(df["year"].max())
year_range = st.sidebar.slider(
    "Release year",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
)

languages = sorted(df["original_language"].dropna().unique().tolist())
selected_languages = st.sidebar.multiselect(
    "Original language",
    languages,
    default=languages,
)

min_rating = st.sidebar.slider(
    "Minimum rating",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.1,
)

filtered = df[
    (df["year"].between(year_range[0], year_range[1]))
    & (df["original_language"].isin(selected_languages))
    & (df["vote_average"] >= min_rating)
].copy()


# -----------------------------
# Hero
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎬 Movie Intelligence Dashboard</h1>
        <p>Interactive analysis of the Movie Dataset (1957–2026 project) with EDA, top-rated insights and machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Source: supplied Kaggle notebook and the two project CSV files. "
    "The dashboard uses the same core cleaning, feature engineering and Random Forest approach from the final notebook."
)


# -----------------------------
# KPI cards
# -----------------------------
st.markdown('<div class="section-title">Project Data Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
metrics = [
    ("Movies after cleaning", f"{len(df):,}", "Main movie table"),
    ("Average rating", f"{df['vote_average'].mean():.2f}", "0–10 scale"),
    ("Languages", f"{df['original_language'].nunique():,}", "Original languages"),
    ("Highly rated", f"{df['highly_rated'].mean()*100:.1f}%", "Rating ≥ 8.0"),
    ("Top-table unique movies", f"{len(top_df):,}", f"{len(raw_top_df):,} raw rows"),
]
for col, (label, value, note) in zip([k1, k2, k3, k4, k5], metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

# Quick data quality notes.
q1, q2 = st.columns([1, 1])
with q1:
    st.markdown(
        f'<div class="insight">The main dataset contains <b>{len(raw_df):,}</b> raw rows. After removing duplicates and rows with missing release dates, <b>{len(df):,}</b> rows remain.</div>',
        unsafe_allow_html=True,
    )
with q2:
    st.markdown(
        f'<div class="insight">The top-rated reference file contains <b>{len(raw_top_df):,}</b> raw rows but only <b>{len(top_df):,}</b> unique movie IDs, so repeated copies are removed for reference analysis.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------
# Main tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 EDA & Visualizations",
    "🏆 Top Rated",
    "🤖 Machine Learning",
    "🔎 Data Explorer",
])

with tab1:
    st.markdown('<div class="section-title">Filtered Overview</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Filtered movies", f"{len(filtered):,}")
    b.metric("Average rating", f"{filtered['vote_average'].mean():.2f}" if len(filtered) else "—")
    c.metric("Average votes", f"{filtered['vote_count'].mean():,.0f}" if len(filtered) else "—")
    d.metric("Highly rated", f"{filtered['highly_rated'].mean()*100:.1f}%" if len(filtered) else "—")

    st.markdown("### Key Insights")
    if len(filtered):
        best_decade = filtered.groupby("decade")["vote_average"].mean().idxmax()
        best_language = filtered.groupby("original_language").size().idxmax()
        most_voted = filtered.sort_values("vote_count", ascending=False).iloc[0]
        st.write(f"• The highest average-rated decade in the selected data is **{int(best_decade)}s**.")
        st.write(f"• The most common original language in the selected data is **{best_language}**.")
        st.write(f"• The most-voted selected movie is **{most_voted['title']}** with **{most_voted['vote_count']:,} votes**.")

    st.markdown("### Dataset Structure")
    overview_cols = [
        "id", "title", "original_language", "release_date", "vote_average", "vote_count",
        "year", "title_length", "overview_length", "decade", "log_votes", "highly_rated"
    ]
    st.dataframe(filtered[overview_cols].head(20), use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    st.markdown("### 1. Movies Released by Year")
    year_count = filtered.groupby("year").size().reset_index(name="movie_count")
    fig = px.line(year_count, x="year", y="movie_count", markers=True, template="plotly_dark")
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Year", yaxis_title="Movies")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 2. Rating Distribution")
        fig = px.histogram(filtered, x="vote_average", nbins=20, template="plotly_dark")
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Rating", yaxis_title="Movies")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### 3. Movies by Decade")
        decade_count = filtered.groupby("decade").size().reset_index(name="movie_count")
        fig = px.bar(decade_count, x="decade", y="movie_count", template="plotly_dark")
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Decade", yaxis_title="Movies")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### 4. Average Rating by Decade")
        decade_rating = filtered.groupby("decade")["vote_average"].mean().reset_index()
        fig = px.line(decade_rating, x="decade", y="vote_average", markers=True, template="plotly_dark")
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Decade", yaxis_title="Average rating")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.markdown("### 5. Top 10 Movies by Vote Count")
        top_votes = filtered.sort_values("vote_count", ascending=False).head(10).sort_values("vote_count")
        fig = px.bar(top_votes, x="vote_count", y="title", orientation="h", template="plotly_dark")
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Vote count", yaxis_title="Movie")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 6. Vote Count vs Rating")
    fig = px.scatter(
        filtered,
        x="vote_count",
        y="vote_average",
        hover_name="title",
        opacity=0.55,
        template="plotly_dark",
    )
    fig.update_layout(height=410, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Vote count", yaxis_title="Rating")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 7. Movies by Original Language")
    language_count = filtered.groupby("original_language").size().sort_values(ascending=False).head(10).reset_index(name="movie_count")
    fig = px.bar(language_count.sort_values("movie_count"), x="movie_count", y="original_language", orientation="h", template="plotly_dark")
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Movies", yaxis_title="Language")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">Top-Rated Reference Movies</div>', unsafe_allow_html=True)
    st.write(
        "This table comes from the supplied `Top_rated_movies (1).csv`. "
        "The source contains repeated copies of the same movie IDs, so the dashboard keeps one row per movie."
    )

    sort_col = st.selectbox("Sort reference movies by", ["rating", "vote_count", "popularity"], index=0)
    if sort_col in top_df.columns:
        display_top = top_df.sort_values(sort_col, ascending=False)
    else:
        display_top = top_df
    show_cols = [c for c in ["title", "original_language", "release_date", "rating", "vote_count", "popularity"] if c in display_top.columns]
    st.dataframe(display_top[show_cols], use_container_width=True, hide_index=True)

    st.markdown("### Highest-Rated Reference Movies")
    highest = top_df.sort_values("rating", ascending=False).head(10)
    fig = px.bar(highest.sort_values("rating"), x="rating", y="title", orientation="h", template="plotly_dark")
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=35, b=20), xaxis_title="Rating", yaxis_title="Movie")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Machine Learning — Highly Rated Movie Classifier</div>', unsafe_allow_html=True)
    st.write(
        "The model follows the notebook: a movie is **Highly Rated** when `vote_average >= 8.0`. "
        "The rating itself is not used as an input feature, so the model learns from other movie information."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Model", "Random Forest")
    m2.metric("Test accuracy", f"{accuracy*100:.2f}%")
    m3.metric("Requirement", "✅ Above 80%" if accuracy >= 0.80 else "⚠️ Below 80%")

    st.markdown("### How the model works")
    st.write("1. 80% of the cleaned data is used for training.")
    st.write("2. The Random Forest learns patterns from year, votes, log-votes, title length, overview length and original language.")
    st.write("3. The remaining 20% is unseen test data.")
    st.write("4. The model predicts `Highly Rated` or `Not Highly Rated`.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Confusion Matrix")
        heat = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Not Highly Rated", "Highly Rated"],
            y=["Not Highly Rated", "Highly Rated"],
            text=cm,
            texttemplate="%{text}",
            colorscale="YlOrBr",
            showscale=False,
        ))
        heat.update_layout(height=350, xaxis_title="Predicted", yaxis_title="Actual", template="plotly_dark")
        st.plotly_chart(heat, use_container_width=True)
    with col_b:
        st.markdown("### Test Metrics")
        metric_df = pd.DataFrame({
            "Metric": ["Accuracy", "Highly Rated Precision", "Highly Rated Recall", "Highly Rated F1"],
            "Score": [
                report["accuracy"],
                report["Highly Rated"]["precision"],
                report["Highly Rated"]["recall"],
                report["Highly Rated"]["f1-score"],
            ],
        })
        metric_df["Score"] = (metric_df["Score"] * 100).round(2).astype(str) + "%"
        st.dataframe(metric_df, hide_index=True, use_container_width=True)
        st.caption("Accuracy is useful for the requested threshold, but the class-level metrics show that the highly-rated class is harder to identify.")

    st.markdown("### Try Your Own Movie")
    p1, p2 = st.columns(2)
    with p1:
        sample_title = st.text_input("Movie title", "The Shawshank Redemption")
        sample_year = st.number_input("Release year", min_value=1900, max_value=2100, value=1994, step=1)
        sample_votes = st.number_input("Vote count", min_value=0, value=30472, step=100)
    with p2:
        sample_overview_length = st.number_input("Overview length", min_value=0, value=392, step=10)
        lang_options = sorted(df["original_language"].unique().tolist())
        default_lang_index = lang_options.index("en") if "en" in lang_options else 0
        sample_language = st.selectbox("Original language", lang_options, index=default_lang_index)
        st.info("The model predicts using metadata only; it does not take the movie's rating as an input.")

    if st.button("Predict Movie", type="primary"):
        sample = pd.DataFrame({
            "year": [sample_year],
            "vote_count": [sample_votes],
            "log_votes": [np.log1p(sample_votes)],
            "title_length": [len(sample_title)],
            "overview_length": [sample_overview_length],
            "original_language": [sample_language],
        })
        sample = pd.get_dummies(sample, columns=["original_language"], drop_first=True, dtype=int)
        sample = sample.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0][1]
        if prediction == 1:
            st.success(f"{sample_title} → Highly Rated")
        else:
            st.warning(f"{sample_title} → Not Highly Rated")
        st.write(f"Estimated probability of the Highly Rated class: **{probability*100:.1f}%**")

    st.markdown("### Target Distribution")
    class_counts = df["highly_rated"].value_counts().rename(index={0: "Not Highly Rated", 1: "Highly Rated"})
    fig = px.bar(class_counts.reset_index(name="count"), x="highly_rated", y="count", template="plotly_dark")
    fig.update_layout(height=300, xaxis_title="Class", yaxis_title="Movies")
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown('<div class="section-title">Data Explorer</div>', unsafe_allow_html=True)
    st.write(f"Showing **{len(filtered):,}** filtered rows.")

    search = st.text_input("Search movie title", "")
    explorer = filtered.copy()
    if search:
        explorer = explorer[explorer["title"].str.contains(search, case=False, na=False)]

    show_cols = [
        "id", "title", "original_language", "release_date", "vote_average",
        "vote_count", "popularity", "year", "decade", "highly_rated"
    ]
    show_cols = [c for c in show_cols if c in explorer.columns]
    st.dataframe(explorer[show_cols], use_container_width=True, hide_index=True)

    csv = explorer[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv,
        file_name="filtered_movies.csv",
        mime="text/csv",
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    '<div class="small-note">Movie Intelligence Dashboard • Built with Streamlit • Data analysis follows the supplied project notebook.</div>',
    unsafe_allow_html=True,
)
