# 🎬 Movie Intelligence Dashboard

A Streamlit dashboard built from the supplied Movie Dataset project notebook and the two project CSV files.

## Included

- Project data overview and KPIs
- Data-quality summary
- Release-year and decade trends
- Rating distribution
- Vote count vs rating
- Top movies by vote count
- Movies by original language
- Top-rated reference table
- Random Forest machine-learning summary
- Test accuracy and confusion matrix
- Interactive movie prediction
- Filtered data explorer and CSV download

## Dataset files

The app reads the files stored in `data/`:

- `data/movie_data.csv`
- `data/Top_rated_movies.csv`

These are the supplied project files.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `.streamlit/config.toml`, and the `data/` folder.
3. In Streamlit Community Cloud, create a new app from the repository.
4. Set the main file to `app.py`.
5. Deploy.

No API key is required.
