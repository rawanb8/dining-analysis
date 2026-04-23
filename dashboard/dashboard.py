import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import json, os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from feature_analysis import render_feature_analysis
from best_for_tags import render_best_for_tags
# PAGE CONFIGURATION

st.set_page_config(
    page_title="Lebanon Restaurant Analysis",
    page_icon=":material/restaurant:",
    layout="wide"
)

# LOAD DATA

@st.cache_data
@st.cache_data
def load_data():
    # Always load master_restaurants.csv as primary data source
    restaurants = pd.read_csv(r"../merged/master_restaurants.csv")
    print("✓ Loaded master restaurant data")
    
    try:
        restaurants_geocoded = pd.read_csv(r"../merged/master_restaurants_geocoded.csv")
        print("✓ Loaded geocoded data for maps")
        
        # Merge lat/long from geocoded file into main restaurants dataframe
        if 'latitude' in restaurants_geocoded.columns and 'longitude' in restaurants_geocoded.columns:
            restaurants = restaurants.merge(
                restaurants_geocoded[['restaurant_id', 'latitude', 'longitude']],
                on='restaurant_id',
                how='left',
                suffixes=('', '_geocoded')
            )
            print(f"✓ Merged coordinates for {restaurants['latitude'].notna().sum()} restaurants")
    except FileNotFoundError:
        print("⚠️ Geocoded file not found, maps will not display")
    
    reviews = pd.read_csv(r"../merged/master_reviews.csv")
    return restaurants, reviews

df_restaurants, df_reviews = load_data()

# Prepare display data
df_display = df_restaurants.copy()

# Convert feature columns for display
feature_cols = [
    'delivery_available', 'outdoor_seating', 'reservation_required',
    'cash_only', 'credit_cards_accepted', 'wifi_available',
    'wheelchair_accessible', 'takeaway_available', 'parking_available',
    'live_music', 'pet_friendly', 'kids_friendly'
]

for col in feature_cols:
    df_display[col] = df_display[col].apply(lambda x: '✓' if x == 'TRUE' else '')

# HEADER

st.title("Lebanese Restaurant Analysis Dashboard")
st.write("COSC 482 - Data Science Project | Interactive exploration of Lebanese restaurants")
st.write("---")

# TOP NAVIGATION

selected_section = option_menu(
    menu_title=None,
    options=["Search & Filter", "EDA", "Feature Analysis", "ML Insights", "NLP Analysis", "Curated Smart Picks"],
    icons=["search", "bar-chart-line", "toggles", "chat-left-text", "cpu", "lightbulb"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0", "margin": "0 0 1rem 0"},
        "nav-link-selected": {"background-color": "#3ecda6"},
    }
)

st.write("---")

# SECTION 1: SEARCH & FILTER

if selected_section == "Search & Filter":
    st.header(":material/search: Search and Filter Restaurants")
    
    # Filters in sidebar
    st.sidebar.subheader("Filters")
    
    # Text search
    search_name = st.sidebar.text_input(" Search by Name:").strip().lower()
    
    # Cuisine filter
    cuisine_options = ["All Cuisines"] + sorted(df_restaurants['cuisine_primary'].dropna().astype(str).unique().tolist())
    selected_cuisine = st.sidebar.selectbox(" Cuisine Type:", cuisine_options)
    
    # Area filter
    area_options = ["All Areas"] + sorted(df_restaurants['area'].dropna().astype(str).unique().tolist())
    selected_area = st.sidebar.selectbox(" Area:", area_options)
    
    # Price filter
    price_options = ["All Prices", "Budget", "Mid-Range", "High-End"]
    selected_price = st.sidebar.selectbox(" Price Category:", price_options)
    
    # Rating filter
    min_rating = st.sidebar.slider(
        " Minimum Rating:",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.5
    )
    
    # Feature filters
    st.sidebar.subheader("Features")

    # Organize in 2 columns for better layout
    col1, col2 = st.sidebar.columns(2)

    with col1:
        filter_delivery = st.checkbox("Delivery", key="s1_delivery")
        filter_takeaway = st.checkbox("Takeaway", key="s1_takeaway")
        filter_outdoor = st.checkbox("Outdoor", key="s1_outdoor")
        filter_parking = st.checkbox("Parking", key="s1_parking")
        filter_wifi = st.checkbox("WiFi", key="s1_wifi")
        filter_music = st.checkbox("Live Music", key="s1_music")

    with col2:
        filter_reservation = st.checkbox("Reservation", key="s1_reservation")
        filter_credit = st.checkbox("Credit Cards", key="s1_credit")
        filter_cash = st.checkbox("Cash Only", key="s1_cash")
        filter_wheelchair = st.checkbox("Wheelchair", key="s1_wheelchair")
        filter_pet = st.checkbox("Pet Friendly", key="s1_pet")
        filter_kids = st.checkbox("Kids Friendly", key="s1_kids")

    # Apply filters
    filtered_df = df_display.copy()

    if search_name:
        filtered_df = filtered_df[filtered_df['name'].str.lower().str.contains(search_name, na=False)]

    if selected_cuisine != "All Cuisines":
        filtered_df = filtered_df[filtered_df['cuisine_primary'] == selected_cuisine]

    if selected_area != "All Areas":
        filtered_df = filtered_df[filtered_df['area'] == selected_area]

    if selected_price != "All Prices":
        filtered_df = filtered_df[filtered_df['price_category'] == selected_price]

    if min_rating > 0:
        filtered_df = filtered_df[filtered_df['rating_overall'] >= min_rating]

    # Apply ALL 12 feature filters
    if filter_delivery:
        filtered_df = filtered_df[filtered_df['delivery_available'] == '✓']
    if filter_takeaway:
        filtered_df = filtered_df[filtered_df['takeaway_available'] == '✓']
    if filter_outdoor:
        filtered_df = filtered_df[filtered_df['outdoor_seating'] == '✓']
    if filter_parking:
        filtered_df = filtered_df[filtered_df['parking_available'] == '✓']
    if filter_wifi:
        filtered_df = filtered_df[filtered_df['wifi_available'] == '✓']
    if filter_music:
        filtered_df = filtered_df[filtered_df['live_music'] == '✓']
    if filter_reservation:
        filtered_df = filtered_df[filtered_df['reservation_required'] == '✓']
    if filter_credit:
        filtered_df = filtered_df[filtered_df['credit_cards_accepted'] == '✓']
    if filter_cash:
        filtered_df = filtered_df[filtered_df['cash_only'] == '✓']
    if filter_wheelchair:
        filtered_df = filtered_df[filtered_df['wheelchair_accessible'] == '✓']
    if filter_pet:
        filtered_df = filtered_df[filtered_df['pet_friendly'] == '✓']
    if filter_kids:
        filtered_df = filtered_df[filtered_df['kids_friendly'] == '✓']
    
    # Display results
    st.subheader(f":material/table_rows: Showing {len(filtered_df)} of {len(df_display)} restaurants")
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total Restaurants", len(filtered_df))
    col2.metric("Average Rating", f"{filtered_df['rating_overall'].mean():.2f}⭐")
    col3.metric("Total Reviews", f"{len(df_reviews):,}")
    col4.metric("Avg Reviews/Restaurant", f"{filtered_df['review_count_total'].mean():.0f}")
    col5.metric("High-End Count", len(filtered_df[filtered_df['price_category'] == 'High-End']))
    
    st.write("---")
    
    # Display table
    display_cols = [
        'name', 'cuisine_primary', 'area', 'rating_overall', 'review_count_total',
        'price_category', 'delivery_available', 'outdoor_seating', 'parking_available',
        'wifi_available', 'live_music', 'phone'
    ]
    
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        height=600
    )
    
    # Top rated restaurants
    st.subheader(":material/emoji_events: Top 10 Highest Rated")
    top_rated = filtered_df.nlargest(10, 'rating_overall')[
        ['name', 'rating_overall', 'review_count_total', 'area', 'cuisine_primary', 'price_category']
    ]
    st.dataframe(top_rated, use_container_width=True)
    
    # Expandable full dataset
    with st.expander(f" Click to view full filtered dataset ({len(filtered_df)} records)"):
        st.dataframe(filtered_df, use_container_width=True)

# SECTION 2: GENERAL ANALYSIS 

elif selected_section == "EDA":
    st.header("Exploratory Data Analysis")

    st.sidebar.subheader("EDA Filters")

    eda_cuisine_opts = ["All Cuisines"] + sorted(
        df_restaurants[df_restaurants["cuisine_primary"] != "Unknown"]["cuisine_primary"].unique().tolist()
    )
    eda_cuisine = st.sidebar.selectbox("Cuisine:", eda_cuisine_opts, key="eda_cuisine")



    eda_price_opts = ["All Prices"] + sorted(
        df_restaurants[df_restaurants["price_category"] != "Unknown"]["price_category"].unique().tolist()
    )
    eda_price = st.sidebar.selectbox("Price Category:", eda_price_opts, key="eda_price")

    eda_min_rating = st.sidebar.slider(
        "Min Rating:", min_value=0.0, max_value=5.0, value=0.0, step=0.5, key="eda_rating"
    )

    only_with_reviews = st.sidebar.checkbox("Only restaurants with reviews", value=False)

    # APPLY FILTERS 
    df_eda = df_restaurants.copy()

    if eda_cuisine != "All Cuisines":
        df_eda = df_eda[df_eda["cuisine_primary"] == eda_cuisine]

    if eda_price != "All Prices":
        df_eda = df_eda[df_eda["price_category"] == eda_price]
    if eda_min_rating > 0:
        df_eda = df_eda[df_eda["rating_overall"] >= eda_min_rating]


    df_eda["review_count_total"] = pd.to_numeric(df_eda["review_count_total"], errors="coerce")
    if only_with_reviews:
        df_eda = df_eda[df_eda["review_count_total"] > 0]

    # ACTIVE FILTER BANNER
    active_filters = []
    if eda_cuisine != "All Cuisines":    active_filters.append(f"Cuisine: {eda_cuisine}")
    if eda_price != "All Prices":        active_filters.append(f"Price: {eda_price}")
    if eda_min_rating > 0:              active_filters.append(f"Rating ≥ {eda_min_rating}")
    if only_with_reviews:
        active_filters.append("With Reviews Only")

    if active_filters:
        st.info(f"Active filters: {' · '.join(active_filters)} — showing **{len(df_eda)}** of **{len(df_restaurants)}** restaurants")
    else:
        st.info(f"Showing all **{len(df_eda)}** restaurants. Use the sidebar to filter.")

    st.write("---")

    # SUMMARY METRICS 
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Restaurants", len(df_eda))
    avg_rating = df_eda["rating_overall"].mean()

    col2.metric(
        "Avg Rating",
        f"{avg_rating:.2f}" if pd.notna(avg_rating) else "—"
    )
    # Count reviews from filtered restaurants
    filtered_restaurant_ids = df_eda['restaurant_id'].unique()
    review_count = len(df_reviews[df_reviews['restaurant_id'].isin(filtered_restaurant_ids)])
    col3.metric("Total Reviews", f"{review_count:,}" if len(df_eda) else "—")
    col4.metric("Areas", df_eda[df_eda['area'] != 'Unknown']['area'].nunique())
    col5.metric("Cuisine Types", df_eda[df_eda['cuisine_primary'] != 'Unknown']['cuisine_primary'].nunique())

    st.write("---")

    if df_eda.empty:
        st.warning("No restaurants match the selected filters.")
    else:
        #  Cuisine Distribution
        st.subheader(" Cuisine Distribution (Top 10 — Excluding Unknown)")

        if eda_cuisine != "All Cuisines":
            st.info(f"Showing only **{eda_cuisine}** restaurants — cuisine distribution not applicable.")

        else:
            unknown_count = (df_eda["cuisine_primary"] == "Unknown").sum()

            if unknown_count > 0:
                unknown_pct = (unknown_count / len(df_eda)) * 100
                st.info(f"⚠️ {unknown_pct:.1f}% of filtered restaurants have unknown cuisine.")

            cuisine_counts = df_eda[df_eda["cuisine_primary"] != "Unknown"]["cuisine_primary"].value_counts().head(10)

            if cuisine_counts.empty:
                st.warning("No cuisine data available after filtering.")
            else:
                fig1 = px.bar(
                    x=cuisine_counts.index,
                    y=cuisine_counts.values,
                    labels={'x': 'Cuisine', 'y': 'Number of Restaurants'},
                    title='Top 10 Cuisines',
                    text=cuisine_counts.values
                )
                fig1.update_traces(textposition='outside')
                fig1.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig1, use_container_width=True)

        # Rating Distribution
        st.subheader(" Rating Distribution")
        rating_series = pd.to_numeric(df_eda["rating_overall"], errors="coerce").dropna()

        if rating_series.empty:
            st.info("No valid rating data to display for the selected filters.")
        else:
            rating_counts = rating_series.value_counts().sort_index()

            fig2 = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={"x": "Rating", "y": "Number of Restaurants"},
                title="Distribution of Restaurant Ratings"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Price Category
        st.subheader(" Price Category Breakdown (Excluding Unknown)")
        price_counts = df_eda[df_eda["price_category"] != "Unknown"]["price_category"].value_counts()
        unknown_price_count = (df_eda["price_category"] == "Unknown").sum()

        if unknown_price_count > 0:
            unknown_price_pct = (unknown_price_count / len(df_eda)) * 100
            st.info(f"⚠️ {unknown_price_pct:.1f}% of filtered restaurants have unknown price category.")
        if len(price_counts):
            fig3 = px.bar(x=price_counts.index, y=price_counts.values,
                        labels={"x": "Price Category", "y": "Number of Restaurants"},
                        title="Distribution by Price Category", text=price_counts.values)
            fig3.update_traces(textposition='outside')
            st.plotly_chart(fig3, use_container_width=True)

        st.write("---")

        st.subheader(" Price vs Rating")

        price_rating_df = df_eda.copy()
        price_rating_df["rating_overall"] = pd.to_numeric(price_rating_df["rating_overall"], errors="coerce")

        price_rating_df = price_rating_df.dropna(subset=["rating_overall"])
        price_rating_df = price_rating_df[price_rating_df["price_category"] != "Unknown"]

        if price_rating_df.empty:
            st.info("No valid price vs rating data for the selected filters.")
        else:
            fig = px.box(
                price_rating_df,
                x="price_category",
                y="rating_overall",
                color="price_category",
                category_orders={"price_category": ["Budget", "Mid-Range", "High-End"]},
                title="Rating Distribution by Price Category"
            )

            fig.update_traces(boxmean=True)

            st.plotly_chart(fig, use_container_width=True)

        st.write("---")
        st.subheader(" Top Rated Restaurants")

        top_rated_clean = df_eda.copy()
        top_rated_clean["rating_overall"] = pd.to_numeric(top_rated_clean["rating_overall"], errors="coerce")
        top_rated_clean = top_rated_clean.dropna(subset=["rating_overall"])

        if top_rated_clean.empty:
            st.info("No valid rating data available for the selected filters.")
        else:
            top_rated = top_rated_clean.nlargest(10, "rating_overall")[
                ["name", "rating_overall", "review_count_total", "area", "cuisine_primary", "price_category"]
            ].copy()
            top_rated.columns = ["Restaurant Name", "Rating", "Total Reviews", "Area", "Cuisine", "Price Category"]
            st.dataframe(top_rated, use_container_width=True)

        st.write("---")

        st.subheader(" Lowest Rated Restaurants")

        worst_rated_clean = df_eda.copy()
        worst_rated_clean["rating_overall"] = pd.to_numeric(worst_rated_clean["rating_overall"], errors="coerce")
        worst_rated_clean = worst_rated_clean.dropna(subset=["rating_overall"])

        if worst_rated_clean.empty:
            st.info("No valid rating data available for the selected filters.")
        else:
            worst_rated = worst_rated_clean.nsmallest(10, "rating_overall")[
                ["name", "rating_overall", "review_count_total", "area", "cuisine_primary", "price_category"]
            ].copy()
            worst_rated.columns = ["Restaurant Name", "Rating", "Total Reviews", "Area", "Cuisine", "Price Category"]
            st.dataframe(worst_rated, use_container_width=True)

        st.write("---")
        #  Restaurants by Area
        st.subheader(" Restaurants by Area (Top 10 — Excluding Unknown)")
        area_counts = df_eda[df_eda["area"] != "Unknown"]["area"].value_counts().head(10)
        unknown_area_count = (df_eda["area"] == "Unknown").sum()

        if unknown_area_count > 0:
            unknown_area_pct = (unknown_area_count / len(df_eda)) * 100
            st.info(f"⚠️ {unknown_area_pct:.1f}% of filtered restaurants have unknown area.")
        if len(area_counts):
            fig4 = px.bar(x=area_counts.index, y=area_counts.values,
                        labels={"x": "Area", "y": "Number of Restaurants"},
                        title="Top 10 Areas by Number of Restaurants", text=area_counts.values)
            fig4.update_traces(textposition='outside')
            fig4.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig4, use_container_width=True)

        st.write("---")

        # Review Count Distribution
        st.subheader(" Review Count Distribution")
        if len(df_eda):
            fig5 = px.histogram(df_eda, x="review_count_total", nbins=30,
                                title="Distribution of Review Counts",
                                labels={"review_count_total": "Number of Reviews"})
            fig5.update_layout(yaxis_title="Number of Restaurants")
            st.plotly_chart(fig5, use_container_width=True)

        st.write("---")

        # Top 10 Most Reviewed
        st.subheader(" Top 10 Most Reviewed Restaurants")
        if len(df_eda):
            top_reviewed = df_eda.nlargest(10, "review_count_total")[
                ["name", "review_count_total", "rating_overall", "area", "cuisine_primary", "price_category"]
            ].copy()
            top_reviewed.columns = ["Restaurant Name", "Total Reviews", "Rating", "Area", "Cuisine", "Price Category"]
            st.dataframe(top_reviewed, use_container_width=True)

        st.write("---")

        # Rating vs Review Count
        st.subheader(" Rating vs Review Count")

        # Clean data
        scatter_df = df_eda.copy()

        scatter_df["rating_overall"] = pd.to_numeric(scatter_df["rating_overall"], errors="coerce")
        scatter_df["review_count_total"] = pd.to_numeric(scatter_df["review_count_total"], errors="coerce")

        # Remove invalid rows
        scatter_df = scatter_df.dropna(subset=["rating_overall", "review_count_total"])
        scatter_df = scatter_df[scatter_df["review_count_total"] > 0]

        # Check if data exists
        if scatter_df.empty:
            st.info("No valid rating vs review data to display for the selected filters.")
        else:
            fig = px.scatter(
                scatter_df,
                x="review_count_total",
                y="rating_overall",
                title="Rating vs Number of Reviews",
                labels={
                    "review_count_total": "Number of Reviews",
                    "rating_overall": "Rating"
                }
            )

            fig.update_xaxes(type="log")  # optional but nice
            st.plotly_chart(fig, use_container_width=True)

        #  Hidden Gems
        st.subheader(":material/diamond: Hidden Gems (Rating ≥ 4.5, Reviews < 50)")
        hidden = df_eda[
            (df_eda["rating_overall"] >= 4.5) & (df_eda["review_count_total"] < 50)
        ].head(10)[["name", "rating_overall", "review_count_total", "area"]]
        if len(hidden):
            st.dataframe(hidden, use_container_width=True)
        else:
            st.info("No hidden gems match the current filters.")

        st.write("---")

        # Density vs Quality by Area
        st.subheader(":material/bubble_chart: Density vs Quality by Area")

        df_geo_eda = df_eda[df_eda["area"] != "Unknown"].copy()

        if df_geo_eda.empty:
            st.info("No area data available for the selected filters.")
        else:
            # Clean data
            df_geo_eda["rating_overall"] = pd.to_numeric(df_geo_eda["rating_overall"], errors="coerce")
            df_geo_eda["review_count_total"] = pd.to_numeric(df_geo_eda["review_count_total"], errors="coerce")

            area_stats = df_geo_eda.groupby("area").agg(
                restaurant_count=("name", "count"),
                rating_overall=("rating_overall", "mean"),
                total_reviews=("review_count_total", "sum")
            )

            # Drop invalid rows
            area_stats = area_stats.dropna(subset=["rating_overall"])

            if area_stats.empty:
                st.info("No valid area statistics to display.")
            else:
                # OPTIONAL: only apply >=3 rule if enough data exists
                if len(area_stats) > 10:
                    area_stats = area_stats[area_stats["restaurant_count"] >= 3]

                fig_bubble = px.scatter(
                    area_stats,
                    x="restaurant_count",
                    y="rating_overall",
                    size="total_reviews",
                    text=area_stats.index,
                    title="Density vs Quality by Area (Bubble = Popularity)"
                )

                fig_bubble.update_traces(textposition="top center")
                st.plotly_chart(fig_bubble, use_container_width=True)

        # Best Areas Combined Score
        st.subheader(":material/emoji_events: Best Areas Overall (Combined Score)")
        if area_stats.empty:
            st.info("No area data available for the selected filters.")
        else:
            if len(df_geo_eda):
                area_stats2 = df_geo_eda.groupby("area").agg(
                    rating_overall=("rating_overall", "mean"),
                    review_count_total=("review_count_total", "sum"),
                    restaurant_count=("name", "count")
                )
                area_stats2 = area_stats2[area_stats2["restaurant_count"] >= 3]
                if len(area_stats2):
                    area_stats2["norm_rating"]  = area_stats2["rating_overall"] / 5
                    area_stats2["norm_reviews"] = area_stats2["review_count_total"] / area_stats2["review_count_total"].max()
                    area_stats2["norm_density"] = area_stats2["restaurant_count"] / area_stats2["restaurant_count"].max()
                    area_stats2["score"] = (area_stats2["norm_rating"] + area_stats2["norm_reviews"] + area_stats2["norm_density"]) / 3

                    top_areas2 = area_stats2.sort_values("score", ascending=False).head(10)
                    fig_best = px.bar(x=top_areas2.index, y=top_areas2["score"],
                                    title="Top Areas (Quality + Popularity + Density)",
                                    labels={"x": "Area", "y": "Score"},
                                    text=[f"{v:.2f}" for v in top_areas2["score"]])
                    fig_best.update_traces(textposition='outside')
                    fig_best.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_best, use_container_width=True)

# SECTION 3: FEATURE ANALYSIS

elif selected_section == "Feature Analysis":
    render_feature_analysis(df_restaurants, feature_cols)

# SECTION 4: ML INSIGHTS 

elif selected_section == "ML Insights":
    st.header(":material/smart_toy: ML Insights — Cuisine Classifier")
    st.write("---")

    ML_DIR = os.path.join(os.path.dirname(__file__), '..', 'machine_learning')

    @st.cache_data
    def load_ml_data():
        with open(os.path.join(ML_DIR, 'ml_cuisine_summary.json'), 'r') as f:
            summary = json.load(f)
        df_model_comparison  = pd.read_csv(os.path.join(ML_DIR, 'ml_model_comparison.csv'))
        df_balancing  = pd.read_csv(os.path.join(ML_DIR, 'ml_balancing_comparison.csv'))
        df_final_comparison  = pd.read_csv(os.path.join(ML_DIR, 'ml_final_comparison.csv'))
        df_before_after = pd.read_csv(os.path.join(ML_DIR, 'ml_before_after.csv'))
        df_pred_dist = pd.read_csv(os.path.join(ML_DIR, 'ml_predicted_distribution.csv'))
        df_confusion= pd.read_csv(os.path.join(ML_DIR, 'ml_confusion_matrix.csv'), index_col=0)
        df_tuning  = pd.read_csv(os.path.join(ML_DIR, 'ml_tuning_before_after.csv'))
        df_cv_folds  = pd.read_csv(os.path.join(ML_DIR, 'ml_cv_fold_scores.csv'))
        df_grid = pd.read_csv(os.path.join(ML_DIR, 'ml_grid_search_results.csv'))
        df_restaurant_preds = pd.read_csv(os.path.join(ML_DIR, 'ml_restaurant_level_predictions.csv'))
        df_rest_vs_rev_f1   = pd.read_csv(os.path.join(ML_DIR, 'ml_restaurant_vs_review_f1.csv'))
        return (summary, df_model_comparison, df_balancing, df_final_comparison,
                df_before_after, df_pred_dist, df_confusion, df_tuning, df_cv_folds, df_grid,
                df_restaurant_preds, df_rest_vs_rev_f1)
    try:
        (ml_summary, df_model_comparison, df_balancing, df_final_comparison,
         df_before_after, df_pred_dist, df_confusion, df_tuning,
         df_cv_folds, df_grid,
         df_restaurant_preds, df_rest_vs_rev_f1) = load_ml_data()
        ml_loaded = True
    except FileNotFoundError:
        st.warning(" ML output files not found. Run `cuisine_classifier.py` first.")
        ml_loaded = False

    if ml_loaded:

        # ── SECTION 1: DATA OVERVIEW ──────────────────────────────────
        st.subheader(":material/inventory_2: Data Overview")

        # Calculate CURRENT counts from actual data
        current_total = len(df_reviews)
        current_known = ml_summary['known_cuisine_reviews'] 
        current_unknown = ml_summary['unknown_cuisine_reviews'] 

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", f"{current_total:,}")  
        col2.metric("Known Cuisine (Train)", f"{current_known:,}")
        col3.metric("Unknown Cuisine", f"{current_unknown:,}")
        col4.metric("Reviews Recovered", f"{ml_summary['predictions_made']:,}")

        st.caption(f" Current dataset: {current_total:,} reviews | ML model trained on: {ml_summary['total_reviews_loaded']:,} reviews")

        # ── SECTION 2: PHASE 1 — MODEL COMPARISON (NO BALANCING) ─────
        st.subheader(":material/emoji_events: Phase 1 Model Comparison (No Balancing)")
        st.caption(
            "All three models trained on an 80/20 split with no balancing. Best model selected by weighted F1. "
            "Note: Phase 1 rewards models that exploit class imbalance, the winner here is not necessarily "
            "the winner after balancing is applied."
            )

        col_left, col_right = st.columns([1, 1])

        with col_left:
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                name='Accuracy',
                x=df_model_comparison['model'],
                y=(df_model_comparison['test_accuracy'] * 100).round(1),
                marker_color='#3498db',
                text=(df_model_comparison['test_accuracy'] * 100).round(1).astype(str) + '%',
                textposition='outside'
            ))
            fig_compare.add_trace(go.Bar(
                name='Weighted F1',
                x=df_model_comparison['model'],
                y=(df_model_comparison['weighted_f1'] * 100).round(1),
                marker_color='#2ecc71',
                text=(df_model_comparison['weighted_f1'] * 100).round(1).astype(str) + '%',
                textposition='outside'
            ))
            fig_compare.update_layout(
                barmode='group',
                title='Accuracy vs Weighted F1 — No Balancing',
                yaxis=dict(title='Score (%)', range=[0, 100]),
                xaxis_title='Model',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                height=400
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        with col_right:
            display_p1 = df_model_comparison[['model', 'test_accuracy', 'weighted_f1', 'weighted_precision', 'weighted_recall', 'overfit_gap', 'is_best']].copy()
            display_p1['test_accuracy'] = (display_p1['test_accuracy'] * 100).round(1).astype(str) + '%'
            display_p1['weighted_f1'] = (display_p1['weighted_f1'] * 100).round(1).astype(str) + '%'
            display_p1['weighted_precision'] = (display_p1['weighted_precision'] * 100).round(1).astype(str) + '%'
            display_p1['weighted_recall'] = (display_p1['weighted_recall'] * 100).round(1).astype(str) + '%'
            display_p1['overfit_gap'] = display_p1['overfit_gap'].apply(lambda x: f"{x:+.3f}")
            best_mask = display_p1['is_best']
            display_p1 = display_p1.drop(columns=['is_best'])
            display_p1.columns = ['Model', 'Accuracy', 'Weighted F1', 'Precision', 'Recall', 'Overfit Gap']
            styled_p1 = display_p1.style.apply(
                lambda row: ['background-color: #1a3a1a; font-weight: bold; color: #2ecc71' if best_mask.iloc[row.name] else '' for _ in row],
                axis=1
            )
            st.dataframe(styled_p1, use_container_width=True, hide_index=True)
            st.caption("**Overfit Gap** = train accuracy − test accuracy. Values near 0 mean the model generalises well; large positive values mean it memorised training data.")
            st.info(f"**Phase 1 winner: {ml_summary['phase1_best_model']}** — carried forward to balancing comparison.")

        st.write("---")

        # ── SECTION 3: PHASE 2 — BALANCING STRATEGY COMPARISON ───────
        st.subheader(":material/balance: Phase 2 Balancing Strategy Comparison")
        st.caption(f"The Phase 1 winner ({ml_summary['phase1_best_model']}) was tested with 2 balancing strategies using 5-Fold Stratified Cross-Validation.")

        col_bal_l, col_bal_r = st.columns([1, 1])

        with col_bal_l:
            fig_bal = go.Figure()
            fig_bal.add_trace(go.Bar(
                name='Mean F1',
                x=df_balancing['strategy'],
                y=(df_balancing['mean_f1'] * 100).round(2),
                marker_color=['#2ecc71' if b else '#3498db' for b in df_balancing['is_best']],
                error_y=dict(type='data', array=(df_balancing['std_f1'] * 100).round(2), visible=True),
                text=(df_balancing['mean_f1'] * 100).round(2).astype(str) + '%',
                textposition='outside'
            ))
            fig_bal.update_layout(
                title='Mean Weighted F1 by Balancing Strategy (5-Fold CV)',
                yaxis=dict(title='Mean F1 (%)', range=[0, 100]),
                xaxis_title='Strategy',
                height=400
            )
            st.plotly_chart(fig_bal, use_container_width=True)

        with col_bal_r:
            display_bal = df_balancing.copy()
            best_mask_bal = display_bal['is_best']
            display_bal['mean_f1'] = (display_bal['mean_f1'] * 100).round(2).astype(str) + '%'
            display_bal['std_f1']  = (display_bal['std_f1'] * 100).round(2).astype(str) + '%'
            display_bal = display_bal.drop(columns=['is_best'])
            display_bal.columns = ['Strategy', 'Mean F1', 'Std F1']
            styled_bal = display_bal.style.apply(
                lambda row: ['background-color: #1a3a1a; font-weight: bold; color: #2ecc71' if best_mask_bal.iloc[row.name] else '' for _ in row],
                axis=1
            )
            st.dataframe(styled_bal, use_container_width=True, hide_index=True)

            st.success(f"**Best strategy: {ml_summary['phase2_best_strategy']}** — applied to all 3 models in Phase 3.")

            st.markdown("**What each strategy does:**")
            st.markdown(
                "- **Baseline** — no balancing, model is biased toward Lebanese (dominant class)\n"
                "- **Class Weights** — penalises mistakes on minority classes more during training\n"
                "  *(Note: LightGBM uses its own `class_weight` parameter — it handles this natively)*\n"
            )

        # Per-fold breakdown — the actual StratifiedGroupKFold output
        st.markdown("##### Per-Fold Cross-Validation Scores")
        st.caption(
            "Each strategy is evaluated on 5 different train/test splits (folds). StratifiedKFold "
            "preserves the class distribution in each fold. Because each row is already one restaurant, "
            "there's no risk of review-level leakage. Tight clusters mean the strategy is stable; wide "
            "spreads mean it's sensitive to which restaurants end up where."
        )

        col_cv_l, col_cv_r = st.columns([1.4, 1])

        with col_cv_l:
            # Box plot with individual fold points overlaid
            fig_cv = go.Figure()
            for strategy in df_cv_folds['strategy'].unique():
                strategy_scores = df_cv_folds[df_cv_folds['strategy'] == strategy]['f1_score']
                fig_cv.add_trace(go.Box(
                    y=strategy_scores,
                    name=strategy,
                    boxpoints='all', # show every fold as a dot
                    jitter=0.3,
                    pointpos=0,
                    marker=dict(size=10, opacity=0.8),
                    line=dict(width=2),
                ))
            fig_cv.update_layout(
                title='Distribution of F1 Scores Across 5 Folds',
                yaxis=dict(title='Weighted F1', range=[0, max(df_cv_folds['f1_score']) * 1.15]),
                xaxis_title='Balancing Strategy',
                showlegend=False,
                height=420)
            st.plotly_chart(fig_cv, use_container_width=True)

        with col_cv_r:
            st.markdown("**Raw fold scores**")
            pivot = df_cv_folds.pivot(index='fold', columns='strategy', values='f1_score')
            pivot.index = [f"Fold {i}" for i in pivot.index]
            pivot_display = pivot.map(lambda x: f"{x:.4f}")
            st.dataframe(pivot_display, use_container_width=True)

            # Quick reading
            best_strat = ml_summary['phase2_best_strategy']
            best_mean = ml_summary['phase2_balancing_strategies'][best_strat]['mean_f1']
            best_std = ml_summary['phase2_balancing_strategies'][best_strat]['std_f1']
            st.info(
                f"**{best_strat}** had the best mean F1 ({best_mean:.4f}) "
                f"with a std of {best_std:.4f} across the 5 folds — "
                f"this is the variance you'd expect on new unseen restaurants."
            )

        with st.expander("What is StratifiedKFold and why 5 folds?"):
            st.markdown(
                "**Cross-validation** trains and tests a model on several different train/test splits "
                "so you're not trusting a single lucky or unlucky split. You get N scores and take their mean.\n\n"
                "**Stratified** means each fold preserves the same class distribution as the full dataset — "
                "if 35% of restaurants are Levantine, each fold is also ~35% Levantine. Without stratification, "
                "a rare class like Jewish (38 restaurants) could land entirely in one fold and be untestable in the others.\n\n"
                "We don't need **grouped** splits here because our training unit is already the restaurant — "
                "each row in X is one restaurant's concatenated reviews. Review-level leakage across folds "
                "is impossible by construction.\n\n"
                "**5 folds** is the standard tradeoff — more folds give tighter mean estimates but take "
                "5× longer to run. With 5 folds, each restaurant is used for training 4 times and testing once."
    )

        st.write("---")

        # SECTION 4: PHASE 3 FINAL MODEL COMPARISON (WITH BALANCING) 
        st.subheader(":material/military_tech: Phase 3 Final Model Comparison (With Balancing)")
        st.caption(f"All 3 models retrained using the best balancing strategy: **{ml_summary['phase2_best_strategy']}**")

        col_fin_l, col_fin_r = st.columns([1, 1])

        with col_fin_l:
            fig_final = go.Figure()
            fig_final.add_trace(go.Bar(
                name='Accuracy',
                x=df_final_comparison['model'],
                y=(df_final_comparison['test_accuracy'] * 100).round(1),
                marker_color='#3498db',
                text=(df_final_comparison['test_accuracy'] * 100).round(1).astype(str) + '%',
                textposition='outside'
            ))
            fig_final.add_trace(go.Bar(
                name='Weighted F1',
                x=df_final_comparison['model'],
                y=(df_final_comparison['weighted_f1'] * 100).round(1),
                marker_color='#2ecc71',
                text=(df_final_comparison['weighted_f1'] * 100).round(1).astype(str) + '%',
                textposition='outside'))
            fig_final.update_layout(
                barmode='group',
                title=f'Accuracy vs Weighted F1 — With {ml_summary["phase2_best_strategy"]}',
                yaxis=dict(title='Score (%)', range=[0, 100]),
                xaxis_title='Model',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                height=400)
            st.plotly_chart(fig_final, use_container_width=True)

        with col_fin_r:
            display_fin = df_final_comparison[['model', 'test_accuracy', 'weighted_f1', 'weighted_precision', 'weighted_recall', 'overfit_gap', 'is_best']].copy()
            best_mask_fin = display_fin['is_best']
            display_fin['test_accuracy'] = (display_fin['test_accuracy'] * 100).round(1).astype(str) + '%'
            display_fin['weighted_f1'] = (display_fin['weighted_f1'] * 100).round(1).astype(str) + '%'
            display_fin['weighted_precision'] = (display_fin['weighted_precision'] * 100).round(1).astype(str) + '%'
            display_fin['weighted_recall'] = (display_fin['weighted_recall'] * 100).round(1).astype(str) + '%'
            display_fin['overfit_gap'] = display_fin['overfit_gap'].apply(lambda x: f"{x:+.3f}")
            display_fin = display_fin.drop(columns=['is_best'])
            display_fin.columns = ['Model', 'Accuracy', 'Weighted F1', 'Precision', 'Recall', 'Overfit Gap']
            styled_fin = display_fin.style.apply(
                lambda row: ['background-color: #1a3a1a; font-weight: bold; color: #2ecc71' if best_mask_fin.iloc[row.name] else '' for _ in row],
                axis=1)
            st.dataframe(styled_fin, use_container_width=True, hide_index=True)
            st.success(f"**Final best model: {ml_summary['phase3_best_model']}** — used for all predictions.")

        st.write("---")
        
        

        # ── SECTION 5: PHASE 4 — BEFORE vs AFTER 
        st.subheader(":material/compare_arrows: Phase 4 Before vs After Balancing")
        st.caption(f"Same model ({ml_summary['before_after']['model']}), with and without the best balancing strategy.")

        # Per-class before vs after bar chart
        df_ba_sorted = df_before_after.sort_values('f1_change', ascending=True)

        fig_ba = go.Figure()
        fig_ba.add_trace(go.Bar(
            name='Before',
            y=df_ba_sorted['cuisine'],
            x=df_ba_sorted['f1_before'],
            orientation='h',
            marker_color='#e74c3c'
        ))
        fig_ba.add_trace(go.Bar(
            name='After',
            y=df_ba_sorted['cuisine'],
            x=df_ba_sorted['f1_after'],
            orientation='h',
            marker_color='#2ecc71'
        ))
        fig_ba.update_layout(
            barmode='group',
            title='Per-Cuisine F1 Score: Before vs After Balancing',
            xaxis=dict(title='F1 Score', range=[0, 1]),
            yaxis_title='Cuisine',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=650
        )
        st.plotly_chart(fig_ba, use_container_width=True)

        # Most improved cuisines table
        col_imp_l, col_imp_r = st.columns([1, 1])
        with col_imp_l:
            st.markdown("**Most Improved Cuisines**")
            most_improved = df_before_after.sort_values('f1_change', ascending=False).head(8)
            display_imp = most_improved[['cuisine', 'f1_before', 'f1_after', 'f1_change']].copy()
            display_imp.columns = ['Cuisine', 'F1 Before', 'F1 After', 'Change']
            display_imp['Change'] = display_imp['Change'].apply(lambda x: f"{x:+.3f}")
            st.dataframe(display_imp, use_container_width=True, hide_index=True)

        with col_imp_r:
            st.markdown("**Least Improved / Declined**")
            least_improved = df_before_after.sort_values('f1_change', ascending=True).head(8)
            display_least = least_improved[['cuisine', 'f1_before', 'f1_after', 'f1_change']].copy()
            display_least.columns = ['Cuisine', 'F1 Before', 'F1 After', 'Change']
            display_least['Change'] = display_least['Change'].apply(lambda x: f"{x:+.3f}")
            st.dataframe(display_least, use_container_width=True, hide_index=True)

        st.write("---")
        
        # BEFORE vs AFTER TUNING ──────────────
        st.subheader(":material/tune: Before vs After Hyperparameter Tuning")
        st.caption(
            f"Same model ({ml_summary['tuning_before_after']['model']}), "
            "with default hyperparameters vs tuned hyperparameters from GridSearchCV (3-fold grouped CV)."
        )

        tba = ml_summary['tuning_before_after']

        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        col_t1.metric("Accuracy Untuned", f"{tba['accuracy_untuned']*100:.1f}%")
        col_t2.metric("Accuracy Tuned",   f"{tba['accuracy_tuned']*100:.1f}%",
                      delta=f"{(tba['accuracy_tuned']-tba['accuracy_untuned'])*100:+.1f}%")
        col_t3.metric("F1 Untuned",       f"{tba['f1_untuned']:.4f}")
        col_t4.metric("F1 Tuned",         f"{tba['f1_tuned']:.4f}",
                      delta=f"{tba['f1_tuned']-tba['f1_untuned']:+.4f}")

        # Overfit-gap delta — tuning can change how much the model overfits
        col_o1, col_o2, col_o3 = st.columns([1, 1, 2])
        col_o1.metric("Overfit Gap Untuned", f"{tba['overfit_gap_untuned']:+.3f}")
        col_o2.metric("Overfit Gap Tuned",   f"{tba['overfit_gap_tuned']:+.3f}",
                      delta=f"{tba['overfit_gap_tuned']-tba['overfit_gap_untuned']:+.3f}",
                      delta_color="inverse")
        with col_o3:
            st.markdown("**Winning Hyperparameters**")
            def clean_param_name(k):
                k = k.replace('tfidf__word__', 'tfidf__').replace('tfidf__char__', 'char_tfidf__')
                return k

            params_df = pd.DataFrame(
                [(clean_param_name(k), v) for k, v in tba['best_params'].items()],
                columns=['Hyperparameter', 'Value']
            )
            st.dataframe(params_df, use_container_width=True, hide_index=True)

        if tba['tuning_improved']:
            st.success(
                f"Tuning improved held-out F1 by {tba['f1_tuned']-tba['f1_untuned']:+.4f}. "
                "The tuned model was kept as final."
            )
        else:
            st.warning(
                "⚠️ Tuning did not improve held-out F1. The untuned model was kept as final. "
                "This happens when the grid's CV winner doesn't generalize to the test set."
            )

        # Per-class before vs after tuning chart
        df_t_sorted = df_tuning.sort_values('f1_change', ascending=True)

        fig_tuning = go.Figure()
        fig_tuning.add_trace(go.Bar(
            name='Untuned',
            y=df_t_sorted['cuisine'],
            x=df_t_sorted['f1_untuned'],
            orientation='h',
            marker_color='#e74c3c'
        ))
        fig_tuning.add_trace(go.Bar(
            name='Tuned',
            y=df_t_sorted['cuisine'],
            x=df_t_sorted['f1_tuned'],
            orientation='h',
            marker_color='#2ecc71'
        ))
        fig_tuning.update_layout(
            barmode='group',
            title='Per-Cuisine F1 Score: Untuned vs Tuned Hyperparameters',
            xaxis=dict(title='F1 Score', range=[0, 1]),
            yaxis_title='Cuisine',
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=650
        )
        st.plotly_chart(fig_tuning, use_container_width=True)

        col_tu_l, col_tu_r = st.columns([1, 1])
        with col_tu_l:
            st.markdown("**Cuisines Helped Most by Tuning**")
            most_helped = df_tuning.sort_values('f1_change', ascending=False).head(8)
            d_helped = most_helped[['cuisine', 'f1_untuned', 'f1_tuned', 'f1_change']].copy()
            d_helped.columns = ['Cuisine', 'F1 Untuned', 'F1 Tuned', 'Change']
            d_helped['Change'] = d_helped['Change'].apply(lambda x: f"{x:+.3f}")
            st.dataframe(d_helped, use_container_width=True, hide_index=True)
        with col_tu_r:
            st.markdown("**Cuisines Hurt Most by Tuning**")
            most_hurt = df_tuning.sort_values('f1_change', ascending=True).head(8)
            d_hurt = most_hurt[['cuisine', 'f1_untuned', 'f1_tuned', 'f1_change']].copy()
            d_hurt.columns = ['Cuisine', 'F1 Untuned', 'F1 Tuned', 'Change']
            d_hurt['Change'] = d_hurt['Change'].apply(lambda x: f"{x:+.3f}")
            st.dataframe(d_hurt, use_container_width=True, hide_index=True)
            
        # Show the full grid search CV distribution
        st.markdown("##### Hyperparameter Search — All Combinations")
        st.caption(
            f"GridSearchCV tried {len(df_grid)} hyperparameter combinations, each scored by 3-fold "
            "StratifiedKFold cross-validation."
            "The chosen combination is the one with the highest mean CV F1."
        )

        fold_score_cols = [c for c in df_grid.columns if c.startswith('split') and c.endswith('_test_score')]
        df_grid_plot = df_grid.copy()
        df_grid_plot['combo_rank'] = range(1, len(df_grid_plot) + 1)

        fig_grid = go.Figure()
        # Error bars: min-to-max across folds
        fold_values = df_grid_plot[fold_score_cols].values
        fig_grid.add_trace(go.Scatter(
            x=df_grid_plot['combo_rank'],
            y=df_grid_plot['mean_test_score'],
            mode='markers',
            marker=dict(
                size=10,
                color=df_grid_plot['mean_test_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Mean F1')
            ),
            error_y=dict(
                type='data',
                symmetric=False,
                array=fold_values.max(axis=1) - df_grid_plot['mean_test_score'],
                arrayminus=df_grid_plot['mean_test_score'] - fold_values.min(axis=1),
                thickness=1,
                width=4,
            ),
            text=df_grid_plot['params'],
            hovertemplate='<b>Rank %{x}</b><br>Mean F1: %{y:.4f}<br>%{text}<extra></extra>',
            name='CV Mean ± range'
        ))
        fig_grid.update_layout(
            title='GridSearchCV: Mean CV F1 per Combination (ranked best to worst)',
            xaxis_title='Combination Rank',
            yaxis_title='Mean CV F1',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_grid, use_container_width=True)

        st.caption(
            "Each dot is one hyperparameter combination; the error bar spans the min–max F1 across "
            "the 3 folds. Combinations where the bar is tall are unstable (performance depends on "
            "which restaurants were held out). The leftmost dot is what the tuner picked."
        )

        st.write("---")
        
        # SECTION 4.5: PIPELINE IMPROVEMENT BREAKDOWN 
        st.subheader(":material/timeline: Pipeline Improvement Breakdown")

        phase1_f1 = ml_summary['phase1_models'][ml_summary['phase1_best_model']]['weighted_f1']
        phase1_acc = ml_summary['phase1_models'][ml_summary['phase1_best_model']]['accuracy']
        tba = ml_summary['tuning_before_after']
        untuned_f1 = tba['f1_untuned']
        untuned_acc = tba['accuracy_untuned']
        tuned_f1 = tba['f1_tuned']
        tuned_acc = tba['accuracy_tuned']

        balancing_delta_f1 = untuned_f1 - phase1_f1
        balancing_delta_acc = untuned_acc - phase1_acc
        tuning_delta_f1 = tuned_f1 - untuned_f1
        tuning_delta_acc = tuned_acc - untuned_acc

        phase1_model = ml_summary['phase1_best_model']
        phase3_model = ml_summary['phase3_best_model']
        model_swapped = phase1_model != phase3_model

        steps_df = pd.DataFrame([
        {
            'Step': '1. Phase 1 baseline',
            'Model': phase1_model,
            'Description': 'Default hyperparameters, no balancing. Rewards whichever model best exploits class imbalance.',
            'Accuracy': f"{phase1_acc*100:.1f}%",
            'Weighted F1': f"{phase1_f1:.4f}",
            'Δ F1': '—',
        },
        {
            'Step': '2. + Balancing',
            'Model': phase3_model,
            'Description': f"Best strategy from Phase 2: {ml_summary['phase2_best_strategy']}. "
                        f"{'Model changed — RF could not capitalize on class weights the way LogReg does.' if model_swapped else ''}",
            'Accuracy': f"{untuned_acc*100:.1f}%",
            'Weighted F1': f"{untuned_f1:.4f}",
            'Δ F1': f"{balancing_delta_f1:+.4f}",
        },
        {
            'Step': '3. + Hyperparameter Tuning',
            'Model': phase3_model,
            'Description': f"GridSearchCV picked: {', '.join(f'{k.replace('tfidf__word__', 'tfidf__')}={v}' for k,v in tba['best_params'].items())}",
            'Accuracy': f"{tuned_acc*100:.1f}%",
            'Weighted F1': f"{tuned_f1:.4f}",
            'Δ F1': f"{tuning_delta_f1:+.4f}",
        },
    ])

        st.dataframe(steps_df, use_container_width=True, hide_index=True)

        # Waterfall-style chart
        fig_waterfall = go.Figure()
        fig_waterfall.add_trace(go.Bar(
            x=[s for s in steps_df['Step']],
            y=[phase1_f1, untuned_f1, tuned_f1],
            marker_color=['#95a5a6', '#3498db', '#27ae60'],
            text=[f'{phase1_f1:.4f}', f'{untuned_f1:.4f}', f'{tuned_f1:.4f}'],
            textposition='outside',
        ))
        deltas = [None, balancing_delta_f1, tuning_delta_f1]
        for i, delta in enumerate(deltas):
            if delta is not None:
                color = '#27ae60' if delta > 0.001 else ('#95a5a6' if abs(delta) < 0.001 else '#e74c3c')
                fig_waterfall.add_annotation(
                    x=i, y=max(phase1_f1, untuned_f1, tuned_f1) * 1.08,
                    text=f"{delta:+.4f}",
                    showarrow=False,
                    font=dict(color=color, size=14),
                )
        fig_waterfall.update_layout(
            title='Weighted F1 at Each Pipeline Step',
            yaxis=dict(title='Weighted F1', range=[0, max(phase1_f1, untuned_f1, tuned_f1) * 1.2]),
            height=450,
            showlegend=False,
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Callouts
        col_w1, col_w2 = st.columns(2)

        with col_w1:
            if balancing_delta_f1 > 0.05:
                st.success(
                    f"**Balancing: +{balancing_delta_f1:.4f} F1**\n\n"
                    f"Applying *{ml_summary['phase2_best_strategy']}* to the training loss pushed the model "
                    "away from collapsing everything into the dominant class. "
                    f"{'This also changed which model won — ' + phase1_model + ' could exploit imbalance but ' + phase3_model + ' benefits more from correcting it.' if model_swapped else ''}"
                )
            elif balancing_delta_f1 > 0:
                st.info(f"**Balancing: +{balancing_delta_f1:.4f} F1**\n\nModest improvement from {ml_summary['phase2_best_strategy']}.")
            else:
                st.warning(f"**Balancing: {balancing_delta_f1:+.4f} F1**\n\nBalancing did not help on this dataset.")

        with col_w2:
            if tuning_delta_f1 > 0:
                st.success(
                    f"**Tuning: +{tuning_delta_f1:.4f} F1**\n\n"
                    "GridSearchCV found hyperparameters that modestly outperform sklearn's defaults. "
                    "The gain is smaller than balancing because the defaults were already reasonable."
                )
            else:
                st.warning(f"**Tuning: {tuning_delta_f1:+.4f} F1**\n\nTuning did not improve held-out performance.")

        with st.expander("Why this ordering matters"):
            st.markdown(
                "The pipeline improvements are **cumulative** — each step builds on the previous one. "
                "This matters because it lets us attribute gains honestly:\n\n"
                f"- **Baseline (Phase 1)** selects the model that does best *before* we address class imbalance. "
                f"Here that was **{phase1_model}**, which overfit hard (train accuracy near 1.0) but got decent F1 "
                "by predicting the dominant class often.\n"
                f"- **Balancing** forces the model to actually distinguish minority cuisines. "
                f"{'This step changed the winning model — ' + phase3_model + ' responds much better to class weights than tree-based models do.' if model_swapped else ''}\n"
                "- **Hyperparameter tuning** fine-tunes the winning model's regularization and vocabulary size.\n\n"
                "Note: **restaurant-level aggregation** isn't a step in this pipeline because it's the *default* — "
                "we train on one document per restaurant (concatenated reviews). This is why per-review "
                "and per-restaurant F1 are the same number in this project."
            )

        # ── SECTION 6: CONFUSION ANALYSIS ────────────────────────────
        st.subheader(":material/shuffle: Where the Model Gets Confused")
        st.caption(ml_summary['confusion_matrix_note'])

        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("**Top Confusion Pairs**")
            st.caption("Cuisines most commonly mispredicted as another class (> 5% bleed rate)")
            pairs = ml_summary['top_confusion_pairs']
            df_pairs = pd.DataFrame(pairs)
            df_pairs.columns = ['Actual', 'Predicted As', 'Rate']
            df_pairs['Rate'] = (df_pairs['Rate'] * 100).round(1).astype(str) + '%'
            st.dataframe(df_pairs, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("**Why This Happens**")
            st.write(
                "The confusion pattern reflects genuine cultural and culinary overlap "
                "in the Lebanese dining context — Armenian, Mediterranean, and Middle Eastern "
                "restaurants in Beirut often serve similar dishes and receive nearly identical "
                "review vocabulary. This is a real-world ambiguity, not purely a model weakness."
            )

        st.write("")
        st.markdown("**Full Confusion Matrix (row-normalized)**")
        st.caption(
            "Rows are the actual cuisine; columns are what the model predicted. "
            "Each row sums to 1 — a cell shows the probability of predicting the column cuisine "
            "given the row's true cuisine. A perfect model would have 1.0 on the diagonal and 0 elsewhere."
        )

        fig_cm = px.imshow(
            df_confusion.values,
            x=df_confusion.columns,
            y=df_confusion.index,
            color_continuous_scale='Blues',
            aspect='auto',
            labels=dict(x='Predicted', y='Actual', color='Rate'),
            text_auto='.2f',
            zmin=0,
            zmax=1,
        )
        fig_cm.update_layout(
            title='Confusion Matrix — Normalized by Actual Class',
            height=600,
            xaxis=dict(tickangle=-45),
        )
        fig_cm.update_xaxes(side='bottom')
        st.plotly_chart(fig_cm, use_container_width=True)

        st.caption(
            "**Reading the matrix:** bright diagonal cells = correct predictions. "
            "Bright off-diagonal cells = systematic confusion. Notice how most errors "
            "bleed rightward into the Levantine column — minority cuisines get pulled "
            "toward the dominant class when the model is uncertain."
        )

        st.write("---")

        st.write("---")

        # ── SECTION 7: PER-CLASS F1 BREAKDOWN (FINAL MODEL) ──────────
        st.subheader(":material/bar_chart: Per-Class F1 Score Breakdown (Final Model)")
        st.caption(f"Performance of {ml_summary['phase3_best_model']} with {ml_summary['phase2_best_strategy']} applied.")

        per_class = ml_summary['per_class_f1']
        df_f1 = pd.DataFrame(
            list(per_class.items()), columns=['Cuisine', 'F1 Score']
        ).sort_values('F1 Score', ascending=True)

        colors = ['#e74c3c' if v == 0 else '#f39c12' if v < 0.3 else '#2ecc71'
                  for v in df_f1['F1 Score']]

        fig_f1 = go.Figure(go.Bar(
            x=df_f1['F1 Score'],
            y=df_f1['Cuisine'],
            orientation='h',
            marker_color=colors,
            text=df_f1['F1 Score'].round(3),
            textposition='outside'
        ))
        fig_f1.update_layout(
            title=f'F1 Score per Cuisine — {ml_summary["phase3_best_model"]} + {ml_summary["phase2_best_strategy"]}',
            xaxis=dict(title='F1 Score', range=[0, 1]),
            yaxis_title='Cuisine',
            height=650
        )
        st.plotly_chart(fig_f1, use_container_width=True)

        st.write("---")


        # SECTION 7.5: RESTAURANT-LEVEL EVALUATION
        st.subheader(":material/store: Restaurant-Level Evaluation")

        rl = ml_summary['restaurant_level']

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Test F1 (Restaurant-Level)", f"{rl['f1_restaurant_level']:.4f}")
        col_r2.metric("Test Accuracy", f"{rl['accuracy_restaurant_level']:.4f}")
        col_r3.metric("Restaurants Predicted", f"{rl['n_restaurants_predicted']:,}")
        col_r4.metric("High-Confidence (≥0.5)", f"{rl['high_confidence_pct']}%")


        # ── SECTION 8: PREDICTION OUTCOMES ───────────────────────────

        st.subheader(":material/auto_awesome: Prediction Outcomes on Unknown Reviews")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Avg Confidence Score",       f"{ml_summary['avg_prediction_confidence']:.2f}")
        col_m2.metric("Low-Confidence Predictions",
                      f"{ml_summary['low_confidence_count']:,}",
                      f"{ml_summary['low_confidence_pct']}% of total",
                      delta_color="inverse")
        col_m3.metric("Confidence Threshold",       f"{ml_summary['low_confidence_threshold']}")

        col_pred_l, col_pred_r = st.columns([1.2, 1])

        with col_pred_l:
            fig_dist = px.bar(
                df_pred_dist,
                x='predicted_count',
                y='cuisine',
                orientation='h',
                color='avg_confidence',
                color_continuous_scale='RdYlGn',
                range_color=[0, 1],
                labels={
                    'predicted_count': 'Restaurants Predicted',
                    'cuisine': 'Cuisine',
                    'avg_confidence': 'Avg Confidence'
                },
                title='Predicted Cuisine Distribution (Recovered Restaurants)')
            fig_dist.update_layout(height=550, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_pred_r:
            st.markdown("**Recovered Restaurants by Cuisine**")
            display_pred = df_pred_dist[['cuisine', 'predicted_count', 'avg_confidence', 'high_conf_count']].copy()
            display_pred.columns = ['Cuisine', 'Restaurants Predicted', 'Avg Conf', 'High Conf (≥0.5)']
            display_pred['Avg Conf'] = display_pred['Avg Conf'].round(3)
            st.dataframe(display_pred, use_container_width=True, hide_index=True, height=500)

        st.write("---")

        # ── SECTION 9: ENRICHED DATASET SUMMARY 
        st.subheader(":material/check_circle: Enriched Dataset")
        st.write("The classifier output was merged back into `master_reviews_enriched.csv`. A `cuisine_source` column tracks which labels are original vs ML-predicted.")

        col_e1, col_e2, col_e3 = st.columns(3)
        col_e1.metric("Total Reviews (Enriched)", f"{ml_summary['enriched_reviews_total']:,}")
        col_e2.metric("Original Labels",           f"{ml_summary['enriched_original_labels']:,}")
        col_e3.metric("ML-Predicted Labels",       f"{ml_summary['enriched_predicted_labels']:,}")

        fig_donut = go.Figure(go.Pie(
            labels=['Original Metadata', 'ML-Predicted'],
            values=[ml_summary['enriched_original_labels'], ml_summary['enriched_predicted_labels']],
            hole=0.55,
            marker_colors=['#2ecc71', '#3498db']
        ))
        fig_donut.update_layout(title='Label Source Breakdown in Enriched Dataset', height=350)
        st.plotly_chart(fig_donut, use_container_width=True)

        st.caption(
            "ML-predicted labels are **restaurant-level** — all of a restaurant's reviews are "
            "concatenated into one document and classified once, so every review of the same "
            "restaurant inherits the same predicted cuisine. Low-confidence predictions (< 0.30) "
            "are still included but flagged via the `prediction_confidence` column.")
        st.write("---")
        
        # SECTION 9.5: ENRICHED RESTAURANTS TABLE
        st.subheader(":material/restaurant: Enriched Restaurants Table")
        st.write("One row per restaurant — cleaner for filtering by cuisine/area at the restaurant level instead of the review level.")

        @st.cache_data
        def load_enriched_restaurants():
            return pd.read_csv(os.path.join(ML_DIR, 'master_restaurants_enriched.csv'))

        try:
            df_rest_enr = load_enriched_restaurants()

            col_rf1, col_rf2, col_rf3, col_rf4 = st.columns(4)
            with col_rf1:
                rsource_filter = st.radio(
                    "Label Source:",
                    options=["All", "ML-Predicted Only", "Original Only"],
                    horizontal=True,
                    key="rest_source_filter"
                )
            with col_rf2:
                rcuisine_opts = ["All"] + sorted(df_rest_enr['cuisine_primary'].dropna().unique().tolist())
                rcuisine_filter = st.selectbox("Cuisine:", rcuisine_opts, key="rest_cuisine_filter")
            with col_rf3:
                rarea_opts = ["All"] + sorted(df_rest_enr['area'].dropna().unique().tolist())
                rarea_filter = st.selectbox("Area:", rarea_opts, key="rest_area_filter")
            with col_rf4:
                rmin_conf = st.slider("Min Confidence (ML rows only):", 0.0, 1.0, 0.0, 0.05, key="rest_conf_slider")

            rf = df_rest_enr.copy()
            if rsource_filter == "ML-Predicted Only":
                rf = rf[rf['cuisine_source'] == 'predicted']
            elif rsource_filter == "Original Only":
                rf = rf[rf['cuisine_source'] == 'original']
            if rcuisine_filter != "All":
                rf = rf[rf['cuisine_primary'] == rcuisine_filter]
            if rarea_filter != "All":
                rf = rf[rf['area'] == rarea_filter]
            if rmin_conf > 0.0:
                ml_mask = rf['cuisine_source'] == 'predicted'
                orig_mask = rf['cuisine_source'] == 'original'
                rf = pd.concat([
                    rf[ml_mask & (rf['prediction_confidence'] >= rmin_conf)],
                    rf[orig_mask]
                ]).sort_index()

            col_rm1, col_rm2, col_rm3, col_rm4 = st.columns(4)
            col_rm1.metric("Showing",         f"{len(rf):,} restaurants")
            col_rm2.metric("ML-Predicted",    f"{(rf['cuisine_source'] == 'predicted').sum():,}")
            col_rm3.metric("Original Labels", f"{(rf['cuisine_source'] == 'original').sum():,}")
            ml_conf = rf[rf['cuisine_source'] == 'predicted']['prediction_confidence'].mean()
            col_rm4.metric("Avg Confidence (ML)", f"{ml_conf:.3f}" if not pd.isna(ml_conf) else "—")

            st.dataframe(
                rf.reset_index(drop=True),
                use_container_width=True,
                height=500,
                column_config={
                    'prediction_confidence': st.column_config.ProgressColumn(
                        'Confidence', min_value=0.0, max_value=1.0, format="%.3f"
                    )
                }
            )
        except FileNotFoundError:
            st.warning("⚠️ `master_restaurants_enriched.csv` not found. Re-run `cuisine_classifier.py`.")

        st.write("---")

        # ── SECTION 10: ENRICHED REVIEWS TABLE ───────────────────────
        st.subheader(":material/table_rows: Enriched Reviews Dataset")
        st.write("Browse the full enriched reviews file. Filter to inspect only the rows where cuisine was filled in by the ML classifier.")

        @st.cache_data
        def load_enriched_reviews():
            return pd.read_csv(os.path.join(ML_DIR, 'master_reviews_enriched.csv'))

        try:
            df_enriched = load_enriched_reviews()

            col_f1, col_f2, col_f3, col_f4 = st.columns(4)

            with col_f1:
                source_filter = st.radio(
                    "Label Source:",
                    options=["All", "ML-Predicted Only", "Original Only"],
                    horizontal=True
                )
            with col_f2:
                cuisine_opts = ["All"] + sorted(df_enriched['cuisine_primary'].dropna().unique().tolist())
                cuisine_filter = st.selectbox("Cuisine:", cuisine_opts, key="enr_table_cuisine")
            with col_f3:
                area_opts = ["All"] + sorted(df_enriched['area'].dropna().unique().tolist())
                area_filter = st.selectbox("Area:", area_opts, key="enr_table_area")
            with col_f4:
                if 'prediction_confidence' in df_enriched.columns:
                    min_conf = st.slider("Min Confidence (ML rows only):", 0.0, 1.0, 0.0, 0.05)
                else:
                    min_conf = 0.0

            filtered = df_enriched.copy()

            if source_filter == "ML-Predicted Only":
                filtered = filtered[filtered['cuisine_source'] == 'predicted']
            elif source_filter == "Original Only":
                filtered = filtered[filtered['cuisine_source'] == 'original']

            if cuisine_filter != "All":
                filtered = filtered[filtered['cuisine_primary'] == cuisine_filter]
            if area_filter != "All":
                filtered = filtered[filtered['area'] == area_filter]
            if min_conf > 0.0:
                ml_mask   = filtered['cuisine_source'] == 'predicted'
                orig_mask = filtered['cuisine_source'] == 'original'
                filtered  = pd.concat([
                    filtered[ml_mask & (filtered['prediction_confidence'] >= min_conf)],
                    filtered[orig_mask]
                ]).sort_index()

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Showing",         f"{len(filtered):,} reviews")
            col_m2.metric("ML-Predicted",    f"{(filtered['cuisine_source'] == 'predicted').sum():,}")
            col_m3.metric("Original Labels", f"{(filtered['cuisine_source'] == 'original').sum():,}")
            if 'prediction_confidence' in filtered.columns:
                avg_conf = filtered[filtered['cuisine_source'] == 'predicted']['prediction_confidence'].mean()
                col_m4.metric("Avg Confidence (ML)", f"{avg_conf:.3f}" if not pd.isna(avg_conf) else "—")

            display_cols = [
                'restaurant_name', 'cuisine_primary', 'cuisine_source',
                'prediction_confidence', 'cuisine_predicted_per_review',
                'area', 'price_category',
                'rating', 'sentiment_category', 'sentiment_score',
                'review_source', 'review_text'
            ]
            display_cols = [c for c in display_cols if c in filtered.columns]

            st.dataframe(
                filtered[display_cols].reset_index(drop=True),
                use_container_width=True,
                height=500,
                column_config={
                    "cuisine_source": st.column_config.TextColumn(
                        "Label Source",
                        help="'predicted' = filled in by ML classifier at restaurant level | 'original' = came from source data"
                    ),
                    "cuisine_primary": st.column_config.TextColumn(
                        "Cuisine (Restaurant-Level)",
                        help="For predicted rows, this is the aggregated restaurant-level label — all reviews of the same restaurant share this label"
                    ),
                    "cuisine_predicted_per_review": st.column_config.TextColumn(
                        "Per-Review Prediction",
                        help="What THIS individual review was predicted as (for transparency). May differ from the restaurant-level label."
                    ),
                    "prediction_confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        help="Restaurant-level confidence (averaged across this restaurant's reviews)",
                        min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "sentiment_score": st.column_config.NumberColumn("Sentiment", format="%.3f"),
                    "review_text": st.column_config.TextColumn("Review Text", width="large")
                }
            )

        except FileNotFoundError:
            st.warning(" `master_reviews_enriched.csv` not found. Run `cuisine_classifier.py` first.")
        
# SECTION 5: NLP ANALYSIS

elif selected_section == "NLP Analysis":
    st.header(":material/forum: NLP Analysis")
    st.write("Sentiment analysis and keyword extraction from restaurant reviews.")
    st.write("---")

    # ── LOAD BOTH NLP DATASETS ──────────────────────────────────────
    @st.cache_data
    def load_nlp_data():
        # Original (pre-ML)
        sa_orig  = pd.read_csv("../nlp/sentiment_by_area.csv")
        sc_orig  = pd.read_csv("../nlp/sentiment_by_cuisine.csv")
        sp_orig  = pd.read_csv("../nlp/sentiment_by_price.csv")
        with open("../nlp/area_keywords.json")    as f: ak_orig = json.load(f)
        with open("../nlp/cuisine_keywords.json") as f: ck_orig = json.load(f)
        try:
            with open("../nlp/nlp_summary.json")  as f: sum_orig = json.load(f)
        except FileNotFoundError:
            sum_orig = None

        # Enriched (post-ML)
        try:
            sa_enr  = pd.read_csv("../nlp/sentiment_by_area_enriched.csv")
            sc_enr  = pd.read_csv("../nlp/sentiment_by_cuisine_enriched.csv")
            sp_enr  = pd.read_csv("../nlp/sentiment_by_price_enriched.csv")
            with open("../nlp/area_keywords_enriched.json")    as f: ak_enr = json.load(f)
            with open("../nlp/cuisine_keywords_enriched.json") as f: ck_enr = json.load(f)
            with open("../nlp/nlp_summary_enriched.json")      as f: sum_enr = json.load(f)
            enriched_loaded = True
        except FileNotFoundError:
            sa_enr = sc_enr = sp_enr = ak_enr = ck_enr = sum_enr = None
            enriched_loaded = False

        return (sa_orig, sc_orig, sp_orig, ak_orig, ck_orig, sum_orig,
                sa_enr,  sc_enr,  sp_enr,  ak_enr,  ck_enr,  sum_enr,
                enriched_loaded)

    (sentiment_area, sentiment_cuisine, sentiment_price, area_keywords, cuisine_keywords, nlp_summary,
     sentiment_area_enr, sentiment_cuisine_enr, sentiment_price_enr, area_keywords_enr, cuisine_keywords_enr, nlp_summary_enr,
     enriched_loaded) = load_nlp_data()

    # TABS 
    tab_before, tab_after = st.tabs(["Before ML — Original Reviews", "After ML — Enriched Reviews"])

    # ── HELPER: renders all NLP charts for a given dataset ──────────
    def render_nlp_tab(s_area, s_cuisine, s_price, a_kw, c_kw, summary, tag):
        """
        tag: short string appended to selectbox keys to avoid duplicate widget IDs
        """
        # SUMMARY STATS
        st.subheader(":material/assignment: Review Dataset Summary")
        if summary:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Reviews Loaded", f"{len(df_reviews):,}")
            col2.metric("Reviews Used in Analysis",   f"{summary['reviews_used']:,}")
            col3.metric("Excluded (Unknown metadata)",f"{summary['reviews_excluded']:,}")
            col4.metric("Neighborhoods Covered",       summary['neighborhoods_covered'])
            col5.metric("Cuisine Types Covered",       summary['cuisines_covered'])

            st.info(
                f"ℹ️ **{summary['reviews_excluded']:,}** reviews excluded due to unknown metadata. "
                f"Analysis based on **{summary['reviews_used']:,}** reviews."
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Sentiment Score", f"{summary['avg_sentiment']:.3f}")
            col2.metric("% Positive",          f"{summary['pct_positive']}%")
            col3.metric("% Neutral",           f"{summary['pct_neutral']}%")
            col4.metric("% Negative",          f"{summary['pct_negative']}%")

            with st.expander(" Reviews by Source"):
                source_df = pd.DataFrame(
                    list(summary['source_breakdown'].items()),
                    columns=["Source", "Review Count"]
                )
                st.dataframe(source_df, use_container_width=True)
        else:
            st.warning("Run `nlp.py` first to generate the summary JSON.")

        st.write("---")

        # SENTIMENT BY AREA
        st.subheader(":material/map: Sentiment by Neighborhood")
        color_area = ['#2ecc71' if x > 0 else '#e74c3c' for x in s_area['avg_sentiment']]
        fig_area = go.Figure(go.Bar(
            x=s_area['area'],
            y=s_area['avg_sentiment'],
            marker_color=color_area,
            text=s_area['avg_sentiment'].round(3),
            textposition='outside',
            customdata=s_area[['review_count', 'pct_positive', 'pct_negative']].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Avg Sentiment: %{y:.3f}<br>"
                "Reviews: %{customdata[0]}<br>"
                "% Positive: %{customdata[1]:.1f}%<br>"
                "% Negative: %{customdata[2]:.1f}%<extra></extra>"
            )
        ))
        fig_area.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        fig_area.update_layout(
            title="Average Review Sentiment by Neighborhood",
            xaxis_title="Neighborhood",
            yaxis_title="Sentiment Score (-1 to +1)",
            xaxis_tickangle=-45,
            height=450
        )
        st.plotly_chart(fig_area, use_container_width=True, key=f"area_{tag}")

        with st.expander(" View full sentiment-by-area data"):
            st.dataframe(s_area, use_container_width=True)

        st.write("---")

        # SENTIMENT BY CUISINE
        st.subheader(":material/restaurant: Sentiment by Cuisine Type")
        color_cuisine = ['#2ecc71' if x > 0 else '#e74c3c' for x in s_cuisine['avg_sentiment']]
        fig_cuisine = go.Figure(go.Bar(
            x=s_cuisine['cuisine_primary'],
            y=s_cuisine['avg_sentiment'],
            marker_color=color_cuisine,
            text=s_cuisine['avg_sentiment'].round(3),
            textposition='outside',
            customdata=s_cuisine[['review_count', 'pct_positive', 'pct_negative']].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Avg Sentiment: %{y:.3f}<br>"
                "Reviews: %{customdata[0]}<br>"
                "% Positive: %{customdata[1]:.1f}%<br>"
                "% Negative: %{customdata[2]:.1f}%<extra></extra>"
            )
        ))
        fig_cuisine.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        fig_cuisine.update_layout(
            title="Average Review Sentiment by Cuisine Type",
            xaxis_title="Cuisine",
            yaxis_title="Sentiment Score (-1 to +1)",
            xaxis_tickangle=-45,
            height=450
        )
        st.plotly_chart(fig_cuisine, use_container_width=True, key=f"cuisine_{tag}")

        with st.expander(" View full sentiment-by-cuisine data"):
            st.dataframe(s_cuisine, use_container_width=True)

        st.write("---")

        # SENTIMENT BY PRICE
        st.subheader(":material/payments: Sentiment by Price Tier")
        price_color_map = {'Budget': '#2ecc71', 'Mid-Range': '#3498db', 'High-End': '#9b59b6'}
        color_price = [
            price_color_map.get(p, '#3498db') if x > 0 else '#e74c3c'
            for p, x in zip(s_price['price_category'], s_price['avg_sentiment'])
]
        fig_price = go.Figure(go.Bar(
            x=s_price['price_category'],
            y=s_price['avg_sentiment'],
            marker_color=color_price,
            text=s_price['avg_sentiment'].round(3),
            textposition='outside',
            customdata=s_price[['review_count', 'pct_positive', 'pct_negative']].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Avg Sentiment: %{y:.3f}<br>"
                "Reviews: %{customdata[0]}<br>"
                "% Positive: %{customdata[1]:.1f}%<br>"
                "% Negative: %{customdata[2]:.1f}%<extra></extra>"
            )
        ))
        fig_price.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
        fig_price.update_layout(
            title="Average Review Sentiment by Price Tier",
            xaxis_title="Price Category",
            yaxis_title="Sentiment Score (-1 to +1)",
            height=400
        )
        st.plotly_chart(fig_price, use_container_width=True, key=f"price_{tag}")

        with st.expander(" View full sentiment-by-price data"):
            st.dataframe(s_price, use_container_width=True)

        st.write("---")
        # TF-IDF KEYWORDS
        st.subheader(":material/key: Top Keywords by Neighborhood (TF-IDF)")
        st.write("Words that uniquely characterize each neighborhood's reviews compared to all others.")

        selected_area_kw = st.selectbox("Select a neighborhood:", sorted(a_kw.keys()), key=f"kw_area_{tag}")
        if selected_area_kw:
            keywords = a_kw[selected_area_kw]

            col_wc, col_tbl = st.columns([1.4, 1])

            with col_wc:
                # Build TF-IDF weighted dict: rank 10 = highest weight, rank 1 = lowest
                n = len(keywords)
                word_weights = {word: (n - i) for i, word in enumerate(keywords)}

                wc = WordCloud(
                    width=600,
                    height=350,
                    background_color='white',
                    colormap='Set2',
                    prefer_horizontal=0.85,
                    min_font_size=12
                ).generate_from_frequencies(word_weights)

                fig_wc, ax_wc = plt.subplots(figsize=(6, 3.5))
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                ax_wc.set_title(f'"{selected_area_kw}" — Distinctive Keywords', fontsize=11)
                plt.tight_layout(pad=0)
                st.pyplot(fig_wc, use_container_width=True)
                plt.close(fig_wc)

            with col_tbl:
                st.write("**Ranked Keywords**")
                kw_df = pd.DataFrame({
                    "Rank": range(1, n + 1),
                    "Keyword": keywords,
                    "Weight":[round((n - i) / n, 2) for i in range(n)]
                })
                st.dataframe(kw_df, use_container_width=True, hide_index=True)

        st.write("---")

        st.subheader(":material/key: Top Keywords by Cuisine (TF-IDF)")
        st.write("Words that uniquely characterize each cuisine type's reviews.")

        selected_cuisine_kw = st.selectbox("Select a cuisine:", sorted(c_kw.keys()), key=f"kw_cuisine_{tag}")
        if selected_cuisine_kw:
            keywords = c_kw[selected_cuisine_kw]

            col_wc2, col_tbl2 = st.columns([1.4, 1])

            with col_wc2:
                n = len(keywords)
                word_weights = {word: (n - i) for i, word in enumerate(keywords)}

                wc2 = WordCloud(
                    width=600,
                    height=350,
                    background_color='white',
                    colormap='tab10',
                    prefer_horizontal=0.85,
                    min_font_size=12
                ).generate_from_frequencies(word_weights)

                fig_wc2, ax_wc2 = plt.subplots(figsize=(6, 3.5))
                ax_wc2.imshow(wc2, interpolation='bilinear')
                ax_wc2.axis('off')
                ax_wc2.set_title(f'"{selected_cuisine_kw}" — Distinctive Keywords', fontsize=11)
                plt.tight_layout(pad=0)
                st.pyplot(fig_wc2, use_container_width=True)
                plt.close(fig_wc2)

            with col_tbl2:
                st.write("**Ranked Keywords**")
                kw_df2 = pd.DataFrame({
                    "Rank":    range(1, n + 1),
                    "Keyword": keywords,
                    "Weight":  [round((n - i) / n, 2) for i in range(n)]
                })
                st.dataframe(kw_df2, use_container_width=True, hide_index=True)

    # ── RENDER EACH TAB ─────────────────────────────────────────────
    with tab_before:
        st.caption("Analysis based on **master_reviews.csv** — reviews with Unknown cuisine are excluded from cuisine-level charts.")
        render_nlp_tab(sentiment_area, sentiment_cuisine, sentiment_price,
                       area_keywords, cuisine_keywords, nlp_summary, tag="orig")

    with tab_after:
        if enriched_loaded:
            st.caption("Analysis based on **master_reviews_enriched.csv** — Unknown cuisine labels have been filled in by the ML classifier, so more reviews are included in cuisine-level charts.")
            render_nlp_tab(sentiment_area_enr, sentiment_cuisine_enr, sentiment_price_enr,
                           area_keywords_enr, cuisine_keywords_enr, nlp_summary_enr, tag="enr")
        else:
            st.warning("⚠️ Enriched NLP files not found. Run `nlp.py` after `cuisine_classifier.py` to generate them.")
            st.info("Expected files: `sentiment_by_area_enriched.csv`, `sentiment_by_cuisine_enriched.csv`, `sentiment_by_price_enriched.csv`, `area_keywords_enriched.json`, `cuisine_keywords_enriched.json`, `nlp_summary_enriched.json`")


# SECTION 6: 

elif selected_section == "Curated Smart Picks":
    st.header(":material/lightbulb: Top Performance Highlights")
    st.write("Separating the gold from the Sand with Best For Tags")
    st.write("---")

    render_best_for_tags(df_restaurants)

# FOOTER

st.write("---")
st.write("**COSC 482 - Data Science Project** | Lebanese Restaurant Analysis")
