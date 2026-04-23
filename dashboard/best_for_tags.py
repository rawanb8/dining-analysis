import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

FEATURE_COLS = [
    'delivery_available', 'outdoor_seating', 'reservation_required',
    'cash_only', 'credit_cards_accepted', 'wifi_available',
    'wheelchair_accessible', 'takeaway_available', 'parking_available',
    'live_music', 'pet_friendly', 'kids_friendly'
]

TAG_META = {
    "❤️ Best for Dates":      ("❤️", "Focuses on high-end or mid-range spots with romantic features like live music or outdoor seating.", "#e74c3c"),
    "👨‍👩‍👧‍👦 Best for Families":   ("👨‍👩‍👧‍👦", "Targets accessible, kids-friendly venues with reliable ratings for a stress-free experience.", "#f39c12"),
    "💎 Best Hidden Gem":     ("💎", "Identifies elite-rated restaurants that are still emerging and haven't reached mass popularity yet.", "#9b59b6"),
    "🌿 Best Outdoor Dining": ("🌿", "Highlights top-rated spots specifically known for their al fresco atmosphere and open-air seating.", "#2ecc71"),
    "💰 Best Budget Pick":    ("💰", "Showcases restaurants in the budget category that maintain excellence despite their low price point.", "#27ae60"),
    "👥 Best for Groups":     ("👥", "Looks for spacious venues that accept reservations and provide parking for larger gatherings.", "#3498db"),
}

DEFAULT_RATING_HIGH = 4.0
DEFAULT_RATING_VERY_HIGH = 4.5
DEFAULT_RATING_GOOD = 3.8
DEFAULT_RATING_FAMILY = 3.5
DEFAULT_RATING_BUDGET = 4.2
DEFAULT_REVIEWS_HIDDEN = 50

def compute_adaptive_thresholds(df):
    rating = df['rating_overall'].dropna()
    reviews = df['review_count_total'].dropna()

    thresholds = {
        'rating_high': np.percentile(rating, 80) if len(rating) > 0 else DEFAULT_RATING_HIGH,
        'rating_very_high': np.percentile(rating, 90) if len(rating) > 0 else DEFAULT_RATING_VERY_HIGH,
        'rating_good': np.percentile(rating, 60) if len(rating) > 0 else DEFAULT_RATING_GOOD,
        'rating_family': DEFAULT_RATING_FAMILY,
        'rating_budget': DEFAULT_RATING_BUDGET,
        'reviews_hidden': np.percentile(reviews, 20) if len(reviews) > 0 else DEFAULT_REVIEWS_HIDDEN,
    }
    thresholds['reviews_hidden'] = max(1, thresholds['reviews_hidden'])
    return thresholds

@st.cache_data
def assign_tags(df):
    df = df.copy()
    ratings = df['rating_overall'].fillna(0)
    reviews = df['review_count_total'].fillna(9999)
    price = df['price_category'].fillna('Unknown')

    thresholds = compute_adaptive_thresholds(df)

    def feat(col):
        return df[col] == 'TRUE'

    tag_masks = {
        "❤️ Best for Dates": (
            (feat('outdoor_seating') | feat('live_music')) &
            (ratings >= thresholds['rating_high']) &
            price.isin(['Mid-Range', 'High-End'])
        ),
        "👨‍👩‍👧‍👦 Best for Families": (
            feat('kids_friendly') &
            (ratings >= thresholds['rating_family'])
        ),
        "💎 Best Hidden Gem": (
            (ratings >= thresholds['rating_very_high']) &
            (reviews <= thresholds['reviews_hidden'])
        ),
        "🌿 Best Outdoor Dining": (
            feat('outdoor_seating') &
            (ratings >= thresholds['rating_high'])
        ),
        "💰 Best Budget Pick": (
            (price == 'Budget') &
            (ratings >= thresholds['rating_budget'])
        ),
        "👥 Best for Groups": (
            feat('reservation_required') &
            (feat('outdoor_seating') | feat('parking_available')) &
            (ratings >= thresholds['rating_good'])
        ),
    }

    tag_col_map = {}
    for tag, mask in tag_masks.items():
        # Clean column names by removing emojis for backend processing
        clean_name = tag.replace('❤️ ', '').replace('👨‍👩‍👧‍👦 ', '').replace('💎 ', '').replace('🌿 ', '').replace('💰 ', '').replace('👥 ', '')
        col = 'tag_' + clean_name.lower().replace(' ', '_')
        df[col] = mask
        tag_col_map[tag] = col

    tag_col_values = list(tag_col_map.values())
    tag_names = list(tag_col_map.keys())

    def make_list(row):
        return [tag_names[i] for i, c in enumerate(tag_col_values) if row[c]]

    df['tags'] = df[tag_col_values].apply(make_list, axis=1)
    df['tag_count'] = df[tag_col_values].sum(axis=1)

    # Weighted Match Score
    min_r, max_r = df['rating_overall'].min(), df['rating_overall'].max()
    rating_norm = (df['rating_overall'] - min_r) / (max_r - min_r) if max_r > min_r else 1
    df['Match_Strength'] = ((rating_norm * 0.7) + 0.3) * 100
    df['Match_Strength'] = df['Match_Strength'].clip(lower=0, upper=100).round(1)

    return df, tag_col_map, thresholds

# ----------------------------------------------------------------------
# MAIN RENDER FUNCTION
# ----------------------------------------------------------------------
def render_best_for_tags(df_restaurants):
    st.subheader("Best For: Restaurant Tags")
    st.markdown(
        "Restaurants are automatically tagged based on **adaptive thresholds** (percentiles) "
        "of ratings and review counts. This system identifies specific strengths relative to the current dataset."
    )
    st.write("---")

    with st.spinner("Analyzing restaurant performance..."):
        df_tagged, tag_col_map, thresholds = assign_tags(df_restaurants)

    all_tags = list(TAG_META.keys())

    # Tag Methodology Section
    st.write("### 📖 Tag Methodology: What are we tackling?")
    st.markdown(
        """
        To make our recommendations more intelligent, each tag is calculated using a mix of 
        **hard constraints** (like specific features) and **dynamic performance** (rating percentiles).
        """
    )
    intro_cols = st.columns(3)
    for i, tag in enumerate(all_tags):
        with intro_cols[i % 3]:
            st.markdown(f"**{tag}**")
            st.caption(TAG_META[tag][1]) # Index 1 is description

    # Threshold Transparency
    with st.expander("View Data Thresholds (Percentile Logic)"):
        thresh_df = pd.DataFrame([
            {"Threshold": "High rating (≥)", "Value": f"{thresholds['rating_high']:.2f} ⭐ (80th percentile)"},
            {"Threshold": "Very high rating (≥)", "Value": f"{thresholds['rating_very_high']:.2f} ⭐ (90th percentile)"},
            {"Threshold": "Good rating (≥)", "Value": f"{thresholds['rating_good']:.2f} ⭐ (60th percentile)"},
            {"Threshold": "Emerging discovery footprint (≤)", "Value": f"≤ {thresholds['reviews_hidden']:.0f} reviews (20th percentile)"},
            {"Threshold": "Family‑friendly rating (≥)", "Value": f"{thresholds['rating_family']:.2f} ⭐"},
            {"Threshold": "Budget pick rating (≥)", "Value": f"{thresholds['rating_budget']:.2f} ⭐"},
        ])
        st.dataframe(thresh_df, hide_index=True, use_container_width=True)

    # KPI Metrics
    tag_counts = {tag: int(df_tagged[col].sum()) for tag, col in tag_col_map.items()}
    total_tagged = int((df_tagged['tag_count'] > 0).sum())

    cols = st.columns(len(all_tags) + 1)
    cols[0].metric("Tagged Restaurants", f"{total_tagged:,}")
    for i, tag in enumerate(all_tags):
        # Using the tag name directly since it contains the emoji
        cols[i+1].metric(tag.split()[-1], f"{tag_counts[tag]:,}")

    st.write("---")

    # Tag Distribution Chart
    st.subheader("Tag Distribution")
    dist_df = pd.DataFrame({
        'Tag': [t for t in all_tags],
        'Count': [tag_counts[t] for t in all_tags],
        'Color': [TAG_META[t][2] for t in all_tags], # Index 2 is color
    }).sort_values('Count', ascending=True)

    fig_dist = px.bar(
        dist_df, x='Count', y='Tag', orientation='h',
        text='Count', color='Tag', color_discrete_sequence=dist_df['Color'].tolist(),
        height=380
    )
    fig_dist.update_traces(textposition='outside')
    fig_dist.update_layout(showlegend=False, yaxis_title=None)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.write("---")

    # Tag Overlap Heatmap
    with st.expander("Tag Overlap — Co-occurrence Analysis", expanded=True):
        tag_cols = list(tag_col_map.values())
        tag_labels = [t for t in tag_col_map]
        overlap = np.zeros((len(tag_cols), len(tag_cols)), dtype=int)
        for i, ci in enumerate(tag_cols):
            for j, cj in enumerate(tag_cols):
                overlap[i, j] = int((df_tagged[ci] & df_tagged[cj]).sum())

        fig_overlap = go.Figure(data=go.Heatmap(
            z=overlap, x=tag_labels, y=tag_labels,
            colorscale='Blues', text=overlap, texttemplate='%{text}',
        ))
        fig_overlap.update_layout(title='Number of Restaurants Sharing Both Tags', height=440, xaxis_tickangle=-30)
        st.plotly_chart(fig_overlap, use_container_width=True)

    # Key Insights Section
    st.write("### Key Insights")
    total_hidden = tag_counts["💎 Best Hidden Gem"]
    excl_hidden = ((df_tagged[tag_col_map["💎 Best Hidden Gem"]]) & (df_tagged['tag_count'] == 1)).sum()
    hidden_rate = (excl_hidden / total_hidden * 100) if total_hidden > 0 else 0

    ins_col1, ins_col2 = st.columns(2)
    with ins_col1:
        st.info(f"**Tag Exclusivity: Hidden Gems**\nHidden Gems show a very high exclusivity rate (~{hidden_rate:.1f}%). This confirms they are specialized, niche spots that rarely overlap with broad categories.")
    with ins_col2:
        st.success("**Versatile Spots: Best for Dates**\n'Best for Dates' has the lowest exclusivity. A romantic atmosphere frequently overlaps with premium features like Outdoor Dining or Live Music.")

    st.write("---")

    # Curated Smart Picks (Recommendation Engine)
    st.subheader("Curated Smart Picks")
    st.markdown("Top-tier restaurants ranked by their **Match Confidence** score.")
    multi = (
        df_tagged[df_tagged['tag_count'] >= 1]
        [['name', 'area', 'rating_overall', 'tags', 'Match_Strength']]
        .sort_values(['Match_Strength', 'rating_overall'], ascending=False)
        .head(15)
    )

    if not multi.empty:
        # Index 0 is now the emoji
        multi['Display Tags'] = multi['tags'].apply(lambda lst: ' '.join(TAG_META[t][0] for t in lst))
        st.dataframe(
            multi[['name', 'area', 'Display Tags', 'rating_overall', 'Match_Strength']],
            column_config={
                "name": "Restaurant",
                "Match_Strength": st.column_config.ProgressColumn("Match Confidence", format="%f%%", min_value=0, max_value=100),
            },
            use_container_width=True, hide_index=True
        )

    # Explore by Category
    st.write("---")
    st.subheader("Explore by Category")
    sel_tag = st.selectbox("Choose a tag to explore:", options=all_tags, key="tag_selector")
    
    tag_info = TAG_META[sel_tag]
    st.markdown(f"<div style='background:{tag_info[2]};color:white;padding:12px;border-radius:8px;'><b>{sel_tag}</b>: {tag_info[1]}</div>", unsafe_allow_html=True)

    df_tag = df_tagged[df_tagged[tag_col_map[sel_tag]]]
    
    f1, f2 = st.columns(2)
    with f1: sel_area = st.multiselect("Filter Area:", sorted(df_tag['area'].dropna().unique()))
    with f2: sel_cuisine = st.multiselect("Filter Cuisine:", sorted(df_tag['cuisine_primary'].dropna().unique()))

    if sel_area: df_tag = df_tag[df_tag['area'].isin(sel_area)]
    if sel_cuisine: df_tag = df_tag[df_tag['cuisine_primary'].isin(sel_cuisine)]

    st.dataframe(df_tag[['name', 'area', 'cuisine_primary', 'rating_overall', 'Match_Strength']].sort_values('rating_overall', ascending=False), use_container_width=True, hide_index=True)

    # Category Breakdown Charts
    c_left, c_right = st.columns(2)
    with c_left:
        area_data = df_tag['area'].value_counts().head(10).reset_index()
        fig_a = px.bar(area_data, x='count', y='area', orientation='h', title=f"{sel_tag} by Area", color_discrete_sequence=[tag_info[2]])
        st.plotly_chart(fig_a, use_container_width=True)
    with c_right:
        rating_dist = df_tag['rating_overall'].dropna()
        fig_h = px.histogram(rating_dist, nbins=10, title=f"{sel_tag} Rating Distribution", color_discrete_sequence=[tag_info[2]])
        st.plotly_chart(fig_h, use_container_width=True)