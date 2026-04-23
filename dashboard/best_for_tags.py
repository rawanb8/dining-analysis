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

# Tag definitions: label → (emoji, description, color)
TAG_META = {
    "Best for Dates":      ("💑", "Romantic atmosphere — outdoor or live music, high rating, mid/high-end", "#e74c3c"),
    "Best for Families":   ("👨‍👩‍👧", "Kids-friendly with solid ratings",                                     "#f39c12"),
    "Best Hidden Gem":     ("💎", "Highly rated but few reviews — still under the radar",                  "#9b59b6"),
    "Best Outdoor Dining": ("🌿", "Outdoor seating with a great rating",                                   "#2ecc71"),
    "Best Budget Pick":    ("💸", "Budget-friendly with outstanding ratings",                              "#27ae60"),
    "Best for Groups":     ("🎉", "Takes reservations, has space/parking, well-rated",                     "#3498db"),
}


@st.cache_data
def assign_tags(df):
    """Return df with a 'tags' list column and one boolean column per tag."""
    df = df.copy()

    rating  = df['rating_overall'].fillna(0)
    reviews = df['review_count_total'].fillna(9999)
    price   = df['price_category'].fillna('Unknown')

    def feat(col):
        return df[col] == 'TRUE'

    tag_masks = {
        "Best for Dates": (
            (feat('outdoor_seating') | feat('live_music')) &
            (rating >= 4.0) &
            price.isin(['Mid-Range', 'High-End'])
        ),
        "Best for Families": (
            feat('kids_friendly') &
            (rating >= 3.5)
        ),
        "Best Hidden Gem": (
            (rating >= 4.5) &
            (reviews < 50)
        ),
        "Best Outdoor Dining": (
            feat('outdoor_seating') &
            (rating >= 4.0)
        ),
        "Best Budget Pick": (
            (price == 'Budget') &
            (rating >= 4.2)
        ),
        "Best for Groups": (
            feat('reservation_required') &
            (feat('outdoor_seating') | feat('parking_available')) &
            (rating >= 3.8)
        ),
    }

    tag_col_map = {}
    for tag, mask in tag_masks.items():
        col = 'tag_' + tag.lower().replace(' ', '_').replace('/', '_')
        df[col] = mask
        tag_col_map[tag] = col

    # Build the list column using the already-computed boolean columns
    tag_col_values = list(tag_col_map.values())
    tag_names      = list(tag_col_map.keys())

    def make_list(row):
        return [tag_names[i] for i, c in enumerate(tag_col_values) if row[c]]

    df['tags'] = df[tag_col_values].apply(make_list, axis=1)
    df['tag_count'] = df[tag_col_values].sum(axis=1)

    return df, tag_col_map


def render_best_for_tags(df_restaurants):
    st.subheader(":material/label: Best For — Restaurant Tags")
    st.write(
        "Restaurants are automatically tagged based on their features, ratings, and price category. "
        "Use the filters below to explore restaurants by what they are best for."
    )
    st.write("---")

    with st.spinner("Computing tags…"):
        df_tagged, tag_col_map = assign_tags(df_restaurants)

    all_tags = list(TAG_META.keys())

    # ── KPI: how many restaurants earned each tag ─────────────────────────────
    tag_counts = {tag: int(df_tagged[col].sum()) for tag, col in tag_col_map.items()}
    total_tagged = int((df_tagged['tag_count'] > 0).sum())

    col_kpis = st.columns(len(all_tags) + 1)
    col_kpis[0].metric("Restaurants with ≥1 Tag", f"{total_tagged:,}")
    for i, tag in enumerate(all_tags):
        emoji = TAG_META[tag][0]
        col_kpis[i + 1].metric(f"{emoji} {tag}", f"{tag_counts[tag]:,}")

    st.write("---")

    # ── TAG DISTRIBUTION BAR CHART ────────────────────────────────────────────
    st.subheader(":material/bar_chart: Tag Distribution")
    dist_df = pd.DataFrame({
        'Tag':   [f"{TAG_META[t][0]} {t}" for t in all_tags],
        'Count': [tag_counts[t] for t in all_tags],
        'Color': [TAG_META[t][2] for t in all_tags],
    }).sort_values('Count', ascending=True)

    fig_dist = px.bar(
        dist_df, x='Count', y='Tag', orientation='h',
        text='Count', title='How many restaurants have each tag?',
        color='Tag',
        color_discrete_sequence=dist_df['Color'].tolist(),
        height=380
    )
    fig_dist.update_traces(textposition='outside')
    fig_dist.update_layout(showlegend=False, yaxis_title=None)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.write("---")

    # ── MULTI-TAG RESTAURANTS ─────────────────────────────────────────────────
    st.subheader(":material/stars: Multi-Tag Champions")
    st.write("Restaurants that qualified for **2 or more tags** — the most versatile spots.")

    multi = (
        df_tagged[df_tagged['tag_count'] >= 2]
        [['name', 'area', 'cuisine_primary', 'price_category', 'rating_overall', 'review_count_total', 'tags', 'tag_count']]
        .sort_values(['tag_count', 'rating_overall'], ascending=[False, False])
        .head(20)
    )

    if len(multi) > 0:
        multi_display = multi.copy()
        multi_display['tags'] = multi_display['tags'].apply(
            lambda lst: '  ·  '.join(f"{TAG_META[t][0]} {t}" for t in lst)
        )
        multi_display = multi_display.rename(columns={
            'name': 'Restaurant', 'area': 'Area',
            'cuisine_primary': 'Cuisine', 'price_category': 'Price',
            'rating_overall': 'Rating', 'review_count_total': 'Reviews',
            'tags': 'Tags', 'tag_count': '# Tags'
        })
        st.dataframe(multi_display, use_container_width=True, hide_index=True)
    else:
        st.info("No restaurants qualified for 2+ tags with the current data.")

    st.write("---")

    # ── EXPLORE BY TAG ────────────────────────────────────────────────────────
    st.subheader(":material/filter_alt: Explore by Tag")

    sel_tag = st.selectbox(
        "Choose a tag to explore:",
        options=all_tags,
        format_func=lambda t: f"{TAG_META[t][0]}  {t} — {TAG_META[t][1]}",
        key="tag_selector"
    )

    tag_col    = tag_col_map[sel_tag]
    tag_emoji  = TAG_META[sel_tag][0]
    tag_color  = TAG_META[sel_tag][2]
    tag_desc   = TAG_META[sel_tag][1]

    st.markdown(
        f"<div style='background:{tag_color};color:white;padding:10px 18px;"
        f"border-radius:8px;margin-bottom:12px;'>"
        f"<b>{tag_emoji} {sel_tag}</b> &nbsp;—&nbsp; {tag_desc}</div>",
        unsafe_allow_html=True
    )

    df_tag = (
        df_tagged[df_tagged[tag_col]]
        [['name', 'area', 'cuisine_primary', 'price_category', 'rating_overall', 'review_count_total', 'tags']]
        .sort_values('rating_overall', ascending=False)
    )

    st.write(f"**{len(df_tag):,} restaurants** match this tag.")

    # Mini filters for the tag view
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        areas_in_tag = sorted(df_tag['area'].dropna().unique().tolist())
        sel_area = st.multiselect("Filter by Area:", areas_in_tag, key="tag_area")
    with filter_col2:
        prices_in_tag = sorted(df_tag['price_category'].dropna().unique().tolist())
        sel_price = st.multiselect("Filter by Price:", prices_in_tag, key="tag_price")
    with filter_col3:
        cuisines_in_tag = sorted(df_tag['cuisine_primary'].dropna().unique().tolist())
        sel_cuisine = st.multiselect("Filter by Cuisine:", cuisines_in_tag, key="tag_cuisine")

    if sel_area:    df_tag = df_tag[df_tag['area'].isin(sel_area)]
    if sel_price:   df_tag = df_tag[df_tag['price_category'].isin(sel_price)]
    if sel_cuisine: df_tag = df_tag[df_tag['cuisine_primary'].isin(sel_cuisine)]

    # Display table
    df_tag_display = df_tag.copy()
    df_tag_display['tags'] = df_tag_display['tags'].apply(
        lambda lst: '  ·  '.join(f"{TAG_META[t][0]} {t}" for t in lst)
    )
    df_tag_display = df_tag_display.rename(columns={
        'name': 'Restaurant', 'area': 'Area',
        'cuisine_primary': 'Cuisine', 'price_category': 'Price',
        'rating_overall': 'Rating', 'review_count_total': 'Reviews', 'tags': 'All Tags'
    })
    st.dataframe(df_tag_display, use_container_width=True, hide_index=True)

    st.write("---")

    # ── BREAKDOWN CHARTS FOR SELECTED TAG ────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{tag_emoji} {sel_tag} — by Area**")
        area_breakdown = (
            df_tagged[df_tagged[tag_col]]['area']
            .value_counts().head(10).reset_index()
        )
        area_breakdown.columns = ['Area', 'Count']
        fig_area = px.bar(
            area_breakdown, x='Count', y='Area', orientation='h',
            text='Count', height=360,
            color_discrete_sequence=[tag_color]
        )
        fig_area.update_traces(textposition='outside')
        fig_area.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        st.plotly_chart(fig_area, use_container_width=True)

    with col2:
        st.markdown(f"**{tag_emoji} {sel_tag} — by Price Category**")
        price_breakdown = (
            df_tagged[df_tagged[tag_col]]['price_category']
            .value_counts().reindex(['Budget', 'Mid-Range', 'High-End']).fillna(0).reset_index()
        )
        price_breakdown.columns = ['Price', 'Count']
        PRICE_COLORS = {'Budget': '#2ecc71', 'Mid-Range': '#3498db', 'High-End': '#e74c3c'}
        fig_price = px.bar(
            price_breakdown, x='Price', y='Count',
            color='Price', color_discrete_map=PRICE_COLORS,
            text='Count', height=360
        )
        fig_price.update_traces(textposition='outside')
        fig_price.update_layout(showlegend=False)
        st.plotly_chart(fig_price, use_container_width=True)

    # ── RATING DISTRIBUTION FOR THIS TAG ─────────────────────────────────────
    st.markdown(f"**{tag_emoji} {sel_tag} — Rating Distribution**")
    tag_ratings = df_tagged[df_tagged[tag_col]]['rating_overall'].dropna()
    if len(tag_ratings) > 0:
        fig_hist = px.histogram(
            tag_ratings, nbins=20,
            labels={'value': 'Rating', 'count': 'Restaurants'},
            title=f'Rating Distribution for "{sel_tag}" Restaurants',
            color_discrete_sequence=[tag_color],
            height=320
        )
        fig_hist.update_layout(yaxis_title="Number of Restaurants", xaxis_title="Rating")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.write("---")

    # ── TAG OVERLAP HEATMAP ───────────────────────────────────────────────────
    with st.expander("View Tag Overlap Heatmap — which tags co-occur?"):
        tag_cols = list(tag_col_map.values())
        tag_labels = [f"{TAG_META[t][0]} {t}" for t in tag_col_map]

        overlap = np.zeros((len(tag_cols), len(tag_cols)), dtype=int)
        for i, ci in enumerate(tag_cols):
            for j, cj in enumerate(tag_cols):
                overlap[i, j] = int((df_tagged[ci] & df_tagged[cj]).sum())

        fig_overlap = go.Figure(data=go.Heatmap(
            z=overlap,
            x=tag_labels, y=tag_labels,
            colorscale='Blues',
            text=overlap, texttemplate='%{text}',
            textfont={"size": 11},
        ))
        fig_overlap.update_layout(
            title='Number of Restaurants Sharing Both Tags',
            height=440, xaxis_tickangle=-30
        )
        st.plotly_chart(fig_overlap, use_container_width=True)
