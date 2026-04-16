import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import json, os

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Lebanon Restaurant Analysis",
    page_icon=":material/restaurant:",
    layout="wide"
)

# LOAD DATA

@st.cache_data
def load_data():
    # Try to load geocoded file first, fallback to regular file
    try:
        restaurants = pd.read_csv(r"../merged/master_restaurants_geocoded.csv")
        print("✓ Loaded geocoded restaurant data")
    except FileNotFoundError:
        restaurants = pd.read_csv(r"../merged/master_restaurants.csv")
        print("⚠️ Geocoded file not found, using non-geocoded data")
    
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

st.title("Lebasese Restaurant Analysis Dashboard")
st.write("COSC 482 - Data Science Project | Interactive exploration of Lebanese restaurants")
st.write("---")

# TOP NAVIGATION

selected_section = option_menu(
    menu_title=None,
    options=["Search & Filter", "EDA", "Feature Analysis", "NLP Analysis", "ML Insights"],
    icons=["search", "bar-chart-line", "toggles", "chat-left-text", "cpu"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0", "margin": "0 0 1rem 0"},
        "nav-link-selected": {"background-color": "#3ad1e5"},
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
    cuisine_options = ["All Cuisines"] + sorted(df_restaurants['cuisine_primary'].unique().tolist())
    selected_cuisine = st.sidebar.selectbox(" Cuisine Type:", cuisine_options)
    
    # Area filter
    area_options = ["All Areas"] + sorted(df_restaurants['area'].unique().tolist())
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
    filter_delivery = st.sidebar.checkbox("Delivery Available")
    filter_outdoor = st.sidebar.checkbox("Outdoor Seating")
    filter_parking = st.sidebar.checkbox("Parking Available")
    filter_wifi = st.sidebar.checkbox("WiFi")
    filter_music = st.sidebar.checkbox("Live Music")
    filter_kids = st.sidebar.checkbox("Kids Friendly")
    
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
    
    if filter_delivery:
        filtered_df = filtered_df[filtered_df['delivery_available'] == '✓']
    if filter_outdoor:
        filtered_df = filtered_df[filtered_df['outdoor_seating'] == '✓']
    if filter_parking:
        filtered_df = filtered_df[filtered_df['parking_available'] == '✓']
    if filter_wifi:
        filtered_df = filtered_df[filtered_df['wifi_available'] == '✓']
    if filter_music:
        filtered_df = filtered_df[filtered_df['live_music'] == '✓']
    if filter_kids:
        filtered_df = filtered_df[filtered_df['kids_friendly'] == '✓']
    
    # Display results
    st.subheader(f":material/table_rows: Showing {len(filtered_df)} of {len(df_display)} restaurants")
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total Restaurants", len(filtered_df))
    col2.metric("Average Rating", f"{filtered_df['rating_overall'].mean():.2f}⭐")
    col3.metric("Total Reviews", f"{filtered_df['review_count_total'].sum():,}")
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
    with st.expander(f"📋 Click to view full filtered dataset ({len(filtered_df)} records)"):
        st.dataframe(filtered_df, use_container_width=True)

# SECTION 2: GENERAL ANALYSIS (Placeholder for Friend 1)

elif selected_section == "EDA":
    st.header("Exploratory Data Analysis")

    
    st.write("---")

    # Top 10 cuisines
    st.subheader("1️⃣ Cuisine Distribution (Top 10 - Excluding Unknown)")
    cuisine_counts = df_restaurants["cuisine_primary"].value_counts().head(10)
    total_restaurants = len(df_restaurants)
    unknown_count = (df_restaurants["cuisine_primary"] == "Unknown").sum()
    unknown_pct = (unknown_count / total_restaurants) * 100

    st.info(f"⚠️ {unknown_pct:.1f}% of restaurants have unknown cuisine.")

    cuisine_counts = df_restaurants[
        df_restaurants["cuisine_primary"] != "Unknown"
    ]["cuisine_primary"].value_counts().head(10)

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

    st.write("---")

    st.subheader("2️⃣ Rating Distribution")

    rating_counts = df_restaurants["rating_overall"].value_counts().sort_index()

    fig2 = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={"x": "Rating", "y": "Number of Restaurants"},
        title="Distribution of Restaurant Ratings"
    )

    st.plotly_chart(fig2, use_container_width=True)
    avg_rating = df_restaurants["rating_overall"].mean()
    st.metric("⭐ Average Rating", f"{avg_rating:.2f}")


    st.write("---")

    st.subheader("3️⃣ Price Category Breakdown (Excluding Unknown)")

    # Calculate Unknown percentage
    total_restaurants = len(df_restaurants)
    unknown_count = (df_restaurants["price_category"] == "Unknown").sum()
    unknown_pct = (unknown_count / total_restaurants) * 100

    st.info(f"⚠️ {unknown_pct:.1f}% of restaurants have unknown price category.")

    # Remove Unknown for visualization
    price_counts = df_restaurants[
        df_restaurants["price_category"] != "Unknown"
    ]["price_category"].value_counts()

    # Plot
    fig3 = px.bar(
        x=price_counts.index,
        y=price_counts.values,
        labels={"x": "Price Category", "y": "Number of Restaurants"},
        title="Distribution of Restaurants by Price Category",
        text=price_counts.values
    )

    fig3.update_traces(textposition='outside')

    st.plotly_chart(fig3, use_container_width=True)

    st.write("---")

    st.subheader("4️⃣ Restaurants by Area (Top 10 - Excluding Unknown)")

    # Calculate Unknown percentage
    total_restaurants = len(df_restaurants)
    unknown_count = (df_restaurants["area"] == "Unknown").sum()
    unknown_pct = (unknown_count / total_restaurants) * 100

    st.info(f"⚠️ {unknown_pct:.1f}% of restaurants have unknown area.")

    # Remove Unknown and get top 10
    area_counts = df_restaurants[
        df_restaurants["area"] != "Unknown"
    ]["area"].value_counts().head(10)

    # Plot
    fig4 = px.bar(
        x=area_counts.index,
        y=area_counts.values,
        labels={"x": "Area", "y": "Number of Restaurants"},
        title="Top 10 Areas by Number of Restaurants",
        text=area_counts.values
    )

    fig4.update_traces(textposition='outside')
    fig4.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig4, use_container_width=True)

    st.write("---")

    st.subheader("5️⃣ Review Count Distribution")

    # Convert to numeric
    df_restaurants["review_count_total"] = pd.to_numeric(
        df_restaurants["review_count_total"], errors="coerce"
    )

    fig5a = px.histogram(
        df_restaurants,
        x="review_count_total",
        nbins=30,
        title="Distribution of Review Counts"
    )
    fig5a.update_layout(
        xaxis_title="Number of Reviews",
        yaxis_title="Number of Restaurants"
    )
    st.plotly_chart(fig5a, use_container_width=True)


    st.write("---")

    st.subheader("6️⃣ Top 10 Most Reviewed Restaurants")

    top_reviewed = df_restaurants.nlargest(10, "review_count_total")[
        ["name", "review_count_total", "rating_overall", "area", "cuisine_primary", "price_category"]
    ].copy()

    top_reviewed.columns = [
        "Restaurant Name",
        "Total Reviews",
        "Rating",
        "Area",
        "Cuisine",
        "Price Category"
    ]

    st.dataframe(top_reviewed, use_container_width=True)

    st.write("---")
    st.subheader(":material/show_chart: Rating vs Review Count")

    fig = px.scatter(
        df_restaurants,
        x="review_count_total",
        y="rating_overall",
        title="Rating vs Number of Reviews",
        labels={"review_count_total": "Number of Reviews", "rating_overall": "Rating"},
        opacity=0.6
    )

    fig.update_layout(xaxis_type="log")

    st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.subheader(":material/diamond: Hidden Gems (High Rating, Low Reviews)")

    hidden = df_restaurants[
        (df_restaurants["rating_overall"] >= 4.5) &
        (df_restaurants["review_count_total"] < 50)
    ].head(10)[
        ["name", "rating_overall", "review_count_total", "area"]
    ]

    st.dataframe(hidden, use_container_width=True)





    st.write("---")
    st.subheader(":material/bubble_chart: Density vs Quality by Area")
    df_geo = df_restaurants[df_restaurants["area"] != "Unknown"].copy()

    area_stats = df_geo.groupby("area").agg({
        "name": "count",
        "rating_overall": "mean"
    }).rename(columns={"name": "restaurant_count"})

    area_stats = area_stats[area_stats["restaurant_count"] >= 5]
    area_stats["total_reviews"] = df_geo.groupby("area")["review_count_total"].sum()

    fig = px.scatter(
        area_stats,
        x="restaurant_count",
        y="rating_overall",
        size="total_reviews", 
        text=area_stats.index,
        title="Density vs Quality by Area (Bubble = Popularity)"
    )


    fig.update_traces(textposition="top center")

    st.plotly_chart(fig, use_container_width=True)
    
    st.write("---")

    st.subheader(":material/emoji_events: Best Areas Overall (Combined Score)")

    df_geo = df_restaurants[df_restaurants["area"] != "Unknown"].copy()

  
    area_stats = df_geo.groupby("area").agg({
        "rating_overall": "mean",
        "review_count_total": "sum",
        "name": "count"
    }).rename(columns={"name": "restaurant_count"})

   
    area_stats = area_stats[area_stats["restaurant_count"] >= 5]


    area_stats["norm_rating"] = area_stats["rating_overall"] / 5
    area_stats["norm_reviews"] = area_stats["review_count_total"] / area_stats["review_count_total"].max()
    area_stats["norm_density"] = area_stats["restaurant_count"] / area_stats["restaurant_count"].max()


    area_stats["score"] = (
        area_stats["norm_rating"] +
        area_stats["norm_reviews"] +
        area_stats["norm_density"]
    ) / 3

    # Top areas
    top_areas = area_stats.sort_values("score", ascending=False).head(10)

    # Plot
    fig = px.bar(
        x=top_areas.index,
        y=top_areas["score"],
        title="Top Areas Overall (Quality + Popularity + Density)",
        labels={"x": "Area", "y": "Score"},
        text=[f"{v:.2f}" for v in top_areas["score"]]
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)

# SECTION 3: FEATURE ANALYSIS (YOUR WORK)

elif selected_section == "Feature Analysis":
    st.header(":material/auto_awesome: Feature Analysis")
    st.write("Analysis of 12 features extracted from restaurant descriptions and reviews")
    
    # Calculate feature statistics
    feature_stats = {}
    for feature in feature_cols:
        count = (df_restaurants[feature] == 'TRUE').sum()
        pct = (count / len(df_restaurants)) * 100
        feature_stats[feature.replace('_', ' ').title()] = {'count': count, 'percent': pct}
    
    # Summary metrics
    st.subheader(":material/bar_chart: Feature Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate actual values
    outdoor_count = (df_restaurants['outdoor_seating'] == 'TRUE').sum()
    outdoor_pct = (outdoor_count / len(df_restaurants)) * 100
    avg_features = sum([v['count'] for v in feature_stats.values()]) / len(df_restaurants)
    
    col1.metric("Most Common Feature", "Outdoor Seating", f"{outdoor_pct:.0f}%")
    col2.metric("Total Restaurants", len(df_restaurants))
    col3.metric("Features Detected", "12 types")
    col4.metric("Avg Features/Restaurant", f"{avg_features:.1f}")
    
    st.write("---")
    
    # Chart 1: Feature Availability
    st.subheader("1️⃣ Feature Availability Overview")
    
    feature_names = list(feature_stats.keys())
    feature_counts = [feature_stats[f]['count'] for f in feature_names]
    feature_pcts = [feature_stats[f]['percent'] for f in feature_names]
    
    fig1 = px.bar(
        x=feature_names,
        y=feature_counts,
        labels={'x': 'Feature', 'y': 'Number of Restaurants'},
        title='Feature Availability Across All Restaurants',
        text=[f"{c}<br>({p:.1f}%)" for c, p in zip(feature_counts, feature_pcts)]
    )
    fig1.update_traces(textposition='outside')
    fig1.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.write("---")
    
    # Charts 2 & 3: Features by Area and Cuisine
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("2️⃣ Features by Top Areas")
        
        # Get top 10 areas
        top_areas = df_restaurants['area'].value_counts().head(10).index.tolist()
        
        # Calculate feature percentages by area
        area_data = []
        for area in top_areas:
            area_df = df_restaurants[df_restaurants['area'] == area]
            for feature in ['outdoor_seating', 'parking_available', 'wifi_available', 'live_music']:
                count = (area_df[feature] == 'TRUE').sum()
                pct = (count / len(area_df) * 100) if len(area_df) > 0 else 0
                area_data.append({
                    'Area': area,
                    'Feature': feature.replace('_', ' ').title(),
                    'Percentage': pct
                })
        
        area_df_plot = pd.DataFrame(area_data)
        
        fig2 = px.bar(
            area_df_plot,
            x='Area',
            y='Percentage',
            color='Feature',
            title='Top Features by Area (Top 10 Areas)',
            labels={'Percentage': 'Percentage (%)'},
            barmode='group',
            height=450
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
        
    with col2:
        st.subheader("3️⃣ Features by Cuisine")
        
        # Get top 8 cuisines (excluding Unknown)
        top_cuisines = df_restaurants[df_restaurants['cuisine_primary'] != 'Unknown']['cuisine_primary'].value_counts().head(8).index.tolist()
        
        # Calculate feature percentages by cuisine
        cuisine_data = []
        for cuisine in top_cuisines:
            cuisine_df = df_restaurants[df_restaurants['cuisine_primary'] == cuisine]
            for feature in ['outdoor_seating', 'delivery_available', 'reservation_required', 'live_music']:
                count = (cuisine_df[feature] == 'TRUE').sum()
                pct = (count / len(cuisine_df) * 100) if len(cuisine_df) > 0 else 0
                cuisine_data.append({
                    'Cuisine': cuisine,
                    'Feature': feature.replace('_', ' ').title(),
                    'Percentage': pct
                })
        
        cuisine_df_plot = pd.DataFrame(cuisine_data)
        
        fig3 = px.bar(
            cuisine_df_plot,
            x='Cuisine',
            y='Percentage',
            color='Feature',
            title='Top Features by Cuisine (Top 8)',
            labels={'Percentage': 'Percentage (%)'},
            barmode='group',
            height=450
        )
        fig3.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)
    
    st.write("---")
    
    # Chart 4: Features by Price
    st.subheader("4️⃣ Features by Price Category")
    
    price_categories = ['Budget', 'Mid-Range', 'High-End']
    price_feature_data = []
    
    for price in price_categories:
        price_df = df_restaurants[df_restaurants['price_category'] == price]
        if len(price_df) == 0:
            continue
        for feature in ['outdoor_seating', 'parking_available', 'wifi_available', 'live_music', 'delivery_available']:
            count = (price_df[feature] == 'TRUE').sum()
            pct = (count / len(price_df) * 100)
            price_feature_data.append({
                'Price': price,
                'Feature': feature.replace('_', ' ').title(),
                'Percentage': pct
            })
    
    price_df_plot = pd.DataFrame(price_feature_data)
    
    fig4 = px.bar(
        price_df_plot,
        x='Feature',
        y='Percentage',
        color='Price',
        title='Feature Availability by Price Category',
        labels={'Percentage': 'Percentage (%)'},
        barmode='group',
        height=500,
        color_discrete_map={'Budget': '#2ecc71', 'Mid-Range': '#3498db', 'High-End': '#e74c3c'}
    )
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)
    
    st.write("---")
    
    # Star Distribution Analysis
    st.subheader(":material/star: Star Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("5️⃣ Average Star Distribution")
        
        star_averages = [
            df_restaurants['star_5_percent'].mean(),
            df_restaurants['star_4_percent'].mean(),
            df_restaurants['star_3_percent'].mean(),
            df_restaurants['star_2_percent'].mean(),
            df_restaurants['star_1_percent'].mean()
        ]
        
        fig5 = go.Figure(data=[
            go.Bar(
                x=['5 Stars', '4 Stars', '3 Stars', '2 Stars', '1 Star'],
                y=star_averages,
                text=[f"{avg:.1f}%" for avg in star_averages],
                textposition='outside',
                marker_color=['#27ae60', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
            )
        ])
        
        fig5.update_layout(
            title='Average Star Distribution Across All Restaurants',
            xaxis_title='Rating',
            yaxis_title='Average Percentage (%)',
            height=450,
            showlegend=False
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        st.subheader("6️⃣ Star Distribution by Price")
        
        # Calculate star distribution by price
        price_star_data = []
        star_cols_list = ['star_5_percent', 'star_4_percent', 'star_3_percent', 'star_2_percent', 'star_1_percent']
        star_labels = ['5 Stars', '4 Stars', '3 Stars', '2 Stars', '1 Star']
        
        for price in price_categories:
            price_df = df_restaurants[df_restaurants['price_category'] == price]
            if len(price_df) == 0:
                continue
            for i, star_col in enumerate(star_cols_list):
                avg = price_df[star_col].mean()
                price_star_data.append({
                    'Price': price,
                    'Stars': star_labels[i],
                    'Percentage': avg
                })
        
        price_star_df = pd.DataFrame(price_star_data)
        
        fig6 = px.bar(
            price_star_df,
            x='Stars',
            y='Percentage',
            color='Price',
            title='Star Distribution by Price Category',
            labels={'Percentage': 'Average Percentage (%)'},
            barmode='group',
            height=450,
            color_discrete_map={'Budget': '#2ecc71', 'Mid-Range': '#3498db', 'High-End': '#e74c3c'}
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    st.write("---")
    
    # Data Quality Analysis
    st.subheader(":material/monitoring: Data Quality Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("7️⃣ Data Completeness by Source")
        
        # Calculate completeness for each source
        sources = ['source1', 'source2', 'source3']
        source_labels = ['Source 1', 'Source 2', 'Source 3']
        completeness_data = []
        
        for i, source in enumerate(sources):
            source_df = df_restaurants[df_restaurants['data_source'] == source]
            
            if len(source_df) == 0:
                continue
            
            phone_pct = (source_df['phone'].notna().sum() / len(source_df)) * 100
            cuisine_pct = ((source_df['cuisine_primary'] != 'Unknown').sum() / len(source_df)) * 100
            hours_pct = ((source_df['hours_monday'] != 'Unknown').sum() / len(source_df)) * 100
            price_pct = ((source_df['price_category'] != 'Unknown').sum() / len(source_df)) * 100
            
            completeness_data.append({
                'Source': source_labels[i],
                'Field': 'Phone',
                'Completeness': phone_pct
            })
            completeness_data.append({
                'Source': source_labels[i],
                'Field': 'Cuisine',
                'Completeness': cuisine_pct
            })
            completeness_data.append({
                'Source': source_labels[i],
                'Field': 'Hours',
                'Completeness': hours_pct
            })
            completeness_data.append({
                'Source': source_labels[i],
                'Field': 'Price',
                'Completeness': price_pct
            })
        
        completeness_df = pd.DataFrame(completeness_data)
        
        fig7 = px.bar(
            completeness_df,
            x='Source',
            y='Completeness',
            color='Field',
            title='Data Completeness by Source',
            labels={'Completeness': 'Completeness (%)'},
            barmode='group',
            height=450
        )
        st.plotly_chart(fig7, use_container_width=True)
        
    with col2:
        st.subheader("8️⃣ Feature Detection by Source")
        
        # Calculate feature detection for each source
        source_feature_data = []
        
        for i, source in enumerate(sources):
            source_df = df_restaurants[df_restaurants['data_source'] == source]
            
            if len(source_df) == 0:
                continue
            
            for feature in ['outdoor_seating', 'parking_available', 'wifi_available', 'live_music', 'delivery_available']:
                count = (source_df[feature] == 'TRUE').sum()
                pct = (count / len(source_df)) * 100
                source_feature_data.append({
                    'Source': source_labels[i],
                    'Feature': feature.replace('_', ' ').title(),
                    'Detection Rate': pct
                })
        
        source_feature_df = pd.DataFrame(source_feature_data)
        
        fig8 = px.bar(
            source_feature_df,
            x='Feature',
            y='Detection Rate',
            color='Source',
            title='Feature Detection Rate by Source',
            labels={'Detection Rate': 'Detection Rate (%)'},
            barmode='group',
            height=450
        )
        fig8.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig8, use_container_width=True)
    
    st.write("---")
    
    # Chart 9: Feature Correlation Heatmap
    st.subheader("9️⃣ Feature Correlation Heatmap")
    
    # Create binary matrix for features
    feature_matrix = df_restaurants[feature_cols].map(lambda x: 1 if x == 'TRUE' else 0)
    
    # Calculate correlation
    correlation = feature_matrix.corr()
    
    # Create heatmap
    fig9 = go.Figure(data=go.Heatmap(
        z=correlation.values,
        x=[f.replace('_', ' ').title() for f in feature_cols],
        y=[f.replace('_', ' ').title() for f in feature_cols],
        colorscale='RdBu',
        zmid=0,
        text=correlation.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 9},
        colorbar=dict(title="Correlation")
    ))
    
    fig9.update_layout(
        title='Feature Correlation Matrix',
        height=700,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig9, use_container_width=True)
    st.write("---")

# Geographic Maps
    if 'latitude' in df_restaurants.columns and 'longitude' in df_restaurants.columns:
     geocoded_restaurants = df_restaurants[df_restaurants['latitude'].notna() & df_restaurants['longitude'].notna()]
    
    if len(geocoded_restaurants) > 0:
        st.subheader(":material/map: Geographic Distribution")
        
        # Map 1 & 2: Side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**All Restaurants by Price**")
            
            map_price = px.scatter_mapbox(
                geocoded_restaurants,
                lat='latitude',
                lon='longitude',
                hover_name='name',
                hover_data=['rating_overall', 'area'],
                color='price_category',
                color_discrete_map={'Budget': '#2ecc71', 'Mid-Range': '#3498db', 'High-End': '#e74c3c'},
                zoom=11,
                height=500
            )
            
            map_price.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(map_price, use_container_width=True)
        
        with col2:
            st.write("**Top Rated (4.5+)**")
            
            top_rated_geo = geocoded_restaurants[geocoded_restaurants['rating_overall'] >= 4.5]
            
            if len(top_rated_geo) > 0:
                map_top = px.scatter_mapbox(
                    top_rated_geo,
                    lat='latitude',
                    lon='longitude',
                    hover_name='name',
                    hover_data=['rating_overall', 'area', 'cuisine_primary'],
                    color='rating_overall',
                    size='rating_overall',
                    color_continuous_scale='RdYlGn',
                    zoom=11,
                    height=500
                )
                
                map_top.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(map_top, use_container_width=True)
        
        # Map 3: Density
        st.write("**Restaurant Density by Area**")
        
        area_counts = geocoded_restaurants['area'].value_counts().head(15).reset_index()
        area_counts.columns = ['Area', 'Count']
        
        area_coords = geocoded_restaurants.groupby('area').agg({
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        area_map_data = area_counts.merge(area_coords, left_on='Area', right_on='area')
        
        map_density = px.scatter_mapbox(
            area_map_data,
            lat='latitude',
            lon='longitude',
            size='Count',
            hover_name='Area',
            hover_data=['Count'],
            color='Count',
            color_continuous_scale='YlOrRd',
            zoom=11,
            height=500
        )
        
        map_density.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(map_density, use_container_width=True)
        
        st.info(f"📊 {len(geocoded_restaurants)} of {len(df_restaurants)} restaurants geocoded")
    st.write("---")
    
    # Feature Insights
    st.subheader(":material/lightbulb: Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Most Common Features:**")
        # Get top 3 features
        top_features = sorted(feature_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
        for i, (feature, stats) in enumerate(top_features, 1):
            st.write(f"{i}. {feature} ({stats['percent']:.1f}%)")
        
    with col2:
        st.write("**High-End Trends:**")
        high_end_df = df_restaurants[df_restaurants['price_category'] == 'High-End']
        if len(high_end_df) > 0:
            parking_pct = (high_end_df['parking_available'] == 'TRUE').sum() / len(high_end_df) * 100
            music_pct = (high_end_df['live_music'] == 'TRUE').sum() / len(high_end_df) * 100
            reserv_pct = (high_end_df['reservation_required'] == 'TRUE').sum() / len(high_end_df) * 100
            st.write(f"- {parking_pct:.0f}% have parking")
            st.write(f"- {music_pct:.0f}% have live music")
            st.write(f"- {reserv_pct:.0f}% require reservations")
        
    with col3:
        st.write("**Data Quality:**")
        for i, source in enumerate(sources):
            source_df = df_restaurants[df_restaurants['data_source'] == source]
            if len(source_df) > 0:
                completeness = (source_df['phone'].notna().sum() + 
                              (source_df['cuisine_primary'] != 'Unknown').sum()) / (2 * len(source_df)) * 100
                st.write(f"- {source_labels[i]}: {completeness:.0f}% complete")

# SECTION 4: ML INSIGHTS 

elif selected_section == "ML Insights":
    st.header(":material/smart_toy: ML Insights — Cuisine Classifier")
    st.write("We trained a text classifier on review content to predict the cuisine type for ~17k reviews that had missing metadata, unlocking them for the full NLP pipeline.")
    st.write("---")

    # ── LOAD ML OUTPUT FILES 
    

    ML_DIR = os.path.join(os.path.dirname(__file__), '..', 'machine_learning')
    @st.cache_data
    def load_ml_data():
        summary_path     = os.path.join(ML_DIR, 'ml_cuisine_summary.json')
        comparison_path  = os.path.join(ML_DIR, 'ml_model_comparison.csv')
        pred_dist_path   = os.path.join(ML_DIR, 'ml_predicted_distribution.csv')

        with open(summary_path, 'r') as f:
            summary = json.load(f)

        df_comparison = pd.read_csv(comparison_path)
        df_pred_dist  = pd.read_csv(pred_dist_path)
        return summary, df_comparison, df_pred_dist

    try:
        ml_summary, df_comparison, df_pred_dist = load_ml_data()
        ml_loaded = True
    except FileNotFoundError:
        st.warning("⚠️ ML output files not found. Run `cuisine_classifier.py` first.")
        ml_loaded = False

    if ml_loaded:

        # ── SECTION 1: PIPELINE OVERVIEW METRICS ─────────────────────
        st.subheader(":material/inventory_2: Data Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews",         f"{ml_summary['total_reviews_loaded']:,}")
        col2.metric("Known Cuisine (Train)",  f"{ml_summary['known_cuisine_reviews']:,}")
        col3.metric("Unknown Cuisine",        f"{ml_summary['unknown_cuisine_reviews']:,}")
        col4.metric("Reviews Recovered",      f"{ml_summary['predictions_made']:,}")

        st.write("---")

        # ── SECTION 2: MODEL COMPARISON ───────────────────────────────
        st.subheader(":material/emoji_events: Model Comparison")
        st.caption("All three models were evaluated on a held-out 20% test set across all cuisine classes. The best model was selected by weighted F1 score.")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            # Bar chart: accuracy vs F1 for each model
            fig_compare = go.Figure()

            fig_compare.add_trace(go.Bar(
                name='Accuracy',
                x=df_comparison['model'],
                y=(df_comparison['accuracy'] * 100).round(1),
                marker_color='#3498db',
                text=(df_comparison['accuracy'] * 100).round(1).astype(str) + '%',
                textposition='outside'
            ))
            fig_compare.add_trace(go.Bar(
                name='Weighted F1',
                x=df_comparison['model'],
                y=(df_comparison['weighted_f1'] * 100).round(1),
                marker_color='#2ecc71',
                text=(df_comparison['weighted_f1'] * 100).round(1).astype(str) + '%',
                textposition='outside'
            ))

            fig_compare.update_layout(
                barmode='group',
                title='Accuracy vs Weighted F1 by Model',
                yaxis=dict(title='Score (%)', range=[0, 100]),
                xaxis_title='Model',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                height=400
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        with col_right:
            # Detailed metrics table
            display_df = df_comparison[['model', 'accuracy', 'weighted_f1', 'weighted_precision', 'weighted_recall', 'is_best']].copy()
            display_df.columns = ['Model', 'Accuracy', 'Weighted F1', 'Precision', 'Recall', 'Best']
            display_df['Accuracy']    = (display_df['Accuracy'] * 100).round(1).astype(str) + '%'
            display_df['Weighted F1'] = (display_df['Weighted F1'] * 100).round(1).astype(str) + '%'
            display_df['Precision']   = (display_df['Precision'] * 100).round(1).astype(str) + '%'
            display_df['Recall']      = (display_df['Recall'] * 100).round(1).astype(str) + '%'
            display_df['Best']        = display_df['Best'].apply(lambda x: '✅' if x else '')

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            best = ml_summary['best_model']
            acc  = ml_summary['best_model_accuracy'] * 100
            f1   = ml_summary['best_model_f1']
            st.success(f"**Best model: {best}** — {acc:.1f}% accuracy, F1: {f1:.4f}")

        st.write("---")

        # ── SECTION 3: CONFUSION ANALYSIS ────────────────────────────
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
            st.warning(
                "**Lebanese over-prediction:** Lebanese cuisine dominates the training data "
                "(5,112 samples vs ~145 for Japanese). The model defaults to Lebanese when "
                "uncertain. This is expected in imbalanced multiclass problems."
            )

        st.write("---")

        # ── SECTION 4: PER-CLASS F1 BREAKDOWN ────────────────────────
        st.subheader(":material/bar_chart: Per-Class F1 Score Breakdown")
        st.caption(f"Performance of {ml_summary['best_model']} broken down by cuisine class. Classes with 0.0 F1 were never correctly predicted.")

        per_class = ml_summary['per_class_f1']
        df_f1 = pd.DataFrame(
            list(per_class.items()),
            columns=['Cuisine', 'F1 Score']
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
            title=f'F1 Score per Cuisine — {ml_summary["best_model"]}',
            xaxis=dict(title='F1 Score', range=[0, 1]),
            yaxis_title='Cuisine',
            height=650
        )
        st.plotly_chart(fig_f1, use_container_width=True)

        st.write("---")

        # ── SECTION 5: PREDICTION OUTCOMES ───────────────────────────
        st.subheader(":material/auto_awesome: Prediction Outcomes on Unknown Reviews")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Avg Confidence Score",     f"{ml_summary['avg_prediction_confidence']:.2f}")
        col_m2.metric("Low-Confidence Predictions",
                      f"{ml_summary['low_confidence_count']:,}",
                      f"{ml_summary['low_confidence_pct']}% of total",
                      delta_color="inverse")
        col_m3.metric("Confidence Threshold",     f"{ml_summary['low_confidence_threshold']}")

        col_pred_l, col_pred_r = st.columns([1.2, 1])

        with col_pred_l:
            # Bar chart of predicted cuisine distribution
            fig_dist = px.bar(
                df_pred_dist,
                x='predicted_count',
                y='cuisine',
                orientation='h',
                color='avg_confidence',
                color_continuous_scale='RdYlGn',
                range_color=[0, 1],
                labels={
                    'predicted_count': 'Reviews Predicted',
                    'cuisine': 'Cuisine',
                    'avg_confidence': 'Avg Confidence'
                },
                title='Predicted Cuisine Distribution (17k Recovered Reviews)'
            )
            fig_dist.update_layout(
                height=550,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_pred_r:
            st.markdown("**Recovered Reviews by Cuisine**")
            display_pred = df_pred_dist[['cuisine', 'predicted_count', 'avg_confidence', 'high_conf_count']].copy()
            display_pred.columns = ['Cuisine', 'Predicted', 'Avg Conf', 'High Conf (≥0.5)']
            display_pred['Avg Conf'] = display_pred['Avg Conf'].round(3)
            st.dataframe(display_pred, use_container_width=True, hide_index=True, height=500)

        st.write("---")

        # ── SECTION 6: ENRICHED DATASET SUMMARY ──────────────────────
        st.subheader(":material/check_circle: Enriched Dataset")
        st.write("The classifier output was merged back into `master_reviews_enriched.csv`. A `cuisine_source` column tracks which labels are original vs ML-predicted.")

        col_e1, col_e2, col_e3 = st.columns(3)
        col_e1.metric("Total Reviews (Enriched)",  f"{ml_summary['enriched_reviews_total']:,}")
        col_e2.metric("Original Labels",            f"{ml_summary['enriched_original_labels']:,}")
        col_e3.metric("ML-Predicted Labels",        f"{ml_summary['enriched_predicted_labels']:,}")

        # Donut chart: original vs predicted
        fig_donut = go.Figure(go.Pie(
            labels=['Original Metadata', 'ML-Predicted'],
            values=[ml_summary['enriched_original_labels'], ml_summary['enriched_predicted_labels']],
            hole=0.55,
            marker_colors=['#2ecc71', '#3498db']
        ))
        fig_donut.update_layout(
            title='Label Source Breakdown in Enriched Dataset',
            height=350
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        st.caption(
            "⚠️ ML-predicted labels should be interpreted with caution. "
            "Low-confidence predictions (< 0.30) are still included but flagged via the `prediction_confidence` column. "
            "The post-ML NLP section uses this enriched file."
        )
        
        st.write("---")

        # ── SECTION 7: ENRICHED REVIEWS TABLE ────────────────────────
        st.subheader(":material/table_rows: Enriched Reviews Dataset")
        st.write("Browse the full enriched reviews file. Filter to inspect only the rows where cuisine was filled in by the ML classifier.")

        @st.cache_data
        def load_enriched_reviews():
            return pd.read_csv(os.path.join(ML_DIR, 'master_reviews_enriched.csv'))

        try:
            df_enriched = load_enriched_reviews()

            # ── FILTER CONTROLS ──────────────────────────────────────
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
                    min_conf = st.slider(
                        "Min Confidence (ML rows only):",
                        min_value=0.0, max_value=1.0, value=0.0, step=0.05
                    )
                else:
                    min_conf = 0.0

            # ── APPLY FILTERS ────────────────────────────────────────
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
                # only apply confidence filter to ML rows; original rows have NaN confidence
                ml_mask      = filtered['cuisine_source'] == 'predicted'
                orig_mask    = filtered['cuisine_source'] == 'original'
                filtered = pd.concat([
                    filtered[ml_mask & (filtered['prediction_confidence'] >= min_conf)],
                    filtered[orig_mask]
                ]).sort_index()

            # ── SUMMARY METRICS ──────────────────────────────────────
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Showing",          f"{len(filtered):,} reviews")
            col_m2.metric("ML-Predicted",     f"{(filtered['cuisine_source'] == 'predicted').sum():,}")
            col_m3.metric("Original Labels",  f"{(filtered['cuisine_source'] == 'original').sum():,}")
            if 'prediction_confidence' in filtered.columns:
                avg_conf = filtered[filtered['cuisine_source'] == 'predicted']['prediction_confidence'].mean()
                col_m4.metric("Avg Confidence (ML)", f"{avg_conf:.3f}" if not pd.isna(avg_conf) else "—")

            # ── TABLE ────────────────────────────────────────────────
            display_cols = [
                'restaurant_name', 'cuisine_primary', 'cuisine_source',
                'prediction_confidence', 'area', 'price_category',
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
                        help="'predicted' = filled in by ML classifier | 'original' = came from source data"
                    ),
                    "prediction_confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        help="ML confidence score (only for predicted rows)",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f"
                    ),
                    "sentiment_score": st.column_config.NumberColumn(
                        "Sentiment", format="%.3f"
                    ),
                    "review_text": st.column_config.TextColumn(
                        "Review Text", width="large"
                    )
                }
            )

        except FileNotFoundError:
            st.warning("⚠️ `master_reviews_enriched.csv` not found in the ML directory. Run `cuisine_classifier.py` first.")
        
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

    # ── TABS ────────────────────────────────────────────────────────
    tab_before, tab_after = st.tabs([" Before ML — Original Reviews", " After ML — Enriched Reviews"])

    # ── HELPER: renders all NLP charts for a given dataset ──────────
    def render_nlp_tab(s_area, s_cuisine, s_price, a_kw, c_kw, summary, tag):
        """
        tag: short string appended to selectbox keys to avoid duplicate widget IDs
        """

        # SUMMARY STATS
        st.subheader(":material/assignment: Review Dataset Summary")
        if summary:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Reviews Loaded",       f"{summary['total_reviews_loaded']:,}")
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

            with st.expander("📊 Reviews by Source"):
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

        with st.expander("📄 View full sentiment-by-area data"):
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

        with st.expander("📄 View full sentiment-by-cuisine data"):
            st.dataframe(s_cuisine, use_container_width=True)

        st.write("---")

        # SENTIMENT BY PRICE
        st.subheader(":material/payments: Sentiment by Price Tier")
        color_price = ['#2ecc71' if x > 0 else '#e74c3c' for x in s_price['avg_sentiment']]
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

        with st.expander("📄 View full sentiment-by-price data"):
            st.dataframe(s_price, use_container_width=True)

        st.write("---")

        # TF-IDF KEYWORDS
        st.subheader(":material/key: Top Keywords by Neighborhood (TF-IDF)")
        st.write("Words that uniquely characterize each neighborhood's reviews compared to all others.")
        selected_area_kw = st.selectbox("Select a neighborhood:", sorted(a_kw.keys()), key=f"kw_area_{tag}")
        if selected_area_kw:
            kw_df = pd.DataFrame({"Rank": range(1, 11), "Keyword": a_kw[selected_area_kw]})
            st.dataframe(kw_df, use_container_width=True, hide_index=True)

        st.write("---")

        st.subheader(":material/key: Top Keywords by Cuisine (TF-IDF)")
        st.write("Words that uniquely characterize each cuisine type's reviews.")
        selected_cuisine_kw = st.selectbox("Select a cuisine:", sorted(c_kw.keys()), key=f"kw_cuisine_{tag}")
        if selected_cuisine_kw:
            kw_df = pd.DataFrame({"Rank": range(1, 11), "Keyword": c_kw[selected_cuisine_kw]})
            st.dataframe(kw_df, use_container_width=True, hide_index=True)

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



# FOOTER

st.write("---")
st.write("**COSC 482 - Data Science Project** | Lebanese Restaurant Analysis")
