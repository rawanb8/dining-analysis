import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Beirut Restaurant Analysis",
    page_icon="🍽️",
    layout="wide"
)

# LOAD DATA

@st.cache_data
def load_data():
    restaurants = pd.read_csv(r"C:\Users\Msi\Desktop\RHU\Spring 2026\DatasSience and WebScraping\Datascience Project\dining-analysis\merged\master_restaurants.csv")
    reviews = pd.read_csv(r"C:\Users\Msi\Desktop\RHU\Spring 2026\DatasSience and WebScraping\Datascience Project\dining-analysis\merged\master_reviews.csv")
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

st.title("🍽️ Beirut Restaurant Analysis Dashboard")
st.write("COSC 482 - Data Science Project | Interactive exploration of Beirut restaurants")
st.write("---")

# SIDEBAR - NAVIGATION

st.sidebar.title("📊 Dashboard Sections")
selected_section = st.sidebar.radio(
    "Select Section:",
    ["🔍 Search & Filter", "📊 General Analysis", "✨ Feature Analysis", 
     "💬 NLP Analysis", "🤖 ML Insights"]
)

st.sidebar.write("---")

# SECTION 1: SEARCH & FILTER

if selected_section == "🔍 Search & Filter":
    st.header("🔍 Search and Filter Restaurants")
    
    # Filters in sidebar
    st.sidebar.subheader("Filters")
    
    # Text search
    search_name = st.sidebar.text_input("🔎 Search by Name:").strip().lower()
    
    # Cuisine filter
    cuisine_options = ["All Cuisines"] + sorted(df_restaurants['cuisine_primary'].unique().tolist())
    selected_cuisine = st.sidebar.selectbox("🍽️ Cuisine Type:", cuisine_options)
    
    # Area filter
    area_options = ["All Areas"] + sorted(df_restaurants['area'].unique().tolist())
    selected_area = st.sidebar.selectbox("📍 Area:", area_options)
    
    # Price filter
    price_options = ["All Prices", "Budget", "Mid-Range", "High-End"]
    selected_price = st.sidebar.selectbox("💰 Price Category:", price_options)
    
    # Rating filter
    min_rating = st.sidebar.slider(
        "⭐ Minimum Rating:",
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
    st.subheader(f"📊 Showing {len(filtered_df)} of {len(df_display)} restaurants")
    
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
    st.subheader("🏆 Top 10 Highest Rated")
    top_rated = filtered_df.nlargest(10, 'rating_overall')[
        ['name', 'rating_overall', 'review_count_total', 'area', 'cuisine_primary', 'price_category']
    ]
    st.dataframe(top_rated, use_container_width=True)
    
    # Expandable full dataset
    with st.expander(f"📋 Click to view full filtered dataset ({len(filtered_df)} records)"):
        st.dataframe(filtered_df, use_container_width=True)

# SECTION 2: GENERAL ANALYSIS (Placeholder for Friend 1)

elif selected_section == "📊 General Analysis":
    st.header("📊 General Analysis (EDA)")

    
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
    st.subheader("📈 Rating vs Review Count")

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
    st.subheader("💎 Hidden Gems (High Rating, Low Reviews)")

    hidden = df_restaurants[
        (df_restaurants["rating_overall"] >= 4.5) &
        (df_restaurants["review_count_total"] < 50)
    ].head(10)[
        ["name", "rating_overall", "review_count_total", "area"]
    ]

    st.dataframe(hidden, use_container_width=True)





    st.write("---")
    st.subheader("📊 Density vs Quality by Area")
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

    st.subheader("🏆 Best Areas Overall (Combined Score)")

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

elif selected_section == "✨ Feature Analysis":
    st.header("✨ Feature Analysis")
    st.write("Analysis of 12 features extracted from restaurant descriptions and reviews")
    
    # Calculate feature statistics
    feature_stats = {}
    for feature in feature_cols:
        count = (df_restaurants[feature] == 'TRUE').sum()
        pct = (count / len(df_restaurants)) * 100
        feature_stats[feature.replace('_', ' ').title()] = {'count': count, 'percent': pct}
    
    # Summary metrics
    st.subheader("📊 Feature Summary")
    
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
    st.subheader("⭐ Star Distribution Analysis")
    
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
    st.subheader("📈 Data Quality Analysis")
    
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
    feature_matrix = df_restaurants[feature_cols].applymap(lambda x: 1 if x == 'TRUE' else 0)
    
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
    
    # Feature Insights
    st.subheader("💡 Key Insights")
    
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

# SECTION 4: NLP ANALYSIS (Placeholder for Friend 2)

elif selected_section == "💬 NLP Analysis":
    st.header("💬 NLP Analysis")
    st.info("⚠️ This section will be populated by Friend 2 (NLP)")
    
    st.write("---")
    st.subheader("Planned Analysis:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sentiment Analysis:**")
        st.write("- Sentiment distribution (Positive/Neutral/Negative)")
        st.write("- Sentiment by cuisine type")
        st.write("- Sentiment by price category")
        
    with col2:
        st.write("**Text Analysis:**")
        st.write("- Word clouds (positive/negative)")
        st.write("- Theme extraction (food, service, ambiance)")
        st.write("- Review text statistics")

# SECTION 5: ML INSIGHTS (Placeholder for Friend 3)

elif selected_section == "🤖 ML Insights":
    st.header("🤖 Machine Learning Insights")
    st.info("⚠️ This section will be populated by Friend 3 (ML)")
    
    st.write("---")
    st.subheader("Planned Models:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Classification Models:**")
        st.write("- Fake review detection")
        st.write("- Restaurant clustering")
        st.write("- Rating prediction")
        
    with col2:
        st.write("**Recommendation System:**")
        st.write("- Content-based filtering")
        st.write("- Collaborative filtering")
        st.write("- Hybrid recommendations")

# FOOTER

st.write("---")
st.write("**COSC 482 - Data Science Project** | Beirut Restaurant Analysis")
st.write("Team: Your Name, Friend 1 (EDA), Friend 2 (NLP), Friend 3 (ML)")