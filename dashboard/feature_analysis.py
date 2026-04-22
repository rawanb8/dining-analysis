import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render_feature_analysis(df_restaurants, feature_cols):
    st.header(":material/auto_awesome: Feature Analysis")
    st.write("Analysis of 12 features extracted from restaurant descriptions and reviews")

    # SIDEBAR FILTERS
    st.sidebar.subheader("Feature Analysis Filters")

    all_cuisines = sorted([str(c) for c in df_restaurants['cuisine_primary'].dropna().unique() if c != 'Unknown'])
    all_areas    = sorted([str(a) for a in df_restaurants['area'].dropna().unique()           if a != 'Unknown'])
    all_prices   = ['Budget', 'Mid-Range', 'High-End']

    filter_cuisine = st.sidebar.multiselect("Filter by Cuisine:", all_cuisines, key="feat_cuisine")
    filter_area    = st.sidebar.multiselect("Filter by Area:",    all_areas,    key="feat_area")
    filter_price   = st.sidebar.multiselect("Filter by Price:",   all_prices,   key="feat_price")

    # Apply filters
    df_feat = df_restaurants.copy()
    if filter_cuisine:
        df_feat = df_feat[df_feat['cuisine_primary'].isin(filter_cuisine)]
    if filter_area:
        df_feat = df_feat[df_feat['area'].isin(filter_area)]
    if filter_price:
        df_feat = df_feat[df_feat['price_category'].isin(filter_price)]

    active_filters = []
    if filter_cuisine: active_filters.append(f"Cuisine: {len(filter_cuisine)} selected")
    if filter_area:    active_filters.append(f"Area: {len(filter_area)} selected")
    if filter_price:   active_filters.append(f"Price: {len(filter_price)} selected")

    if active_filters:
        st.info(f"Active filters: {' · '.join(active_filters)} — showing **{len(df_feat)}** of **{len(df_restaurants)}** restaurants")

    st.write("---")

    # KPI CARDS
    geocoded_count = df_feat[df_feat['latitude'].notna()]['latitude'].count() if 'latitude' in df_feat.columns else 0
    geocoded_pct   = (geocoded_count / len(df_feat) * 100) if len(df_feat) > 0 else 0
    known_cuisine  = df_feat[df_feat['cuisine_primary'] != 'Unknown']['cuisine_primary']
    most_common_cuisine = known_cuisine.value_counts().index[0] if len(known_cuisine) > 0 else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Restaurants", f"{len(df_feat):,}")
    col2.metric("Avg Rating", f"{df_feat['rating_overall'].mean():.2f}⭐" if len(df_feat) > 0 else "—")
    col3.metric("Geocoded", f"{geocoded_count:,} ({geocoded_pct:.1f}%)")
    col4.metric("Top Cuisine", most_common_cuisine)

    st.write("---")

    # CUISINE PIE CHART
    st.subheader(":material/pie_chart: Top 10 Cuisines Distribution")
    top_10_cuisines = df_feat[df_feat['cuisine_primary'] != 'Unknown']['cuisine_primary'].value_counts().head(10)
    if len(top_10_cuisines) > 0:
        fig_pie = px.pie(
            values=top_10_cuisines.values,
            names=top_10_cuisines.index,
            title='Top 10 Cuisines',
            hole=0.3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No cuisine data to display after filtering.")

    st.write("---")

    # RATING HISTOGRAM
    st.subheader(":material/bar_chart: Rating Distribution")
    if len(df_feat) > 0:
        fig_hist = px.histogram(
            df_feat,
            x='rating_overall',
            nbins=20,
            title='Rating Distribution',
            labels={'rating_overall': 'Rating', 'count': 'Number of Restaurants'},
            color_discrete_sequence=['#3498db']
        )
        fig_hist.update_layout(yaxis_title="Number of Restaurants")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No rating data to display after filtering.")

    st.write("---")

    # Feature statistics
    feature_stats = {}
    for feature in feature_cols:
        count = (df_feat[feature] == 'TRUE').sum()
        pct   = (count / len(df_feat)) * 100 if len(df_feat) > 0 else 0
        feature_stats[feature.replace('_', ' ').title()] = {'count': count, 'percent': pct}

    # FEATURE SUMMARY TABLE
    st.subheader(":material/table_chart: Feature Summary Table")
    feature_summary_data = [
        {'Feature': name, 'Count': stats['count'], 'Percentage': f"{stats['percent']:.1f}%"}
        for name, stats in feature_stats.items()
    ]
    feature_summary = pd.DataFrame(feature_summary_data).sort_values('Count', ascending=False)
    st.dataframe(feature_summary, use_container_width=True, hide_index=True)

    st.write("---")

    # FEATURE SUMMARY METRICS
    st.subheader(":material/bar_chart: Feature Summary")
    col1, col2, col3, col4 = st.columns(4)

    outdoor_count = (df_feat['outdoor_seating'] == 'TRUE').sum()
    outdoor_pct   = (outdoor_count / len(df_feat)) * 100 if len(df_feat) > 0 else 0
    avg_features  = sum([v['count'] for v in feature_stats.values()]) / len(df_feat) if len(df_feat) > 0 else 0

    col1.metric("Most Common Feature", "Outdoor Seating", f"{outdoor_pct:.0f}%")
    col2.metric("Total Restaurants", len(df_feat))
    col3.metric("Features Detected", "12 types")
    col4.metric("Avg Features/Restaurant", f"{avg_features:.1f}")

    st.write("---")

    # Chart 1: Feature Availability
    st.subheader("1. Feature Availability Overview")
    feature_names  = list(feature_stats.keys())
    feature_counts = [feature_stats[f]['count']   for f in feature_names]
    feature_pcts   = [feature_stats[f]['percent'] for f in feature_names]

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
        st.subheader("2. Features by Top Areas")
        top_areas = df_feat['area'].value_counts().head(10).index.tolist()
        area_data = []
        for area in top_areas:
            area_df = df_feat[df_feat['area'] == area]
            for feature in ['outdoor_seating', 'parking_available', 'wifi_available', 'live_music']:
                count = (area_df[feature] == 'TRUE').sum()
                pct   = (count / len(area_df) * 100) if len(area_df) > 0 else 0
                area_data.append({'Area': area, 'Feature': feature.replace('_', ' ').title(), 'Percentage': pct})
        fig2 = px.bar(
            pd.DataFrame(area_data), x='Area', y='Percentage', color='Feature',
            title='Top Features by Area (Top 10 Areas)',
            labels={'Percentage': 'Percentage (%)'}, barmode='group', height=450
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("3. Features by Cuisine")
        top_cuisines = df_feat[df_feat['cuisine_primary'] != 'Unknown']['cuisine_primary'].value_counts().head(8).index.tolist()
        cuisine_data = []
        for cuisine in top_cuisines:
            cuisine_df = df_feat[df_feat['cuisine_primary'] == cuisine]
            for feature in ['outdoor_seating', 'delivery_available', 'reservation_required', 'live_music']:
                count = (cuisine_df[feature] == 'TRUE').sum()
                pct   = (count / len(cuisine_df) * 100) if len(cuisine_df) > 0 else 0
                cuisine_data.append({'Cuisine': cuisine, 'Feature': feature.replace('_', ' ').title(), 'Percentage': pct})
        fig3 = px.bar(
            pd.DataFrame(cuisine_data), x='Cuisine', y='Percentage', color='Feature',
            title='Top Features by Cuisine (Top 8)',
            labels={'Percentage': 'Percentage (%)'}, barmode='group', height=450
        )
        fig3.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)

    st.write("---")

    # Chart 4: Features by Price
    st.subheader("4. Features by Price Category")
    price_categories  = ['Budget', 'Mid-Range', 'High-End']
    price_feature_data = []
    for price in price_categories:
        price_df = df_feat[df_feat['price_category'] == price]
        if len(price_df) == 0:
            continue
        for feature in ['outdoor_seating', 'parking_available', 'wifi_available', 'live_music', 'delivery_available']:
            count = (price_df[feature] == 'TRUE').sum()
            pct   = (count / len(price_df) * 100)
            price_feature_data.append({'Price': price, 'Feature': feature.replace('_', ' ').title(), 'Percentage': pct})
    fig4 = px.bar(
        pd.DataFrame(price_feature_data), x='Feature', y='Percentage', color='Price',
        title='Feature Availability by Price Category',
        labels={'Percentage': 'Percentage (%)'},
        barmode='group', height=500,
        color_discrete_map={'Budget': '#2ecc71', 'Mid-Range': '#3498db', 'High-End': '#e74c3c'}
    )
    fig4.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig4, use_container_width=True)

    st.write("---")

    # Star Distribution Analysis
    st.subheader(":material/star: Star Distribution Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("5. Average Star Distribution")
        star_averages = [
            df_feat['star_5_percent'].mean(),
            df_feat['star_4_percent'].mean(),
            df_feat['star_3_percent'].mean(),
            df_feat['star_2_percent'].mean(),
            df_feat['star_1_percent'].mean()
        ]
        fig5 = go.Figure(data=[go.Bar(
            x=['5 Stars', '4 Stars', '3 Stars', '2 Stars', '1 Star'],
            y=star_averages,
            text=[f"{avg:.1f}%" for avg in star_averages],
            textposition='outside',
            marker_color=['#27ae60', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
        )])
        fig5.update_layout(
            title='Average Star Distribution Across All Restaurants',
            xaxis_title='Rating', yaxis_title='Average Percentage (%)',
            height=450, showlegend=False
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader("6. Star Distribution by Price")
        star_cols_list = ['star_5_percent', 'star_4_percent', 'star_3_percent', 'star_2_percent', 'star_1_percent']
        star_labels    = ['5 Stars', '4 Stars', '3 Stars', '2 Stars', '1 Star']
        price_star_data = []
        for price in price_categories:
            price_df = df_feat[df_feat['price_category'] == price]
            if len(price_df) == 0:
                continue
            for i, star_col in enumerate(star_cols_list):
                price_star_data.append({'Price': price, 'Stars': star_labels[i], 'Percentage': price_df[star_col].mean()})
        fig6 = px.bar(
            pd.DataFrame(price_star_data), x='Stars', y='Percentage', color='Price',
            title='Star Distribution by Price Category',
            labels={'Percentage': 'Average Percentage (%)'},
            barmode='group', height=450,
            color_discrete_map={'Budget': '#2ecc71', 'Mid-Range': '#3498db', 'High-End': '#e74c3c'}
        )
        st.plotly_chart(fig6, use_container_width=True)

    st.write("---")

    # Data Quality Analysis
    st.subheader(":material/monitoring: Data Quality Analysis")
    sources       = ['source1', 'source2', 'source3']
    source_labels = ['Wandorlog', 'Guru', 'TripAdvisor']
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("7. Data Completeness by Source")
        completeness_data = []
        for i, source in enumerate(sources):
            source_df = df_feat[df_feat['data_source'] == source]
            if len(source_df) == 0:
                continue
            for field, mask in [
                ('Phone',   source_df['phone'].notna()),
                ('Cuisine', source_df['cuisine_primary'] != 'Unknown'),
                ('Hours',   source_df['hours_monday'] != 'Unknown'),
                ('Price',   source_df['price_category'] != 'Unknown'),
            ]:
                completeness_data.append({
                    'Source': source_labels[i],
                    'Field': field,
                    'Completeness': (mask.sum() / len(source_df)) * 100
                })
        fig7 = px.bar(
            pd.DataFrame(completeness_data), x='Source', y='Completeness', color='Field',
            title='Data Completeness by Source',
            labels={'Completeness': 'Completeness (%)'}, barmode='group', height=450
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        st.subheader("8. Feature Detection by Source")
        source_feature_data = []
        for i, source in enumerate(sources):
            source_df = df_feat[df_feat['data_source'] == source]
            if len(source_df) == 0:
                continue
            for feature in ['outdoor_seating', 'parking_available', 'wifi_available', 'live_music', 'delivery_available']:
                count = (source_df[feature] == 'TRUE').sum()
                pct   = (count / len(source_df)) * 100
                source_feature_data.append({
                    'Source': source_labels[i],
                    'Feature': feature.replace('_', ' ').title(),
                    'Detection Rate': pct
                })
        fig8 = px.bar(
            pd.DataFrame(source_feature_data), x='Feature', y='Detection Rate', color='Source',
            title='Feature Detection Rate by Source',
            labels={'Detection Rate': 'Detection Rate (%)'}, barmode='group', height=450
        )
        fig8.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig8, use_container_width=True)

    st.write("---")

    # Chart 9: Feature Correlation Heatmap
    st.subheader("9. Feature Correlation Heatmap")
    feature_matrix = df_feat[feature_cols].map(lambda x: 1 if x == 'TRUE' else 0)
    correlation    = feature_matrix.corr()
    fig9 = go.Figure(data=go.Heatmap(
        z=correlation.values,
        x=[f.replace('_', ' ').title() for f in feature_cols],
        y=[f.replace('_', ' ').title() for f in feature_cols],
        colorscale='RdBu', zmid=0,
        text=correlation.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 9},
        colorbar=dict(title="Correlation")
    ))
    fig9.update_layout(title='Feature Correlation Matrix', height=700, xaxis_tickangle=-45)
    st.plotly_chart(fig9, use_container_width=True)

    # Correlation insights
    st.write("**Key Correlation Insights:**")
    corr_matrix        = correlation.copy()
    mask               = np.triu(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix_masked = corr_matrix.where(~mask)
    corr_unstacked     = corr_matrix_masked.unstack()
    top_positive       = corr_unstacked[corr_unstacked > 0.40].sort_values(ascending=False).head(4)
    top_negative       = corr_unstacked[corr_unstacked < -0.30].sort_values().head(2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strong Positive Correlations (>0.40):**")
        for (feat1, feat2), val in top_positive.items():
            st.write(f"• {feat1.replace('_',' ').title()} ↔ {feat2.replace('_',' ').title()} ({val:.2f})")
    with col2:
        st.markdown("**Strong Negative Correlations (<-0.30):**")
        for (feat1, feat2), val in top_negative.items():
            st.write(f"• {feat1.replace('_',' ').title()} ↔ {feat2.replace('_',' ').title()} ({val:.2f})")

    st.markdown("**Service Model Clusters:**")
    st.write("1- **Dine-In Premium:** Outdoor + Parking + Live Music + Reservations")
    st.write("2- **Modern Convenience:** Delivery + Takeaway + WiFi + Credit Cards")
    st.write("3- **Traditional Budget:** Cash Only + Limited amenities")
    st.write("---")

    # Geographic Maps
    if 'latitude' in df_feat.columns and 'longitude' in df_feat.columns:
        geocoded_restaurants = df_feat[df_feat['latitude'].notna() & df_feat['longitude'].notna()]

        if len(geocoded_restaurants) > 0:
            st.subheader(":material/map: Geographic Distribution")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**All Restaurants by Price**")
                map_price = px.scatter_mapbox(
                    geocoded_restaurants, lat='latitude', lon='longitude',
                    hover_name='name', hover_data=['rating_overall', 'area'],
                    color='price_category',
                    color_discrete_map={'Budget': '#2ecc71', 'Mid-Range': '#3498db', 'High-End': '#e74c3c'},
                    zoom=11, height=500
                )
                map_price.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(map_price, use_container_width=True)

            with col2:
                st.write("**Top Rated (4.5+)**")
                top_rated_geo = geocoded_restaurants[geocoded_restaurants['rating_overall'] >= 4.5]
                if len(top_rated_geo) > 0:
                    map_top = px.scatter_mapbox(
                        top_rated_geo, lat='latitude', lon='longitude',
                        hover_name='name', hover_data=['rating_overall', 'area', 'cuisine_primary'],
                        color='rating_overall', size='rating_overall',
                        color_continuous_scale='RdYlGn', zoom=11, height=500
                    )
                    map_top.update_layout(mapbox_style="open-street-map")
                    st.plotly_chart(map_top, use_container_width=True)

            st.write("**Restaurant Density by Area**")
            area_counts   = geocoded_restaurants['area'].value_counts().head(15).reset_index()
            area_counts.columns = ['Area', 'Count']
            area_coords   = geocoded_restaurants.groupby('area').agg({'latitude': 'mean', 'longitude': 'mean'}).reset_index()
            area_map_data = area_counts.merge(area_coords, left_on='Area', right_on='area')
            map_density   = px.scatter_mapbox(
                area_map_data, lat='latitude', lon='longitude',
                size='Count', hover_name='Area', hover_data=['Count'],
                color='Count', color_continuous_scale='YlOrRd', zoom=11, height=500
            )
            map_density.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(map_density, use_container_width=True)
            st.info(f"{len(geocoded_restaurants)} of {len(df_feat)} restaurants geocoded")

        st.write("---")

    # Key Insights
    st.subheader(":material/lightbulb: Key Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Most Common Features:**")
        top_features = sorted(feature_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
        for i, (feature, stats) in enumerate(top_features, 1):
            st.write(f"{i}. {feature} ({stats['percent']:.1f}%)")

    with col2:
        st.write("**High-End Trends:**")
        high_end_df = df_feat[df_feat['price_category'] == 'High-End']
        if len(high_end_df) > 0:
            parking_pct = (high_end_df['parking_available'] == 'TRUE').sum() / len(high_end_df) * 100
            music_pct   = (high_end_df['live_music'] == 'TRUE').sum()         / len(high_end_df) * 100
            reserv_pct  = (high_end_df['reservation_required'] == 'TRUE').sum() / len(high_end_df) * 100
            st.write(f"- {parking_pct:.0f}% have parking")
            st.write(f"- {music_pct:.0f}% have live music")
            st.write(f"- {reserv_pct:.0f}% require reservations")

    with col3:
        st.write("**Data Quality:**")
        for i, source in enumerate(sources):
            source_df = df_feat[df_feat['data_source'] == source]
            if len(source_df) > 0:
                completeness = (
                    source_df['phone'].notna().sum() +
                    (source_df['cuisine_primary'] != 'Unknown').sum()
                ) / (2 * len(source_df)) * 100
                st.write(f"- {source_labels[i]}: {completeness:.0f}% complete")
