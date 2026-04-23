import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report


FEATURE_COLS = [
    'delivery_available', 'outdoor_seating', 'reservation_required',
    'cash_only', 'credit_cards_accepted', 'wifi_available',
    'wheelchair_accessible', 'takeaway_available', 'parking_available',
    'live_music', 'pet_friendly', 'kids_friendly'
]

# Features strongly associated with high-end dining
LUXURY_FEATURES = [
    'outdoor_seating', 'parking_available', 'live_music',
    'reservation_required', 'credit_cards_accepted', 'wifi_available'
]

# Features associated with casual / budget-friendly service
CONVENIENCE_FEATURES = [
    'delivery_available', 'takeaway_available', 'wifi_available', 'credit_cards_accepted'
]

STAR_COLS = ['star_5_percent', 'star_4_percent', 'star_3_percent', 'star_2_percent', 'star_1_percent']

PRICE_COLORS = {'Budget': '#2ecc71', 'Mid-Range': '#3498db', 'High-End': '#e74c3c'}

# Human-readable names for every feature in the model
FEATURE_DISPLAY = {
    'cuisine_enc':        'Cuisine Type',
    'area_enc':           'Area / Location',
    'rating_overall':     'Overall Rating',
    'log_reviews':        'Popularity (log reviews)',
    'feature_count':      'Total Amenities',
    'luxury_score':       'Luxury Score',
    'convenience_score':  'Convenience Score',
    'star_high_pct':      'High-Star % (4★ + 5★)',
    'star_low_pct':       'Low-Star % (1★ + 2★)',
    'rating_x_luxury':    'Rating × Luxury (interaction)',
    'star_5_percent':     '5★ Review %',
    'star_4_percent':     '4★ Review %',
    'star_3_percent':     '3★ Review %',
    'star_2_percent':     '2★ Review %',
    'star_1_percent':     '1★ Review %',
}


def _smart_impute_rating(df):
    """
    Fill missing rating_overall using progressively broader group means.
    Always operates on a copy — never touches the original dataframe.
    """
    df = df.copy()
    groups = [
        ['area', 'cuisine_primary', 'price_category'],  # most specific
        ['area', 'price_category'],
        ['cuisine_primary', 'price_category'],
        ['price_category'],                              # broadest fallback
    ]
    for group_cols in groups:
        group_means = df.groupby(group_cols)['rating_overall'].transform('mean')
        missing = df['rating_overall'].isna()
        df.loc[missing, 'rating_overall'] = group_means[missing]

    # Final safety net: global mean
    df['rating_overall'] = df['rating_overall'].fillna(df['rating_overall'].mean())
    return df


def _engineer_features(df):
    """
    Add all derived features used by the model.
    Expects df to already be a copy.
    """
    # Binary encode the 12 amenity columns
    for col in FEATURE_COLS:
        df[col + '_bin'] = (df[col] == 'TRUE').astype(int)

    # Aggregate amenity signals
    df['feature_count']     = sum(df[f + '_bin'] for f in FEATURE_COLS)
    df['luxury_score']      = sum(df[f + '_bin'] for f in LUXURY_FEATURES)
    df['convenience_score'] = sum(df[f + '_bin'] for f in CONVENIENCE_FEATURES)

    # Log-transform review count — the raw values span 0 to 11,908
    # log(1 + x) keeps 0 → 0 and compresses the extreme skew
    df['log_reviews'] = np.log1p(df['review_count_total'].fillna(0))

    # Star distribution quality signals (all star columns are 100% complete)
    df['star_high_pct'] = df['star_5_percent'] + df['star_4_percent']
    df['star_low_pct']  = df['star_1_percent'] + df['star_2_percent']

    # Interaction: a restaurant that is BOTH highly rated AND luxury-amenity-rich
    # is a much stronger signal for High-End than either feature alone
    df['rating_x_luxury'] = df['rating_overall'] * df['luxury_score']

    return df


@st.cache_data
def train_price_model(df):
    """
    Full training pipeline. Never modifies the input — works entirely on copies.

    Steps:
      1. Filter to known price categories
      2. Smart group-based imputation of missing ratings
      3. Feature engineering (amenity scores, log reviews, star signals, interactions)
      4. Label-encode cuisine and area
      5. Train Random Forest with class balancing
      6. Return model + all diagnostics needed by the UI
    """
    # ── 1. Filter & copy (never touch the original df) ───────────────────────
    df_model = df[df['price_category'].isin(['Budget', 'Mid-Range', 'High-End'])].copy()
    n_before_impute = df_model['rating_overall'].isna().sum()

    # ── 2. Smart rating imputation ────────────────────────────────────────────
    df_model = _smart_impute_rating(df_model)
    n_after_impute  = df_model['rating_overall'].isna().sum()   # should be 0
    n_recovered     = n_before_impute - n_after_impute

    # ── 3. Feature engineering ────────────────────────────────────────────────
    df_model = _engineer_features(df_model)

    # ── 4. Encode categoricals ────────────────────────────────────────────────
    le_cuisine = LabelEncoder()
    le_area    = LabelEncoder()
    df_model['cuisine_enc'] = le_cuisine.fit_transform(df_model['cuisine_primary'].fillna('Unknown'))
    df_model['area_enc']    = le_area.fit_transform(df_model['area'].fillna('Unknown'))

    # ── 5. Assemble feature list ──────────────────────────────────────────────
    binary_feats  = [c + '_bin' for c in FEATURE_COLS]
    feature_names = (
        binary_feats
        + ['cuisine_enc', 'area_enc']
        + ['rating_overall', 'log_reviews']
        + STAR_COLS
        + ['feature_count', 'luxury_score', 'convenience_score',
           'star_high_pct', 'star_low_pct', 'rating_x_luxury']
    )

    df_model = df_model.dropna(subset=feature_names + ['price_category'])

    X = df_model[feature_names]
    y = df_model['price_category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, output_dict=True)

    # 5-fold CV on full dataset for a more reliable accuracy estimate
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    # ── 7. Feature importances with clean names ───────────────────────────────
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    display_names = []
    for name in importances.index:
        if name in FEATURE_DISPLAY:
            display_names.append(FEATURE_DISPLAY[name])
        else:
            display_names.append(name.replace('_bin', '').replace('_', ' ').title())

    importances_display = pd.DataFrame({
        'Feature':    display_names,
        'Importance': importances.values
    })

    # Store mean star values (used to fill unknowns in interactive predictor)
    star_means = {col: float(df_model[col].mean()) for col in STAR_COLS}

    return {
        'model':               model,
        'le_cuisine':          le_cuisine,
        'le_area':             le_area,
        'accuracy':            accuracy,
        'cv_mean':             float(cv_scores.mean()),
        'cv_std':              float(cv_scores.std()),
        'report':              report,
        'feature_names':       feature_names,
        'importances_display': importances_display,
        'y_test':              y_test,
        'y_pred':              y_pred,
        'classes':             list(model.classes_),
        'cuisine_classes':     list(le_cuisine.classes_),
        'area_classes':        list(le_area.classes_),
        'n_train':             len(X_train),
        'n_test':              len(X_test),
        'n_total':             len(df_model),
        'n_recovered':         n_recovered,
        'star_means':          star_means,
    }


def _encode_input(result, cuisine, area, rating, review_count, feature_flags):
    """Build a single-row feature vector matching exactly what the model was trained on."""
    cuisine_enc = (result['cuisine_classes'].index(cuisine)
                   if cuisine in result['cuisine_classes'] else 0)
    area_enc    = (result['area_classes'].index(area)
                   if area in result['area_classes'] else 0)

    binary = {col + '_bin': int(feature_flags.get(col, False)) for col in FEATURE_COLS}

    feature_count     = sum(binary.values())
    luxury_score      = sum(int(feature_flags.get(f, False)) for f in LUXURY_FEATURES)
    convenience_score = sum(int(feature_flags.get(f, False)) for f in CONVENIENCE_FEATURES)
    log_reviews       = np.log1p(review_count)
    rating_x_luxury   = rating * luxury_score

    # Star percentages: for a restaurant being described by the user we don't have
    # real review data yet, so we use the dataset-wide average as a neutral baseline
    star_vals = result['star_means'].copy()
    star_high_pct = star_vals.get('star_5_percent', 0) + star_vals.get('star_4_percent', 0)
    star_low_pct  = star_vals.get('star_1_percent', 0) + star_vals.get('star_2_percent', 0)

    row = {**binary}
    row.update({
        'cuisine_enc':        cuisine_enc,
        'area_enc':           area_enc,
        'rating_overall':     rating,
        'log_reviews':        log_reviews,
        **star_vals,
        'feature_count':      feature_count,
        'luxury_score':       luxury_score,
        'convenience_score':  convenience_score,
        'star_high_pct':      star_high_pct,
        'star_low_pct':       star_low_pct,
        'rating_x_luxury':    rating_x_luxury,
    })

    return pd.DataFrame([row])[result['feature_names']]


def render_price_predictor(df_restaurants):
    st.subheader(":material/price_check: Price Category Predictor")
    st.write(
        "A Random Forest trained on **location, cuisine, 12 amenity signals, "
        "rating quality, review volume, and star-distribution patterns** "
        "to predict whether a restaurant is **Budget**, **Mid-Range**, or **High-End**."
    )
    st.write("---")

    with st.spinner("Training model…"):
        result = train_price_model(df_restaurants)

    # ── DATA RECOVERY CALLOUT ─────────────────────────────────────────────────
    st.info(
        f"**Smart Imputation:** {result['n_recovered']:,} restaurants with missing ratings were "
        f"recovered using group-based averages (area × cuisine × price tier). "
        f"Total usable rows: **{result['n_total']:,}** (up from ~1,045 with naive drop)."
    )

    # ── KPI ROW ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Test-Set Accuracy",   f"{result['accuracy']:.1%}")
    col2.metric("5-Fold CV Accuracy",  f"{result['cv_mean']:.1%}", f"±{result['cv_std']:.1%}")
    col3.metric("Training Samples",    f"{result['n_train']:,}")
    col4.metric("Test Samples",        f"{result['n_test']:,}")
    col5.metric("Price Categories",    len(result['classes']))

    st.write("---")

    # ── FEATURE IMPORTANCE + PER-CLASS PERFORMANCE ───────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**What drives the prediction?**")
        st.caption(
            "Importance = how much each feature reduces prediction error across all 300 trees. "
            "Higher = more influential."
        )
        imp_df = result['importances_display'].head(15)
        fig_imp = px.bar(
            imp_df, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Blues',
            height=480
        )
        fig_imp.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            coloraxis_showscale=False,
            margin=dict(l=0)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_right:
        st.markdown("**Per-class performance**")
        st.caption(
            "Precision = when the model predicts X, how often is it right? "
            "Recall = of all real X restaurants, how many did the model catch?"
        )
        report  = result['report']
        classes = result['classes']
        f1_data = pd.DataFrame({
            'Class':     classes,
            'Precision': [report[cls]['precision'] for cls in classes],
            'Recall':    [report[cls]['recall']    for cls in classes],
            'F1 Score':  [report[cls]['f1-score']  for cls in classes],
        })
        fig_f1 = px.bar(
            f1_data.melt(id_vars='Class', var_name='Metric', value_name='Score'),
            x='Class', y='Score', color='Metric', barmode='group',
            color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c'],
            height=480
        )
        fig_f1.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig_f1, use_container_width=True)

    st.write("---")

    # ── ENGINEERED FEATURES EXPLAINER ────────────────────────────────────────
    with st.expander("What are Luxury Score, Convenience Score, and the Interaction feature?"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Luxury Score** *(0 – 6)*")
            st.write("Count of high-end indicators a restaurant has:")
            for f in LUXURY_FEATURES:
                st.write(f"• {f.replace('_', ' ').title()}")
        with col_b:
            st.markdown("**Convenience Score** *(0 – 4)*")
            st.write("Count of casual-service amenities:")
            for f in CONVENIENCE_FEATURES:
                st.write(f"• {f.replace('_', ' ').title()}")
        with col_c:
            st.markdown("**Rating × Luxury (interaction)**")
            st.write(
                "A restaurant scoring 4.8★ *and* having 5 luxury amenities gives a much "
                "stronger High-End signal than either value alone. "
                "Multiplying them lets the model capture this combined effect."
            )

    # ── CONFUSION MATRIX ─────────────────────────────────────────────────────
    with st.expander("View Confusion Matrix"):
        classes = result['classes']
        conf = pd.crosstab(
            pd.Series(result['y_test'], name='Actual'),
            pd.Series(result['y_pred'], name='Predicted')
        ).reindex(index=classes, columns=classes, fill_value=0)

        fig_conf = go.Figure(data=go.Heatmap(
            z=conf.values,
            x=conf.columns.tolist(),
            y=conf.index.tolist(),
            colorscale='Blues',
            text=conf.values,
            texttemplate='%{text}',
            textfont={"size": 15},
        ))
        fig_conf.update_layout(
            title='Confusion Matrix — rows = Actual, columns = Predicted',
            xaxis_title='Predicted', yaxis_title='Actual', height=380
        )
        st.plotly_chart(fig_conf, use_container_width=True)
        st.caption(
            "Diagonal = correct predictions. Off-diagonal = errors. "
            "High-End tends to be confused with Mid-Range because it's the smallest class."
        )

    st.write("---")

    # ── INTERACTIVE PREDICTOR ─────────────────────────────────────────────────
    st.subheader(":material/magic_button: Try It — Predict a Restaurant's Price Category")
    st.caption(
        "Star distribution is estimated from the dataset average since real reviews "
        "aren't available for a new restaurant — the other inputs drive the prediction."
    )

    form_col1, form_col2, form_col3 = st.columns([1.2, 1.2, 1.6])

    with form_col1:
        st.markdown("**Location & Rating**")
        known_cuisines = ['Unknown'] + sorted([c for c in result['cuisine_classes'] if c != 'Unknown'])
        known_areas    = ['Unknown'] + sorted([a for a in result['area_classes']    if a != 'Unknown'])
        sel_cuisine = st.selectbox("Cuisine Type",   known_cuisines, key="pp_cuisine")
        sel_area    = st.selectbox("Area",           known_areas,    key="pp_area")
        sel_rating  = st.slider("Overall Rating",    1.0, 5.0, 4.0, 0.1, key="pp_rating")
        sel_reviews = st.number_input("Review Count", min_value=0, max_value=50000,
                                      value=100, step=10, key="pp_reviews")

        # Show derived scores live
        st.write("---")
        st.caption("**Derived scores from your amenity choices:**")

    with form_col2:
        st.markdown("**Amenities**")
        flags = {}
        for col in FEATURE_COLS:
            flags[col] = st.checkbox(col.replace('_', ' ').title(), key=f"pp_{col}")

    # Compute derived scores to show user
    luxury_live      = sum(int(flags.get(f, False)) for f in LUXURY_FEATURES)
    convenience_live = sum(int(flags.get(f, False)) for f in CONVENIENCE_FEATURES)
    total_live       = sum(int(v) for v in flags.values())

    with form_col1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Luxury", f"{luxury_live}/6")
        c2.metric("Convenience", f"{convenience_live}/4")
        c3.metric("Total", f"{total_live}/12")

    with form_col3:
        st.markdown("**Prediction**")
        X_input = _encode_input(result, sel_cuisine, sel_area, sel_rating, sel_reviews, flags)
        pred    = result['model'].predict(X_input)[0]
        proba   = result['model'].predict_proba(X_input)[0]
        classes = result['classes']

        color = PRICE_COLORS.get(pred, '#999')
        st.markdown(
            f"<div style='background:{color};color:white;padding:16px 24px;"
            f"border-radius:10px;font-size:1.5rem;font-weight:bold;text-align:center;"
            f"margin-bottom:16px;'>{pred}</div>",
            unsafe_allow_html=True
        )

        prob_df = pd.DataFrame({'Category': classes, 'Probability': proba})
        fig_prob = px.bar(
            prob_df, x='Category', y='Probability',
            color='Category', color_discrete_map=PRICE_COLORS,
            text=[f"{p:.0%}" for p in proba],
            title='Confidence per Category',
            height=340
        )
        fig_prob.update_traces(textposition='outside')
        fig_prob.update_layout(
            yaxis_range=[0, 1.15], showlegend=False,
            yaxis_tickformat='.0%'
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.write("---")

    # ── PRICE DISTRIBUTION IN DATA ────────────────────────────────────────────
    st.subheader(":material/bar_chart: Price Category Distribution in Data")
    known = df_restaurants[df_restaurants['price_category'].isin(['Budget', 'Mid-Range', 'High-End'])]
    dist  = (known['price_category']
             .value_counts()
             .reindex(['Budget', 'Mid-Range', 'High-End'])
             .reset_index())
    dist.columns = ['Category', 'Count']
    fig_dist = px.bar(
        dist, x='Category', y='Count',
        color='Category', color_discrete_map=PRICE_COLORS,
        text='Count', height=350
    )
    fig_dist.update_traces(textposition='outside')
    fig_dist.update_layout(showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption(
        "Class imbalance (High-End is rare) is handled via `class_weight='balanced'` in the model, "
        "which internally upweights minority classes during training."
    )
