import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EduPro - Course Demand & Revenue",
    page_icon="📚",
    layout="wide"
)

# =========================================================
# LOAD TRAINED ML MODELS
# =========================================================

enrollment_model = joblib.load(
    "edupro_enrollment_model.pkl"
)

revenue_model = joblib.load(
    "edupro_revenue_model.pkl"
)

feature_columns = joblib.load(
    "feature_columns.pkl"
)

# =========================================================
# GET CATEGORIES AND LEVELS
# =========================================================

try:

    preprocessor = enrollment_model.named_steps["prep"]

    categorical_encoder = (
        preprocessor.named_transformers_["cat"]
    )

    categories = list(
        categorical_encoder.categories_[0]
    )

    levels = list(
        categorical_encoder.categories_[1]
    )

except:

    categories = [
        "Programming",
        "Data Science",
        "Web Development",
        "Artificial Intelligence",
        "Business",
        "Design"
    ]

    levels = [
        "Beginner",
        "Intermediate",
        "Advanced"
    ]

# =========================================================
# TITLE
# =========================================================

st.title(
    "📚 EduPro Course Demand & Revenue Forecasting"
)

st.write(
    "Predict course enrollment demand and revenue "
    "using Machine Learning."
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 EduPro Dashboard")

st.sidebar.write(
    "Use the course information below to "
    "generate demand and revenue predictions."
)

st.sidebar.divider()

st.sidebar.info(
    "The predictions are generated using "
    "trained Machine Learning models."
)

# =========================================================
# COURSE INFORMATION
# =========================================================

st.header("🎯 Course Information")

st.write(
    "Enter the course details to estimate "
    "future enrollment demand and revenue."
)

col1, col2 = st.columns(2)

# ---------------------------------------------------------
# LEFT COLUMN
# ---------------------------------------------------------

with col1:

    course_price = st.number_input(
        "Course Price (₹)",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    course_duration = st.number_input(
        "Course Duration",
        min_value=1.0,
        value=10.0,
        step=1.0
    )

    course_rating = st.slider(
        "Course Rating",
        min_value=0.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

# ---------------------------------------------------------
# RIGHT COLUMN
# ---------------------------------------------------------

with col2:

    past_enrollments = st.number_input(
        "Past Enrollments",
        min_value=0,
        value=50,
        step=1
    )

    course_category = st.selectbox(
        "Course Category",
        categories
    )

    course_level = st.selectbox(
        "Course Level",
        levels
    )

# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "🔮 Predict Course Demand",
    type="primary",
    use_container_width=True
)

# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "CoursePrice": [
            course_price
        ],

        "CourseDuration": [
            course_duration
        ],

        "CourseRating": [
            course_rating
        ],

        "PastEnrollments": [
            past_enrollments
        ],

        "CourseCategory": [
            course_category
        ],

        "CourseLevel": [
            course_level
        ]
    })

    # -----------------------------------------------------
    # ARRANGE FEATURES
    # -----------------------------------------------------

    input_data = input_data[
        feature_columns
    ]

    # -----------------------------------------------------
    # ENROLLMENT PREDICTION
    # -----------------------------------------------------

    enrollment_prediction = (
        enrollment_model.predict(
            input_data
        )[0]
    )

    # -----------------------------------------------------
    # REVENUE PREDICTION
    # -----------------------------------------------------

    revenue_prediction = (
        revenue_model.predict(
            input_data
        )[0]
    )

    # -----------------------------------------------------
    # REMOVE NEGATIVE PREDICTIONS
    # -----------------------------------------------------

    enrollment_prediction = max(
        0,
        enrollment_prediction
    )

    revenue_prediction = max(
        0,
        revenue_prediction
    )

    # =====================================================
    # PREDICTION RESULTS
    # =====================================================

    st.divider()

    st.header("📊 Prediction Results")

    result_col1, result_col2 = st.columns(2)

    # -----------------------------------------------------
    # ENROLLMENT RESULT
    # -----------------------------------------------------

    with result_col1:

        st.metric(
            "Predicted Enrollments",
            f"{enrollment_prediction:.0f}"
        )

    # -----------------------------------------------------
    # REVENUE RESULT
    # -----------------------------------------------------

    with result_col2:

        st.metric(
            "Predicted Revenue",
            f"₹{revenue_prediction:,.2f}"
        )

    st.success(
        "Prediction generated successfully!"
    )

    # =====================================================
    # FORECAST VISUALIZATION
    # =====================================================

    st.divider()

    st.header("📈 Forecast Visualization")

    chart_col1, chart_col2 = st.columns(2)

    # -----------------------------------------------------
    # ENROLLMENT CHART
    # -----------------------------------------------------

    with chart_col1:

        st.subheader(
            "Predicted Enrollment"
        )

        enrollment_chart = pd.DataFrame({
            "Predicted Enrollments": [
                enrollment_prediction
            ]
        })

        st.bar_chart(
            enrollment_chart
        )

    # -----------------------------------------------------
    # REVENUE CHART
    # -----------------------------------------------------

    with chart_col2:

        st.subheader(
            "Predicted Revenue"
        )

        revenue_chart = pd.DataFrame({
            "Predicted Revenue": [
                revenue_prediction
            ]
        })

        st.bar_chart(
            revenue_chart
        )

    # =====================================================
    # CATEGORY LEVEL DEMAND COMPARISON
    # =====================================================

    st.divider()

    st.header(
        "📊 Category-Level Demand Comparison"
    )

    st.write(
        "Comparison of predicted enrollments across "
        "different course categories."
    )

    category_input = pd.DataFrame({

        "CoursePrice": [
            course_price
        ] * len(categories),

        "CourseDuration": [
            course_duration
        ] * len(categories),

        "CourseRating": [
            course_rating
        ] * len(categories),

        "PastEnrollments": [
            past_enrollments
        ] * len(categories),

        "CourseCategory": categories,

        "CourseLevel": [
            course_level
        ] * len(categories)
    })

    category_input = category_input[
        feature_columns
    ]

    category_predictions = (
        enrollment_model.predict(
            category_input
        )
    )

    category_predictions = pd.Series(
        category_predictions
    ).clip(lower=0)

    category_results = pd.DataFrame({

        "Category": categories,

        "Predicted Enrollments":
            category_predictions
    })

    category_results = (
        category_results
        .sort_values(
            "Predicted Enrollments",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # CATEGORY TABLE
    # -----------------------------------------------------

    st.dataframe(
        category_results,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # CATEGORY CHART
    # -----------------------------------------------------

    st.bar_chart(
        category_results.set_index(
            "Category"
        )
    )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.divider()

    st.header(
        "🔍 Feature Importance"
    )

    st.write(
        "Features that contribute most to the "
        "enrollment prediction."
    )

    try:

        model = enrollment_model.named_steps[
            "model"
        ]

        prep = enrollment_model.named_steps[
            "prep"
        ]

        feature_names = (
            prep.get_feature_names_out()
        )

        # -------------------------------------------------
        # TREE MODEL
        # -------------------------------------------------

        if hasattr(
            model,
            "feature_importances_"
        ):

            importance_values = (
                model.feature_importances_
            )

        # -------------------------------------------------
        # LINEAR MODEL
        # -------------------------------------------------

        elif hasattr(
            model,
            "coef_"
        ):

            importance_values = abs(
                model.coef_.ravel()
            )

        else:

            importance_values = None

        # -------------------------------------------------
        # DISPLAY IMPORTANCE
        # -------------------------------------------------

        if importance_values is not None:

            importance_df = pd.DataFrame({

                "Feature":
                    feature_names,

                "Importance":
                    importance_values
            })

            # Remove preprocessing prefixes

            importance_df["Feature"] = (
                importance_df["Feature"]
                .str.replace(
                    "num__",
                    "",
                    regex=False
                )
                .str.replace(
                    "cat__",
                    "",
                    regex=False
                )
            )

            # -------------------------------------------------
            # GROUP CATEGORICAL FEATURES
            # -------------------------------------------------

            def get_original_feature(feature):

                for category in categories:

                    if feature == category:
                        return "CourseCategory"

                for level in levels:

                    if feature == level:
                        return "CourseLevel"

                return feature

            importance_df[
                "Original Feature"
            ] = importance_df[
                "Feature"
            ].apply(
                get_original_feature
            )

            importance_df = (
                importance_df
                .groupby(
                    "Original Feature"
                )["Importance"]
                .sum()
                .reset_index()
                .sort_values(
                    "Importance",
                    ascending=False
                )
            )

            # -------------------------------------------------
            # TABLE
            # -------------------------------------------------

            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True
            )

            # -------------------------------------------------
            # CHART
            # -------------------------------------------------

            st.bar_chart(
                importance_df.set_index(
                    "Original Feature"
                )
            )

        else:

            st.info(
                "Feature importance is not available "
                "for this model type."
            )

    except Exception:

        st.warning(
            "Feature importance could not be "
            "displayed for this model."
        )

# =========================================================
# PROJECT INFORMATION
# =========================================================

st.divider()

st.subheader(
    "ℹ️ About This Project"
)

st.write(
    "EduPro uses predictive analytics to estimate "
    "course enrollment demand and revenue. The "
    "dashboard supports data-driven decisions related "
    "to course planning, pricing and category demand."
)

st.caption(
    "EduPro Predictive Analytics | Machine Learning Project"
)