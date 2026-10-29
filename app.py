import os
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import streamlit as st
from bs4 import BeautifulSoup
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

st.set_page_config(page_title="Countries: Development and Economy", layout="wide")

# ---------------------------------------------------------
# STYLE (same look as the Gradio custom_css)
# ---------------------------------------------------------
st.markdown("""
<style>
.main-header { text-align: center; margin-bottom: 1rem; }
.main-header h1 { color: #1e3d59; font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
.main-header h3 { color: #17b978; font-size: 1.1rem; font-weight: 500; }
.meta-box { background-color: #f5f7fa; border-radius: 8px; padding: 12px; border-left: 5px solid #1e3d59; color: #222; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA: scrape once, then reuse countries_data.csv
# ---------------------------------------------------------
HEADERS = {"User-Agent": "Mozilla/5.0"}
URLS = {
    "regions":    "https://www.worldometers.info/geography/countries-of-the-world/",
    "population": "https://www.worldometers.info/world-population/population-by-country/",
    "life":       "https://www.worldometers.info/demographics/life-expectancy/",
    "gdp":        "https://www.worldometers.info/gdp/gdp-per-capita/",
}
CSV_FILE = "countries_data.csv"
LABELS = ["Low income", "Lower-middle", "Upper-middle", "High income"]

if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    df["Development Group"] = pd.cut(
        df["GDP per Capita"], bins=[0, 2000, 6000, 15000, np.inf], labels=LABELS
    )
    data_source = "Saved CSV (countries_data.csv)"
else:
    try:
        tables = {}
        for key in URLS:
            response = requests.get(URLS[key], headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find_all("table")[0]
            tables[key] = pd.read_html(StringIO(str(table)))[0]

        regions_raw = tables["regions"]
        pop_raw = tables["population"]
        life_raw = tables["life"]
        gdp_raw = tables["gdp"]

        regions_country = [c for c in regions_raw.columns if "country" in str(c).lower()][0]
        regions_region = [c for c in regions_raw.columns if "region" in str(c).lower()][0]
        regions = regions_raw[[regions_country, regions_region]].copy()
        regions.columns = ["Country", "Region"]

        pop = pd.DataFrame()
        pop["Country"] = pop_raw[[c for c in pop_raw.columns if "country" in str(c).lower()][0]]
        wanted = {
            "Population": "Population",
            "Density": "Density",
            "Fert": "Fertility Rate",
            "Median": "Median Age",
            "Urban": "Urban Pop %",
        }
        for word in wanted:
            found = [c for c in pop_raw.columns if word.lower() in str(c).lower()]
            if len(found) > 0:
                pop[wanted[word]] = pd.to_numeric(
                    pop_raw[found[0]].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce"
                )

        life = pd.DataFrame()
        life["Country"] = life_raw[[c for c in life_raw.columns if "country" in str(c).lower()][0]]
        life_found = [c for c in life_raw.columns if "both" in str(c).lower()]
        if len(life_found) == 0:
            life_found = [c for c in life_raw.columns if "life" in str(c).lower()]
        if len(life_found) == 0:
            life_found = [life_raw.columns[2]]
        life["Life Expectancy"] = pd.to_numeric(
            life_raw[life_found[0]].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce"
        )

        gdp = pd.DataFrame()
        gdp["Country"] = gdp_raw[[c for c in gdp_raw.columns if "country" in str(c).lower()][0]]
        gdp_col = [c for c in gdp_raw.columns if "gdp" in str(c).lower()][0]
        gdp_text = gdp_raw[gdp_col].astype(str).str.extract(r"\$\s?([\d,]+)")[0]
        gdp["GDP per Capita"] = pd.to_numeric(
            gdp_text.astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce"
        )

        df = regions.merge(pop, on="Country", how="left")
        df = df.merge(life, on="Country", how="left")
        df = df.merge(gdp, on="Country", how="left")
        df = df.dropna(subset=["Life Expectancy", "GDP per Capita"]).reset_index(drop=True)
        df["Development Group"] = pd.cut(
            df["GDP per Capita"], bins=[0, 2000, 6000, 15000, np.inf], labels=LABELS
        )
        df.to_csv(CSV_FILE, index=False)
        data_source = "Live scrape from Worldometers"
    except Exception:
        np.random.seed(42)
        n = 194
        regions_list = ["Africa", "Asia", "Europe", "Latin America", "Northern America", "Oceania"]
        reg_choice = np.random.choice(regions_list, size=n, p=[0.28, 0.25, 0.22, 0.15, 0.02, 0.08])

        gdp_vals = np.random.lognormal(mean=8.8, sigma=1.2, size=n)
        gdp_vals = np.clip(gdp_vals, 450, 288002)

        life_vals = 35 + 3.5 * np.log(gdp_vals) + np.random.normal(0, 2.5, n)
        life_vals = np.clip(life_vals, 54.95, 86.73)

        median_age = 15 + (life_vals - 50) * 0.7 + np.random.normal(0, 2, n)
        fertility = np.clip(6.0 - (life_vals - 50) * 0.1 + np.random.normal(0, 0.3, n), 0.76, 5.83)
        urban_pop = np.clip((life_vals - 50) * 2 + np.random.normal(0, 10, n), 12.7, 99.0)
        density = np.random.exponential(scale=200, size=n) + 2
        pop_vals = np.random.randint(10000, 1400000000, size=n)

        df = pd.DataFrame({
            "Country": [f"Country_{i+1}" for i in range(n)],
            "Region": reg_choice,
            "Population": pop_vals.astype(int),
            "Density": density.astype(int),
            "Fertility Rate": np.round(fertility, 2),
            "Median Age": np.round(median_age, 1),
            "Urban Pop %": np.round(urban_pop, 1),
            "Life Expectancy": np.round(life_vals, 2),
            "GDP per Capita": np.round(gdp_vals, 2),
        })
        df["Development Group"] = pd.cut(
            df["GDP per Capita"], bins=[0, 2000, 6000, 15000, np.inf], labels=LABELS
        )
        data_source = "Fallback dataset (website could not be reached)"

df["Log GDP"] = np.log(df["GDP per Capita"])

# ---------------------------------------------------------
# REGRESSION DATA + MODELS
# ---------------------------------------------------------
reg_features = ["Log GDP", "Median Age", "Fertility Rate", "Urban Pop %", "Density"]
reg_target = "Life Expectancy"
df_reg = df[reg_features + [reg_target]].dropna()

X_reg = df_reg[reg_features]
y_reg = df_reg[reg_target]
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
Xr_train_scaled = scaler_reg.fit_transform(Xr_train)
Xr_test_scaled = scaler_reg.transform(Xr_test)

reg_models = {
    "Linear Regression": LinearRegression().fit(Xr_train, yr_train),
    "Ridge Regression": Ridge(alpha=1.0).fit(Xr_train_scaled, yr_train),
    "Lasso Regression": Lasso(alpha=0.1).fit(Xr_train_scaled, yr_train),
    "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=42).fit(Xr_train, yr_train),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42).fit(Xr_train, yr_train),
}

# ---------------------------------------------------------
# CLASSIFICATION DATA + MODELS
# ---------------------------------------------------------
clf_features = ["Life Expectancy", "Median Age", "Fertility Rate", "Urban Pop %", "Density"]
clf_target = "Development Group"
df_clf = df[clf_features + [clf_target]].dropna().copy()
df_clf[clf_target] = df_clf[clf_target].astype(str)

Xc = df_clf[clf_features]
yc = df_clf[clf_target]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)

scaler_clf = StandardScaler()
Xc_train_scaled = scaler_clf.fit_transform(Xc_train)
Xc_test_scaled = scaler_clf.transform(Xc_test)

clf_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000).fit(Xc_train_scaled, yc_train),
    "KNN": KNeighborsClassifier(n_neighbors=5).fit(Xc_train_scaled, yc_train),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xc_train, yc_train),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42).fit(Xc_train, yc_train),
    "SVM": SVC(kernel="rbf").fit(Xc_train_scaled, yc_train),
}

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-header"><h1>Countries: Development and Economy</h1>'
    '<h3>Interactive Data Analysis & Machine Learning Platform</h3></div>',
    unsafe_allow_html=True,
)

with st.expander("Project Metadata & Aim", expanded=False):
    st.markdown(
        '<div class="meta-box"><p><strong>Name:</strong> Rudrani Deshpande | '
        '<strong>Section-Batch:</strong> B1-B1 | <strong>Roll No.:</strong> 13</p></div>',
        unsafe_allow_html=True,
    )
    st.write("**Aim:** Study how population, life expectancy and income (GDP per capita) differ across the countries of the world")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "1. Scraping & Raw Data",
    "2. Exploratory Data Analysis",
    "3. Probability Distribution",
    "4. Sampling Techniques",
    "5. Central Limit Theorem",
    "6. Hypothesis Testing",
    "7. Life Expectancy Regression",
    "8. Development Group Classification",
])

# ---------------------------------------------------------
# TAB 1: Scraping & Raw Data
# ---------------------------------------------------------
with tab1:
    st.markdown("### Web Scraped Dataset (Worldometers)")
    st.caption("Data source: " + data_source)
    st.dataframe(df)
    st.markdown("**Descriptive Statistics**")
    st.dataframe(df.describe().reset_index())

# ---------------------------------------------------------
# TAB 2: EDA
# ---------------------------------------------------------
with tab2:
    chart_type = st.selectbox(
        "Select Visualization Chart",
        ["Life Expectancy Distribution", "GDP per Capita Distribution", "Life Expectancy by Region",
         "GDP per Capita by Region", "GDP vs Life Expectancy (Scatter Plot)", "Correlation Heatmap",
         "Development Group Counts"],
        index=0,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if chart_type == "Life Expectancy Distribution":
        sns.histplot(df["Life Expectancy"], bins=15, kde=True, ax=ax, color="#2b5c8f")
        ax.set_title("Distribution of Life Expectancy", fontsize=14, fontweight="bold")
    elif chart_type == "GDP per Capita Distribution":
        sns.histplot(df["GDP per Capita"], bins=30, kde=True, ax=ax, color="#27ae60")
        ax.set_title("Distribution of GDP per Capita", fontsize=14, fontweight="bold")
    elif chart_type == "Life Expectancy by Region":
        life_by_reg = df.groupby("Region")["Life Expectancy"].mean().sort_values(ascending=False)
        sns.barplot(x=life_by_reg.index, y=life_by_reg.values, hue=life_by_reg.index, legend=False, ax=ax, palette="Blues_r")
        ax.set_title("Average Life Expectancy by Region", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
    elif chart_type == "GDP per Capita by Region":
        sns.boxplot(x="Region", y="GDP per Capita", hue="Region", legend=False, data=df, ax=ax, palette="Set2")
        ax.set_title("GDP per Capita by Region", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
    elif chart_type == "GDP vs Life Expectancy (Scatter Plot)":
        sns.scatterplot(x="GDP per Capita", y="Life Expectancy", hue="Region", data=df, ax=ax, s=70)
        ax.set_xscale("log")
        ax.set_title("GDP per Capita vs Life Expectancy (Log Scale)", fontsize=14, fontweight="bold")
    elif chart_type == "Correlation Heatmap":
        num_cols = df.select_dtypes(include="number").columns
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
    elif chart_type == "Development Group Counts":
        sns.countplot(x="Development Group", hue="Development Group", legend=False, data=df, ax=ax, palette="viridis")
        ax.set_title("Number of Countries in Each Development Group", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------
# TAB 3: Probability Distribution
# ---------------------------------------------------------
with tab3:
    col1, col2 = st.columns([3, 1])
    with col1:
        n_samples = st.slider("Simulated Samples", 100, 5000, 1000, step=100)
    with col2:
        st.write("")
        st.button("Simulate", type="primary", key="sim_btn")

    x_life = df["Life Expectancy"].dropna()
    mean_val = x_life.mean()
    std_val = x_life.std()
    nd = np.random.normal(mean_val, std_val, int(n_samples))

    left, right = st.columns(2)
    with left:
        st.markdown("### Probability Distribution Analysis")
        st.markdown(f"* **Sample Mean:** {mean_val:.2f}\n* **Standard Deviation:** {std_val:.2f}")
    with right:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(nd, bins=25, color="#34495e", edgecolor="white")
        ax.set_title(f"Simulated Normal Distribution (N={n_samples})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------
# TAB 4: Sampling Techniques
# ---------------------------------------------------------
with tab4:
    col1, col2 = st.columns([3, 1])
    with col1:
        strat_frac = st.slider("Stratified Fraction", 0.1, 0.5, 0.2, step=0.05)
    with col2:
        st.write("")
        st.button("Evaluate Sampling", type="primary", key="sample_btn")

    sr = df.sample(n=20, random_state=1)
    sys_sample = df.iloc[::10]
    strat = df.groupby("Region").sample(frac=float(strat_frac), random_state=1)
    selected_regions = df["Region"].drop_duplicates().sample(n=2, random_state=1)
    clst = df[df["Region"].isin(selected_regions)]

    cmp = pd.DataFrame({
        "Life Expectancy Mean": [df["Life Expectancy"].mean(), sr["Life Expectancy"].mean(),
                                 sys_sample["Life Expectancy"].mean(), strat["Life Expectancy"].mean(),
                                 clst["Life Expectancy"].mean()],
        "GDP per Capita Mean": [df["GDP per Capita"].mean(), sr["GDP per Capita"].mean(),
                                sys_sample["GDP per Capita"].mean(), strat["GDP per Capita"].mean(),
                                clst["GDP per Capita"].mean()],
    }, index=["Population (All Data)", "Simple Random", "Systematic", "Stratified", "Cluster"]).round(2)
    cmp = cmp.reset_index().rename(columns={"index": "Sampling Strategy"})

    st.dataframe(cmp, hide_index=True)
    st.markdown("**Stratified Sampling** preserves regional proportions accurately.")

# ---------------------------------------------------------
# TAB 5: Central Limit Theorem
# ---------------------------------------------------------
with tab5:
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        sample_size = st.slider("Sample Size (n)", 5, 100, 30, step=5)
    with col2:
        num_samples = st.slider("Number of Samples", 100, 3000, 1000, step=100)
    with col3:
        st.write("")
        st.button("Run CLT", type="primary", key="clt_btn")

    x_gdp = df["GDP per Capita"].dropna()
    sample_means = [np.mean(np.random.choice(x_gdp, int(sample_size))) for _ in range(int(num_samples))]

    left, right = st.columns(2)
    with left:
        st.markdown("### Central Limit Theorem Metrics")
        st.markdown(
            f"* **Population Mean:** ${np.mean(x_gdp):,.2f}\n"
            f"* **Mean of Sample Means:** ${np.mean(sample_means):,.2f}"
        )
    with right:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.hist(x_gdp, bins=25, color="#e74c3c", edgecolor="white")
        ax1.set_title("Original Data (Right-Skewed)", fontsize=11, fontweight="bold")
        ax2.hist(sample_means, bins=25, color="#2980b9", edgecolor="white")
        ax2.set_title(f"CLT Sample Means (n={sample_size})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---------------------------------------------------------
# TAB 6: Hypothesis Testing (no plot)
# ---------------------------------------------------------
with tab6:
    st.markdown("### Statistical Tests & Relationship Analysis")
    test_type = st.radio(
        "Select Statistical Test",
        ["1-Sample t-test (Life Expectancy = 70)",
         "2-Sample t-test (Europe vs Africa Life Expectancy)",
         "ANOVA (Life Expectancy across all Regions)",
         "Chi-Square Test (Region vs Development Group)",
         "Correlation (GDP vs Life Expectancy)"],
        index=0,
    )
    st.button("Execute Test", type="primary", key="test_btn")

    alpha = 0.05
    if test_type == "1-Sample t-test (Life Expectancy = 70)":
        t_stat, p_val = stats.ttest_1samp(df["Life Expectancy"], 70)
        result_text = (f"**H0:** Mean = 70\n* **T-Statistic:** {t_stat:.4f}\n* **p-value:** {p_val:.5f}\n"
                       f"* **Decision:** {'Reject H0' if p_val < alpha else 'Fail to reject H0'}")
    elif test_type == "2-Sample t-test (Europe vs Africa Life Expectancy)":
        eur = df[df["Region"] == "Europe"]["Life Expectancy"]
        afr = df[df["Region"] == "Africa"]["Life Expectancy"]
        t_stat, p_val = stats.ttest_ind(eur, afr, equal_var=False)
        result_text = (f"**Europe Mean:** {eur.mean():.2f} | **Africa Mean:** {afr.mean():.2f}\n"
                       f"* **T-Statistic:** {t_stat:.4f}\n* **p-value:** {p_val:.5f}")
    elif test_type == "ANOVA (Life Expectancy across all Regions)":
        groups = [g["Life Expectancy"] for _, g in df.groupby("Region")]
        f_stat, p_val = stats.f_oneway(*groups)
        result_text = f"* **F-Statistic:** {f_stat:.4f}\n* **p-value:** {p_val:.5f}"
    elif test_type == "Chi-Square Test (Region vs Development Group)":
        table = pd.crosstab(df["Region"], df["Development Group"])
        chi2, p_val, _, _ = stats.chi2_contingency(table)
        result_text = f"* **Chi-Square Stat:** {chi2:.4f}\n* **p-value:** {p_val:.5f}"
    else:
        r, p_r = stats.pearsonr(np.log(df["GDP per Capita"]), df["Life Expectancy"])
        rho, p_s = stats.spearmanr(df["GDP per Capita"], df["Life Expectancy"])
        result_text = (f"* **Pearson r (Log GDP):** {r:.3f} (p: {p_r:.2e})\n"
                       f"* **Spearman rho:** {rho:.3f} (p: {p_s:.2e})")

    st.markdown(result_text)

# ---------------------------------------------------------
# TAB 7: Life Expectancy Regression
# ---------------------------------------------------------
with tab7:
    st.markdown("### Predict Life Expectancy & View Performance Plot")
    st.button("Evaluate Models & Plot Fit", type="primary", key="reg_eval_btn")

    reg_preds = {
        "Linear Regression": reg_models["Linear Regression"].predict(Xr_test),
        "Ridge Regression": reg_models["Ridge Regression"].predict(Xr_test_scaled),
        "Lasso Regression": reg_models["Lasso Regression"].predict(Xr_test_scaled),
        "Decision Tree": reg_models["Decision Tree"].predict(Xr_test),
        "Random Forest": reg_models["Random Forest"].predict(Xr_test),
    }

    reg_results = []
    best_r2 = -999
    best_reg_pred = None
    best_reg_name = ""
    for m_name in reg_preds:
        pred = reg_preds[m_name]
        mae = mean_absolute_error(yr_test, pred)
        mse = mean_squared_error(yr_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(yr_test, pred)
        reg_results.append({"Model": m_name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R-Squared": r2})
        if r2 > best_r2:
            best_r2 = r2
            best_reg_pred = pred
            best_reg_name = m_name
    reg_res_df = pd.DataFrame(reg_results).round(4)

    left, right = st.columns(2)
    with left:
        st.dataframe(reg_res_df, hide_index=True)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(yr_test, best_reg_pred, color="#2980b9", alpha=0.8, edgecolors="k")
        ax.plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], "r--", lw=2, label="Ideal Fit")
        ax.set_title(f"Actual vs Predicted ({best_reg_name})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual Life Expectancy")
        ax.set_ylabel("Predicted Life Expectancy")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.markdown(f"**Best Regression Model:** `{best_reg_name}`")

    st.markdown("---")
    st.markdown("#### Live Life Expectancy Predictor")
    c1, c2, c3 = st.columns(3)
    with c1:
        in_gdp = st.number_input("GDP per Capita ($)", value=15000.0, key="in_gdp")
        in_dens = st.number_input("Density", value=150.0, key="in_dens")
    with c2:
        in_med_age = st.slider("Median Age", 10, 60, 30, key="in_med_age")
        in_urb = st.slider("Urban Pop %", 0, 100, 60, key="in_urb")
    with c3:
        in_fert = st.slider("Fertility Rate", 0.5, 7.0, 2.1, key="in_fert")
        reg_model_choice = st.selectbox("Model", list(reg_models.keys()), index=2, key="reg_model_choice")

    if st.button("Predict", key="predict_reg_btn"):
        log_gdp = np.log(max(in_gdp, 1))
        input_data = pd.DataFrame([[log_gdp, in_med_age, in_fert, in_urb, in_dens]], columns=reg_features)
        model = reg_models[reg_model_choice]
        if reg_model_choice in ["Ridge Regression", "Lasso Regression"]:
            pred_value = model.predict(scaler_reg.transform(input_data))[0]
        else:
            pred_value = model.predict(input_data)[0]
        st.markdown(f"### Predicted Life Expectancy: **{pred_value:.2f} years**")

# ---------------------------------------------------------
# TAB 8: Development Group Classification
# ---------------------------------------------------------
with tab8:
    st.markdown("### Classify Country Income Level & View Confusion Matrix")
    st.button("Evaluate Classifiers & Plot Matrix", type="primary", key="clf_eval_btn")

    clf_preds = {
        "Logistic Regression": clf_models["Logistic Regression"].predict(Xc_test_scaled),
        "KNN": clf_models["KNN"].predict(Xc_test_scaled),
        "Decision Tree": clf_models["Decision Tree"].predict(Xc_test),
        "Random Forest": clf_models["Random Forest"].predict(Xc_test),
        "SVM": clf_models["SVM"].predict(Xc_test_scaled),
    }

    clf_results = []
    for m_name in clf_preds:
        pred = clf_preds[m_name]
        acc = accuracy_score(yc_test, pred)
        prec = precision_score(yc_test, pred, average="weighted", zero_division=0)
        rec = recall_score(yc_test, pred, average="weighted", zero_division=0)
        f1 = f1_score(yc_test, pred, average="weighted", zero_division=0)
        clf_results.append({"Model": m_name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1})
    clf_res_df = pd.DataFrame(clf_results).round(4)
    best_clf_name = clf_res_df.loc[clf_res_df["Accuracy"].idxmax(), "Model"]
    best_clf_pred = clf_preds[best_clf_name]

    cm_labels = [l for l in LABELS if l in yc.unique()]
    cm = confusion_matrix(yc_test, best_clf_pred, labels=cm_labels)

    left, right = st.columns(2)
    with left:
        st.dataframe(clf_res_df, hide_index=True)
    with right:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels, ax=ax)
        ax.set_title(f"Confusion Matrix ({best_clf_name})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.markdown(f"**Best Classifier:** `{best_clf_name}`")

    st.markdown("---")
    st.markdown("#### Live Income Category Predictor")
    d1, d2, d3 = st.columns(3)
    with d1:
        c_in_life = st.slider("Life Expectancy", 50.0, 90.0, 74.0, key="c_in_life")
        c_in_dens = st.number_input("Density", value=150.0, key="c_in_dens")
    with d2:
        c_in_med_age = st.slider("Median Age", 10, 60, 30, key="c_in_med_age")
        c_in_urb = st.slider("Urban Pop %", 0, 100, 60, key="c_in_urb")
    with d3:
        c_in_fert = st.slider("Fertility Rate", 0.5, 7.0, 2.1, key="c_in_fert")
        clf_model_choice = st.selectbox("Model", list(clf_models.keys()), index=0, key="clf_model_choice")

    if st.button("Classify", key="predict_clf_btn"):
        input_df = pd.DataFrame([[c_in_life, c_in_med_age, c_in_fert, c_in_urb, c_in_dens]], columns=clf_features)
        model = clf_models[clf_model_choice]
        if clf_model_choice in ["Logistic Regression", "KNN", "SVM"]:
            pred_group = model.predict(scaler_clf.transform(input_df))[0]
        else:
            pred_group = model.predict(input_df)[0]
        st.markdown(f"### Predicted Development Group: **{pred_group}**")
