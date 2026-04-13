import streamlit as st
import joblib
import pandas as pd

# -------- LOAD MODEL -------- #
model = joblib.load("model.pkl")
columns = joblib.load("modelcolumns.pkl")
threshold = joblib.load("threshold.pkl")


# -------- HELPER FUNCTIONS -------- #
def prepare_data(df, feature_columns):
    X = df.copy()

    if "FoodInsecure" in X.columns:
        X = X.drop(columns=["FoodInsecure"])

    for col in X.columns:
        X[col] = X[col].apply(lambda x: str(x) if isinstance(x, (dict, list)) else x)

    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=feature_columns, fill_value=0)

    return X

def categorize_risk(score, threshold):
    if score >= threshold:
        return "High"
    elif score >= threshold * 0.5:
        return "Medium"
    else:
        return "Low"


# -------- UI -------- #

st.set_page_config(page_title="Food Insecurity Predictor", layout="wide")
st.title("🍽️ Household Food Insecurity Predictor")

# -------- PROVINCE → DISTRICT MAP -------- #

province_district_map = {
    "Central": ["Chibombo","Chisamba","Chitambo","Kabwe","Kapiri Mposhi","Mkushi","Mumbwa","Serenje"],
    "Copperbelt": ["Chililabombwe","Chingola","Kalulushi","Kitwe","Luanshya","Lufwanyama","Masaiti","Mpongwe","Mufulira","Ndola"],
    "Eastern": ["Chadiza","Chipata","Katete","Lundazi","Mambwe","Nyimba","Petauke","Sinda","Vubwi","Kasenengwa","Lumezi","Lusangazi"],
    "Luapula": ["Chiengi","Chifunabuli","Chipili","Kawambwa","Mansa","Milenge","Mwansabombwe","Nchelenge","Samfya"],
    "Lusaka": ["Chilanga","Chongwe","Kafue","Luangwa","Lusaka","Rufunsa","Shibuyunji"],
    "Muchinga": ["Chama","Chinsali","Isoka","Kanchibiya","Lavushimanda","Mafinga","Mpika","Nakonde","Shiwangandu"],
    "Northern": ["Chilubi","Kaputa","Kasama","Lunte","Luwingu","Mbala","Mporokoso","Mpulungu","Mungwi","Nsama","Senga Hill"],
    "North-Western": ["Chavuma","Ikelenge","Kabompo","Kalumbila","Kasempa","Manyinga","Mufumbwe","Mwinilunga","Solwezi","Zambezi"],
    "Southern": ["Chikankata","Choma","Gwembe","Itezhi-Tezhi","Kalomo","Kazungula","Livingstone","Mazabuka","Monze","Namwala","Pemba","Siavonga","Sinazongwe","Zimba"],
    "Western": ["Kalabo","Kaoma","Limulunga","Luampa","Lukulu","Mitete","Mongu","Mulobezi","Nalolo","Nkeyema","Senanga","Sesheke","Shangombo","Sikongo","Sioma"]
}

genders = ["Female", "Male", "Unknown"]

marital_statuses = [
    "Cohabitating","Divorced/Separated","Divorced",
    "Married (Monogamous)","Married",
    "Married (Monogamy)","Married (Polygamous)",
    "Married (Polygamy)","Never Married","Other",
    "Separated","Single","Unknown","Widowed"
]

education_levels = [
    "College","Never been to school","No school",
    "Post secondary - vocational","Primary","Secondary",
    "Tertiary / University","University","Unknown",
    "Upper secondary - vocational","Vocational"
]


# -------- INPUTS -------- #

col1, col2, col3 = st.columns(3)

with col1:
    Province = st.selectbox("Province", list(province_district_map.keys()))
    District = st.selectbox("District", province_district_map[Province])
    Year = st.number_input("Year", 2000, 2100, 2024)
    MonthN = st.number_input("Month (1-12)", 1, 12, 1)
    Age = st.number_input("Age", 1, 65, 35)
    Gender = st.selectbox("Gender", genders)

with col2:
    MaritalStatus = st.selectbox("Marital Status", marital_statuses)
    Education = st.selectbox("Education", education_levels)
    HHSize = st.number_input("Household Size", 1, 20, 5)
    LandOwned = st.number_input("Land Owned (ha)", 0.0, 10.0, 1.5)
    LandCultivated = st.number_input("Land Cultivated (ha)", 0.0, 10.0, 1.0)

    rCSICat = st.selectbox(
        "rCSI Category",
        [1, 2, 3],
        help="0–3 (1)= Low, 4–18 (2) = Medium, ≥19 (3 )= High"
    )

with col3:
    Rainfall_mm = st.number_input("Rainfall (mm)", 0.0, 500.0, 120.0)

    RainfallAnomaly = st.number_input(
        "Rainfall Anomaly",
        -200.0, 200.0, -10.0,
        help="Rainfall_mm − Mean Rainfall (District, Month)"
    )

    Rainfall_z = st.number_input(
        "Rainfall Z-score",
        -5.0, 5.0, -0.5,
        help="Standardized rainfall; < -1 indicates drought"
    )

    NDVI = st.number_input("NDVI", 0.0, 1.0, 0.6)

    Drought = st.selectbox(
        "Drought",
        [0, 1],
        help="1 = Rainfall_z < -1 (Drought)"
    )

    FoodBasket = st.number_input(
        "Food Basket",
        0.0, 1000.0, 250.0,
        help="Sum of food group prices"
    )

    FoodBasketWeighted = st.number_input(
        "Weighted Food Basket",
        0.0, 2000.0, 300.0,
        help="Weighted by dietary importance"
    )

RainySeason = st.selectbox("Rainy Season", [0, 1])
LeanSeason = st.selectbox("Lean Season", [0, 1])
HarvestSeason = st.selectbox("Harvest Season", [0, 1])
DrySeason = st.selectbox("Dry Season", [0, 1])


# -------- SINGLE PREDICTION -------- #

if st.button("Predict"):
    
    df = pd.DataFrame([{
                        "Province": Province,
                        "District": District,
                        "Year": Year,
                        "MonthN": MonthN,
                        "Age": Age,
                        "Gender": Gender,
                        "MaritalStatus": MaritalStatus,
                        "Education": Education,
                        "HHSize": HHSize,
                        "LandOwned": LandOwned,
                        "LandCultivated": LandCultivated,
                        "rCSICat": rCSICat,
                        "Rainfall_mm": Rainfall_mm,
                        "RainfallAnomaly": RainfallAnomaly,
                        "Rainfall_z": Rainfall_z,
                        "NDVI": NDVI,
                        "Drought": Drought,
                        "FoodBasket": FoodBasket,
                        "FoodBasketWeighted": FoodBasketWeighted,
                        "RainySeason": RainySeason,
                        "LeanSeason": LeanSeason,
                        "HarvestSeason": HarvestSeason,
                        "DrySeason": DrySeason
                    }])

    X = prepare_data(df, columns)

    risk_score = model.predict_proba(X)[:, 1][0]
    food_insecure = int(risk_score >= threshold)
    risk_category = categorize_risk(risk_score, threshold)

    st.subheader("Prediction Results")

    color = "green" if risk_category == "Low" else "orange" if risk_category == "Medium" else "red"

    st.markdown(f"### Food Insecure: **{food_insecure}**")
    st.markdown(f"### Risk Score: **{round(risk_score,4)}**")
    st.markdown(f"### Risk Category: <span style='color:{color}; font-size:24px'><b>{risk_category}</b></span>", unsafe_allow_html=True)


# -------- FILE UPLOAD (BATCH) -------- #

st.divider()
st.subheader("📂 Batch Scoring (Upload Excel/CSV)")

file = st.file_uploader("Upload file", type=["csv", "xlsx"])

if file:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    X = prepare_data(df, columns)

    risk_scores = model.predict_proba(X)[:, 1]

    df["FoodInsecure"] = (risk_scores >= threshold).astype(int)
    df["RiskScore"] = risk_scores
    df["RiskCategory"] = df["RiskScore"].apply(
        lambda x: categorize_risk(x, threshold)
    )

    st.success("Scoring complete!")

    st.dataframe(df.head())

    st.download_button(
        "Download Results",
        df.to_csv(index=False),
        "scored_results.csv",
        "text/csv"
    )


#run in terminal using: 
# cd "D:\SCHOOL\Data Science\ZCAS\Computing Project\Model\Deployment"
# streamlit run hh_scorer_stream.py
# Then browser:
# http://localhost:8501