import pandas as pd
import gradio as gr
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data.csv")

# Keep only top features
categorical = ["A site #1", "B site #1", "X site"]
numerical = [
    "Number of elements",
    "Density_AB_avg",
    "Ionization Energy (kJ/mol)_AB_avg",
    "Atomic Volume (cm³/mol)_AB_avg"
]
target = "formation_energy (eV/atom)"

# Drop NaNs and prepare
df = df.dropna(subset=[target])
df = df[categorical + numerical + [target]]
X = df[categorical + numerical]
y = df[target]

# Preprocessing and model
preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ]), numerical)
])

model = Pipeline([
    ("prep", preprocessor),
    ("reg", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
model.fit(X_train, y_train)
r2 = r2_score(y_test, model.predict(X_test))

# Prediction + plot
def predict_and_plot(a1, b1, x, num_elem, density, ion_energy, atomic_vol):
    data = {
        "A site #1": a1,
        "B site #1": b1,
        "X site": x,
        "Number of elements": float(num_elem),
        "Density_AB_avg": float(density),
        "Ionization Energy (kJ/mol)_AB_avg": float(ion_energy),
        "Atomic Volume (cm³/mol)_AB_avg": float(atomic_vol)
    }
    df_input = pd.DataFrame([data])
    pred = model.predict(df_input)[0]

    # Stability logic
    if pred < -1.0:
        status = "🟢 Stable"
    elif -1.0 <= pred <= 0.5:
        status = "🟡 Metastable"
    else:
        status = "🔴 Unstable"

    # Plot
    fig, ax = plt.subplots()
    ax.barh(["Formation Energy"], [pred], color="green" if pred < -1 else "orange" if pred <= 0.5 else "red")
    ax.set_xlim(-3, 2)
    ax.set_xlabel("eV/atom")
    ax.set_title(f"Prediction: {round(pred, 4)} eV/atom — {status}")
    plt.tight_layout()

    return round(pred, 5), status, fig

# Inputs
inputs = [
    gr.Textbox(label="A site #1"),
    gr.Textbox(label="B site #1"),
    gr.Textbox(label="X site"),
    gr.Number(label="Number of elements", value=5),
    gr.Number(label="Density_AB_avg", value=5.5),
    gr.Number(label="Ionization Energy (kJ/mol)_AB_avg", value=700),
    gr.Number(label="Atomic Volume (cm³/mol)_AB_avg", value=10.0)
]

# Interface
demo = gr.Interface(
    fn=predict_and_plot,
    inputs=inputs,
    outputs=[
        gr.Number(label="Predicted Formation Energy (eV/atom)"),
        gr.Text(label="Stability Status"),
        gr.Plot(label="Stability Visualization")
    ],
    title="Formation Energy Predictor",
    description=(
        "🎯 This tool predicts the **formation energy** (eV/atom) of a compound "
        "based on elemental and physical properties.\n\n"
        "**Interpretation**:\n"
        "- 🟢 Low/Negative → Stable\n"
        "- 🟡 Close to Zero → Metastable\n"
        "- 🔴 Positive → Unstable\n\n"
        f"📈 Model trained with R² score: **{round(r2, 4)}**"
    )
)

if __name__ == "__main__":
    demo.launch()
