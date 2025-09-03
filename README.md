##  Formation Energy Predictor

**FormationEnergyPredictor** is a machine learning web application that predicts the **formation energy (eV/atom)** of perovskite-based materials using their atomic and physical properties. It also classifies the material as **Stable**, **Metastable**, or **Unstable**, helping materials scientists rapidly screen novel compositions for energy, electronics, and photovoltaic applications.

## 🔬 What It Does

* Takes in a material’s **A site**, **B site**, and **X site** elements and a few physical properties
* Predicts the **formation energy** using a trained machine learning model
* Classifies material stability:

  * 🟢 Stable (low/negative formation energy)
  * 🟡 Metastable (near-zero formation energy)
  * 🔴 Unstable (positive formation energy)
* Visualizes the result on a stability scale

---

## How It Was Built

This project was built using:

* **Python & Scikit-learn**: for training a Gradient Boosting model
* **Gradio**: for building the web interface
* **Matplotlib**: to visualize formation energy in a stability chart
* **Hugging Face Spaces**: to host the application online

### 🔧 Workflow

1. **Data Source**: A curated dataset of perovskite materials with elemental properties and formation energies.
2. **Feature Engineering**:

   * Selected top features including:

     * `A site #1`, `B site #1`, `X site`
     * `Number of elements`
     * `Density_AB_avg`
     * `Ionization Energy_AB_avg`
     * `Atomic Volume_AB_avg`
3. **Modeling**:

   * Used `GradientBoostingRegressor` inside a preprocessing pipeline
   * Trained with 80/20 train-test split and achieved a good R² score
4. **Deployment**:

   * Packaged with `gradio` and deployed to Hugging Face using `requirements.txt`

---

##  Use Case & Problem Solved

### Challenge:

In materials science, **discovering new stable compounds** is time-consuming and expensive. Researchers often simulate or synthesize hundreds of compositions before finding one that works.

### This app solves:

* Fast **pre-screening** of candidate perovskite materials
* Prediction of **stability** before physical experiments or simulations
* Reduction in **cost and time** spent on unstable materials
* Supports **green energy** and **optoelectronic research**

---

## How to Use

1. Visit the app: [[FormationEnergyPredictor on Hugging Face](https://huggingface.co/spaces/DanielEmeka/FormationEnergyPredictor)](https://huggingface.co/spaces/DanielEmeka/FormationEnergyPredictor)

2. Fill in:

   * A site element (e.g., `Na`)
   * B site element (e.g., `Ti`)
   * X site element (e.g., `O`)
   * Additional properties like density, ionization energy, etc.
3. Click **Submit**
4. View:

   * Formation energy prediction (eV/atom)
   * Stability classification
   * Visual chart showing result

---

## Examples to Try

| A site | B site | X site | Ionization Energy | Stability     |
| ------ | ------ | ------ | ----------------- | ------------- |
| Na     | Ti     | O      | 640               | 🟢 Stable     |
| Cs     | Sn     | I      | 700               | 🟡 Metastable |
| Li     | Pb     | Br     | 850               | 🔴 Unstable   |

---

## Key Feature Insights

From the trained model, these features were found most important:

* **Ionization Energy** — higher values lead to instability
* **Density and Atomic Volume** — influence bonding compactness
* **A, B, X site identity** — define lattice behavior in perovskites

---

## Future Improvements

* Add support for ABX₂ and double perovskite structures
* Enable batch predictions via file upload
* Integrate SHAP or permutation importance plots for full explainability
* Add default dropdown menus and example presets

---

## Author

**Daniel Emeka**
Physicist | Nanomaterials Researcher | Machine Learning Enthusiast
Built for material discovery and sustainability-driven innovation.
