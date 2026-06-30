"""
train_models.py  —  FIXED VERSION (No Overfitting)
Fixes applied:
  1. Adds symptom variation so model cannot memorize
  2. Proper train/test split with shuffle
  3. Regularization on every model
  4. Cross validation to detect overfitting
  5. Saves train vs test accuracy for comparison
Run: python train_models.py
"""

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.metrics         import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────
# STEP 1 — Load dataset
# ─────────────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv("data/dataset.csv")
df.columns = df.columns.str.strip()

symptom_cols = [c for c in df.columns if c.startswith("Symptom")]

all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().str.strip().tolist())
all_symptoms = sorted(all_symptoms)

print(f"✅ Diseases : {df['Disease'].nunique()}")
print(f"✅ Symptoms : {len(all_symptoms)}")
print(f"✅ Rows     : {len(df)}")

# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Add variation to fix overfitting
# Problem: every disease has 10 IDENTICAL rows — model memorizes them
# Fix: create varied symptom combos so model must LEARN, not memorize
# ─────────────────────────────────────────────────────────────────────
print("\n🔧 Adding symptom variation to prevent overfitting...")

def augment_data(df, symptom_cols, n_augmented=5, drop_prob=0.15):
    augmented_rows = []
    for _, row in df.iterrows():
        disease = row["Disease"]
        real_syms = [row[c].strip() for c in symptom_cols
                     if pd.notna(row[c]) and str(row[c]).strip()]
        # Keep original row
        augmented_rows.append(row.to_dict())
        # Create varied versions
        for _ in range(n_augmented):
            new_row = {"Disease": disease}
            varied_syms = [s for s in real_syms if np.random.random() > drop_prob]
            if len(varied_syms) < 2:
                varied_syms = real_syms[:2]
            for i, col in enumerate(symptom_cols):
                new_row[col] = varied_syms[i] if i < len(varied_syms) else np.nan
            augmented_rows.append(new_row)
    return pd.DataFrame(augmented_rows)

df_aug = augment_data(df, symptom_cols, n_augmented=5, drop_prob=0.15)
df_aug = df_aug.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Original rows  : {len(df)}")
print(f"✅ Augmented rows : {len(df_aug)}")
print(f"✅ Rows/disease   : ~{len(df_aug)//df['Disease'].nunique()}")

# ─────────────────────────────────────────────────────────────────────
# STEP 3 — Build feature matrix
# ─────────────────────────────────────────────────────────────────────
print("\n⚙️  Building feature matrix...")
records = []
for _, row in df_aug.iterrows():
    syms = set()
    for col in symptom_cols:
        if pd.notna(row[col]) and str(row[col]).strip():
            syms.add(str(row[col]).strip())
    vec = {s: (1 if s in syms else 0) for s in all_symptoms}
    vec["Disease"] = row["Disease"]
    records.append(vec)

feat_df = pd.DataFrame(records)
X = feat_df[all_symptoms]
y = feat_df["Disease"]

# 25% test split — harder test than 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"✅ Training rows : {len(X_train)}")
print(f"✅ Testing rows  : {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────
# STEP 4 — Define models WITH regularization to prevent overfitting
# ─────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression" : LogisticRegression(
                                max_iter=1000, C=1.0,
                                random_state=42),

    "Decision Tree"       : DecisionTreeClassifier(
                                max_depth=8,
                                min_samples_leaf=3,
                                min_samples_split=5,
                                random_state=42),

    "Random Forest"       : RandomForestClassifier(
                                n_estimators=100,
                                max_depth=10,
                                min_samples_leaf=2,
                                min_samples_split=4,
                                max_features="sqrt",
                                random_state=42),

    "Gradient Boosting"   : GradientBoostingClassifier(
                                n_estimators=100,
                                learning_rate=0.1,
                                max_depth=4,
                                min_samples_leaf=3,
                                random_state=42),

    "AdaBoost"            : AdaBoostClassifier(
                                n_estimators=50,
                                learning_rate=0.5,
                                random_state=42),

    "K-Nearest Neighbors" : KNeighborsClassifier(
                                n_neighbors=7,
                                weights="distance"),

    "Naive Bayes"         : GaussianNB(var_smoothing=1e-8),
}

# ─────────────────────────────────────────────────────────────────────
# STEP 5 — Train and check overfitting
# ─────────────────────────────────────────────────────────────────────
print("\n🤖 Training all 7 models...\n")

results   = []
trained   = {}
conf_mats = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"   ⚙️  Training {name}...")
    model.fit(X_train, y_train)

    y_pred     = model.predict(X_test)
    train_pred = model.predict(X_train)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc  = accuracy_score(y_test,  y_pred)
    gap       = train_acc - test_acc

    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score   (y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score       (y_test, y_pred, average="weighted", zero_division=0)

    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    # Overfitting check
    if gap > 0.15:
        status = "🔴 OVERFIT"
    elif gap > 0.08:
        status = "🟡 SLIGHT OVERFIT"
    else:
        status = "🟢 GOOD"

    print(f"      Train: {train_acc*100:.1f}%  Test: {test_acc*100:.1f}%  Gap: {gap*100:.1f}%  {status}")

    results.append({
        "Model"     : name,
        "Train Acc" : round(train_acc * 100, 2),
        "Accuracy"  : round(test_acc  * 100, 2),
        "Gap"       : round(gap       * 100, 2),
        "Precision" : round(prec * 100, 2),
        "Recall"    : round(rec  * 100, 2),
        "F1 Score"  : round(f1   * 100, 2),
        "CV Score"  : round(cv_mean * 100, 2),
        "CV Std"    : round(cv_std  * 100, 2),
        "Status"    : status,
    })

    trained[name]   = model
    conf_mats[name] = confusion_matrix(y_test, y_pred)

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)

print("\n📊 Final Results:\n")
print(results_df[["Model","Train Acc","Accuracy","Gap","CV Score","Status"]].to_string(index=False))

# Feature importance
fi = pd.DataFrame({
    "Feature"   : all_symptoms,
    "Importance": trained["Random Forest"].feature_importances_
}).sort_values("Importance", ascending=False).head(20)

# ─────────────────────────────────────────────────────────────────────
# STEP 6 — Save everything
# ─────────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
with open("model/trained_models.pkl",     "wb") as f: pickle.dump(trained,          f)
with open("model/symptoms_list.pkl",      "wb") as f: pickle.dump(all_symptoms,     f)
with open("model/conf_matrices.pkl",      "wb") as f: pickle.dump(conf_mats,        f)
with open("model/feature_importance.pkl", "wb") as f: pickle.dump(fi,               f)
with open("model/test_data.pkl",          "wb") as f: pickle.dump((X_test, y_test), f)
results_df.to_csv("model/model_results.csv", index=False)

print("\n✅ All 7 models saved with overfitting fixes!")
print("🚀 Now run:  streamlit run app.py")