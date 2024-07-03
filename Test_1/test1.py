import pandas as pd
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = pd.read_csv("diabetes.csv")
# profile = ProfileReport(data, title="Diabetes Report", explorative=True)
# profile.to_file("report.html")

# Split data
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5426)

# Preprocessing
num_trans = Pipeline(steps=[
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_trans, ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
])
cls_model = Pipeline(steps=[
    ('pre', preprocessor),
    ('model', RandomForestClassifier(random_state=5426))
])
params = {
    "model__n_estimators": [50, 100, 200, 500],
    "model__criterion": ["gini", "entropy", "log_loss"],
    "model__max_depth": [None, 2, 5],
    "model__min_samples_split": [2, 5, 10]
}
model = GridSearchCV(estimator=cls_model, param_grid=params, scoring="recall_macro", cv=6, verbose=2,
                     n_jobs=6)

# Fit model
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# Model evaluation
print(classification_report(y_test, predictions))



