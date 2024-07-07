import pandas as pd
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

data = pd.read_excel("diabetes_data.xlsx")
# profile = ProfileReport(data, title="Diabetes_dataReport", explorative=True)
# profile.to_file("Diabetes_dataReport.html")
target = "DiabeticClass"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5426)

# Preprocessing
num_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
gender = ['Male', 'Female']
excessUrination = ['Yes', 'No']
polydipsia = ['Yes', 'No']
weightLossSudden = ['Yes', 'No']
fatigue = ['Yes', 'No']
polyphagia = ['Yes', 'No']
genitalThrush = ['Yes', 'No']
blurredVision = ['Yes', 'No']
itching = ['Yes', 'No']
irritability = ['Yes', 'No']
delayHealing = ['Yes', 'No']
partialPsoriasis = ['Yes', 'No']
muscleStiffness = ['Yes', 'No']
alopecia = ['Yes', 'No']
obesity = ['Yes', 'No']
ord_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[gender, excessUrination, polydipsia, weightLossSudden, fatigue, polyphagia, genitalThrush, blurredVision, itching, irritability, delayHealing, partialPsoriasis, muscleStiffness, alopecia, obesity]))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_trans, ['Age']),
    ('ord', ord_trans, ['Gender', 'ExcessUrination', 'Polydipsia', 'WeightLossSudden', 'Fatigue', 'Polyphagia', 'GenitalThrush', 'BlurredVision', 'Itching', 'Irritability', 'DelayHealing', 'PartialPsoriasis', 'MuscleStiffness', 'Alopecia', 'Obesity'])
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



