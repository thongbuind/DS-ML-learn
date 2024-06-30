import pandas as pd
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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
    ('model', RandomForestClassifier())
])

# Fit model
cls_model.fit(x_train, y_train)
predictions = cls_model.predict(x_test)

# Model evaluation
print(classification_report(y_test, predictions))
