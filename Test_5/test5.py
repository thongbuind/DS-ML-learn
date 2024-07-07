import pandas as pd
# from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report

data = pd.read_csv("stroke.csv")
# profile = ProfileReport(data, title="Stroke Report", explorative=True)
# profile.to_file("StrokeReport.html")
data = data.drop(data[data['gender'] == 'Other'].index)
target = 'stroke'
x = data.drop(target, axis=1)
x = x.drop('pat_id', axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5426)

# corr = data[[target, "age", 'avg_glucose_level', 'bmi']].corr()
# print(corr)
#                         stroke       age  avg_glucose_level       bmi
#    stroke             1.000000  0.245239           0.131991  0.042341
#    age                0.245239  1.000000           0.238323  0.333314
#    avg_glucose_level  0.131991  0.238323           1.000000  0.175672
#    bmi                0.042341  0.333314           0.175672  1.000000
# ==> Chon mo hinh cls

# Preprocessing
num_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
gender = ['Female', 'Male']
hypertension = [0, 1]
heart_disease = [0, 1]
work_related_stress = [0, 1]
urban_residence = [0, 1]
smokes = [0, 1]
ord_trans = Pipeline(steps=[
    ('encoder', OrdinalEncoder(categories=[gender, hypertension, heart_disease, work_related_stress, urban_residence, smokes]))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_trans, ['age', 'avg_glucose_level', 'bmi']),
    ('bool', ord_trans, ['gender', 'hypertension', 'heart_disease', 'work_related_stress', 'urban_residence', 'smokes'])
])

cls_model = ImbPipeline([
    ('pre', preprocessor),
    # ('balance', SMOTE(random_state=5426)),
    ('balance', RandomOverSampler(random_state=5426)),
    ('model', RandomForestClassifier(random_state=5426, criterion='entropy', max_depth=5, min_samples_split=2, n_estimators=50))
])
# params = {
#     "model__n_estimators": [50, 100, 200, 500],
#     "model__criterion": ["gini", "entropy", "log_loss"],
#     "model__max_depth": [None, 2, 5],
#     "model__min_samples_split": [2, 5, 10]
# }
# model = GridSearchCV(estimator=cls_model, param_grid=params, scoring="recall_macro", cv=6, verbose=2,
#                      n_jobs=8)
cls_model.fit(x_train, y_train)
# print("Best parameters found by GridSearchCV:")
# print(model.best_params_)
predictions = cls_model.predict(x_test)

print(classification_report(y_test, predictions))



