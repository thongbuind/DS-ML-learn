import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Đọc data, chia thành bộ train/test
data = pd.read_csv("csgo.csv")
target = "result"
x = data.drop(target, axis=1)
x = x.drop("day", axis=1)
x = x.drop("month", axis=1)
x = x.drop("year", axis=1)
x = x.drop("date", axis=1)
x = x.drop("wait_time_s", axis=1)
x = x.drop("match_time_s", axis=1)
x = x.drop("team_a_rounds", axis=1)
x = x.drop("team_b_rounds", axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5426)

#Preprocessing
numerical_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
nominal_trans = Pipeline(steps= [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_trans, ['ping', 'kills', 'assists', 'deaths', 'mvps', 'hs_percent', 'points']),
    ('nor', nominal_trans, ['map'])
])

#Select model
pipeline = Pipeline (steps=[
    ('preprocessor', preprocessor),
    ('classification', RandomForestClassifier())
])

# Fit model
pipeline.fit(x_train, y_train)
predictions = pipeline.predict(x_test)

# Model evaluation
print(classification_report(y_test, predictions))

