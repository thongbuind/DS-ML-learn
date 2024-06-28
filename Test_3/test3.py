import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


data = pd.read_csv("StudentScore.xls")
print(data)
# profile = ProfileReport(data, title="StudentScore Report", explorative=True)
# profile.to_file("ScoreReport.html")
# corr = data[[target, "writing score", "reading score"]].corr()
# print(corr)
    #                    math score  writing score  reading score
    # math score       1.000000       0.802642       0.817580
    # writing score    0.802642       1.000000       0.954598
    # reading score    0.817580       0.954598       1.000000
    # ==> Hệ số tương quan lớn, dùng regression
target = "math score"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5426)

# Preprocessing
numerical_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
education_value = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
gender = ["male", "female"]
lunch = ["standard", "free/reduced"]
course = ["none", "completed"]
ordinal_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[education_value, gender, lunch, course]))
])
nominal_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('encoder', OneHotEncoder())
])


