import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score


data = pd.read_csv("StudentScore.xls")
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
num_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
education_value = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
gender = ["male", "female"]
lunch = ["standard", "free/reduced"]
course = ["none", "completed"]
ord_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=[education_value, gender, lunch, course]))
])
nom_trans = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder())
])
preprocessor = ColumnTransformer(transformers=[
    ('num', num_trans, ['reading score', 'writing score']),
    ('nom', nom_trans, ['race/ethnicity']),
    ('ord', ord_trans, ['parental level of education', 'gender', 'lunch', 'test preparation course'])
])

# Select model
reg_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Fit model
reg_model.fit(x_train, y_train)
predictions = reg_model.predict(x_test)

# Model evaluation
print("R2_score: {}".format(r2_score(y_test, predictions)))


