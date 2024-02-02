
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Original link (modified for direct download)
url = r"C:\Users\14040\Desktop\test/org_role_100K.txt"

# Assuming the file is a CSV or TSV, we'll try reading it with common delimiters
try:
    # Trying to read as a CSV
    df = pd.read_csv(url)
except pd.errors.ParserError:
    # If CSV parsing fails, trying as a TSV
    df = pd.read_csv(url, delimiter='\t')


# see what unique the data set has on role_original
unique_roles = df["role_original"].unique()

# only preserve unique value on role_original
df.drop_duplicates(subset='role_original', keep='first', inplace=True)

# change all role_original to lowercase 
df['role_original'] = df['role_original'].str.lower()


#drop all duplicates 
df.drop_duplicates(subset='role_original', keep='first', inplace=True)

# handing missing value, replace "?" to NaN#
df.replace("?", pd.NA, inplace=True)

df.dropna(inplace=True)

# handing missing value, replace "-" to NaN#
df.replace("-", pd.NA, inplace=True)
df.dropna(inplace=True)


#identified keyword for each department, left hand side is value, right hand side of the colon is value

keyword_dict = {
    "Customer Service": ["Customer", "Consultant", "Sales", "Advisor", "Agent", "personal banker", 
                         "service", "client", "gestor clientèle", "planner", "private", "Cashier", 
                         "Realtor", "Adviseur", "CSR", "Solutions", "Crew", "Preservation"],
    "Marketing": ["Analyst", "Audit", "Auditor"],
    "Administration": ["Director", "president", "vp", "svp", "Partner", "Administrative", "Officer", 
                       "Eno", "Administrativo", "Authorized", "Recruiter"],
    "Claims": ["Claims"],
    "Finance": ["Finance", "teller", "accountant", "fund", "financiero", "Conseiller", "Fraud", 
                "Underwriter", "Mortgage", "Credit", "Financiero", "Paraplanner", "Assurance", 
                "Brokerage", "Foreclosure", "Document"],
    "IT": ["Software", "engineer", "technology", "developer", "programmer", "Application", "Specialist", 
           "development", "Project", "Analyst", "Process", "AML", "Data", "Technical"],
    "Intern": ["Intern", "trainee", "assistant", "auxiliaire", "Probationary", "Chargé D'Affaires", 
               "Stagiaire", "Estagiário"],
    "Management": ["Agency", "Coordinator", "associate", "Manager", "supervisor", "d'agence", "Leader", 
                   "Store", "Executive"],
    "Retired": ["Retired"],
    "Loan": ["Loan"],
    "Hospitality Services": ["Barista", "Sandwich", "Driver", "Package", "Handler"]
}

#import a excel contain all 336 row 

url1 = r"C:\Users\14040\Desktop\test/job_titles_departments.xlsx"
try:
    # Trying to read as a excel
    job_titles_departments = pd.read_excel(url1)
except pd.errors.ParserError:
    # If excel parsing fails, trying as a TSV
    job_titles_departments = pd.read_excel(url1, delimiter='\t')

# Display the first few rows of the DataFrame
job_titles_departments['Job Title'] = job_titles_departments['Job Title'].str.lower()

# Initialize columns for each department in the keyword dictionary
for department in keyword_dict:
    job_titles_departments[department] = 0

# Feature creation: mark 1 if any keyword of a department is found in the job title
for department, keywords in keyword_dict.items():
    for keyword in keywords:
        job_titles_departments[department] |= job_titles_departments['Job Title'].str.contains(keyword.lower(), case=False, regex=False)


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
X = job_titles_departments[keyword_dict.keys()]
y = job_titles_departments['Department']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
evaluation_report = classification_report(y_test, y_pred)

print(evaluation_report)



