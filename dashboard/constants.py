DATA_PATH = "data/Training_Liver_Disease_Dataset.csv"

NUMERICAL = [
    "Age", "BMI", "Waist_Circumference", "Sleep_Hours",
    "ALT", "AST", "Bilirubin", "Albumin", "Platelets",
    "Alk_Phosphatase", "GGT", "Triglycerides", "INR",
]
BINARY = [
    "Sym_Fatigue", "Sym_Jaundice", "Sym_Abdominal_Pain", "Sym_Itching",
    "Sym_Ascites", "Sym_Dark_Urine", "Sym_Weight_Loss",
    "Comorb_Diabetes", "Comorb_Hypertension", "Comorb_Genetic_History",
]
CATEGORICAL = [
    "Gender", "Occupation", "Obesity_Class", "Diet_Quality",
    "Physical_Activity", "Smoking_Status", "Alcohol_Consumption",
    "Medication_History", "Source",
]
# Exact column order the models were trained on (all features, target excluded)
FEATURE_ORDER = [
    "Age", "Gender", "Occupation", "BMI", "Obesity_Class",
    "Waist_Circumference", "Diet_Quality", "Physical_Activity",
    "Sleep_Hours", "Smoking_Status", "Alcohol_Consumption",
    "Sym_Fatigue", "Sym_Jaundice", "Sym_Abdominal_Pain", "Sym_Itching",
    "Sym_Ascites", "Sym_Dark_Urine", "Sym_Weight_Loss",
    "Comorb_Diabetes", "Comorb_Hypertension", "Comorb_Genetic_History",
    "ALT", "AST", "Bilirubin", "Albumin", "Platelets",
    "Alk_Phosphatase", "GGT", "Triglycerides", "INR",
    "Medication_History", "Source",
]
TARGET = "Liver_Disease_Class"

# LabelEncoder sorts alphabetically → indices match these class names
CLASS_LABELS = {
    0: "Alcoholic Liver Disease",
    1: "Fatty Liver Disease (NAFLD)",
    2: "General Liver Disease Severity",
    3: "Healthy Liver",
    4: "Liver Cirrhosis Risk",
}
CLASS_COLORS = {
    "Healthy Liver": "#2ecc71",
    "Fatty Liver Disease (NAFLD)": "#f39c12",
    "Alcoholic Liver Disease": "#e74c3c",
    "General Liver Disease Severity": "#9b59b6",
    "Liver Cirrhosis Risk": "#c0392b",
}
BINARY_LABELS = {
    "Sym_Fatigue": "Fatigue",
    "Sym_Jaundice": "Jaundice",
    "Sym_Abdominal_Pain": "Abdominal Pain",
    "Sym_Itching": "Itching",
    "Sym_Ascites": "Ascites",
    "Sym_Dark_Urine": "Dark Urine",
    "Sym_Weight_Loss": "Weight Loss",
    "Comorb_Diabetes": "Diabetes",
    "Comorb_Hypertension": "Hypertension",
    "Comorb_Genetic_History": "Genetic History",
}
MODEL_META = {
    "KNN (k=5)": {"accuracy": 0.8928, "file": "KNN_liver_model.pkl"},
    "Decision Tree": {"accuracy": 0.9410, "file": "dtree_liver_model.pkl"},
}
