import pandas as pd

# Load dataset
df = pd.read_csv("ResumeDataset.csv.csv")

print("Dataset Loaded Successfully!")
print("Total Resumes:", len(df))
print("Columns:", df.columns.tolist())

print("\nJob Categories:")
print(df["Category"].unique())

print("\nNumber of Categories:", df["Category"].nunique())
