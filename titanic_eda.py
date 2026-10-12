
# TITANIC DATASET - EXPLORATORY DATA ANALYSIS (EDA)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# STEP 1: Load the Titanic dataset
df = sns.load_dataset("titanic")

print("TITANIC DATASET - FIRST 5 ROWS")
print(df.head())

# STEP 2: Understand the dataset
print("\nDataset Shape (Rows, Columns):")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# STEP 3: Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# STEP 4: Remove duplicate rows
df = df.drop_duplicates()

# STEP 5: Handle missing values

# Replace missing age values with the median age
df["age"] = df["age"].fillna(df["age"].median())

# Replace missing embarkation ports with the most common port
df["embarked"] = df["embarked"].fillna(
    df["embarked"].mode()[0]
)

# Replace missing town names with the most common town
df["embark_town"] = df["embark_town"].fillna(
    df["embark_town"].mode()[0]
)

# Replace missing fare values with the median fare
df["fare"] = df["fare"].fillna(df["fare"].median())

# Fill missing cabin deck information with "Unknown"
df["deck"] = df["deck"].cat.add_categories(
    ["Unknown"]
).fillna("Unknown")

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# STEP 6: Analyze survival
print("\nSurvival Counts (0 = No, 1 = Yes):")
print(df["survived"].value_counts())

print("\nSurvival Percentage:")
print((df["survived"].value_counts(normalize=True) * 100).round(2))

# STEP 7: Visualize survival counts
sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="survived")
plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# STEP 8: Survival by gender
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="sex", y="survived")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# STEP 9: Survival by passenger class
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="class", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# STEP 10: Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=25, kde=True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# STEP 11: Survival by gender and class
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="class",
    y="survived",
    hue="sex"
)
plt.title("Survival by Gender and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

# STEP 12: Correlation heatmap
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# STEP 13: Save the cleaned dataset
df.to_csv("titanic_cleaned.csv", index=False)

print("\nEDA Completed Successfully!")
print("Cleaned dataset saved as titanic_cleaned.csv")
