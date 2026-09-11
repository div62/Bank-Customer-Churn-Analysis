import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported successfully!")
df = pd.read_csv("Churn_Modelling.csv")

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Records:")
print(df.duplicated().sum())
print("\nStatistical Summary:")
print(df.describe())
print("\nChurn Distribution:")
print(df["Exited"].value_counts())
print("\nChurn Percentage:")
print(df["Exited"].value_counts(normalize=True) * 100)

sns.countplot(x="Exited", data=df)
plt.title("Customer Churn Distribution")
plt.xlabel("Exited (0 = Stayed, 1 = Churned)")
plt.ylabel("Number of Customers")
plt.savefig("churn_distribution.png")
plt.show()
print("\nChurn by Geography:")
print(df.groupby("Geography")["Exited"].sum())

sns.countplot(x="Geography", hue="Exited", data=df)
plt.title("Customer Churn by Geography")
plt.xlabel("Country")
plt.ylabel("Number of Customers")
plt.savefig("churn_by_geography.png")
plt.show()
print("\nChurn by Gender:")
print(df.groupby("Gender")["Exited"].sum())
sns.countplot(x="Gender", hue="Exited", data=df)
plt.title("Customer Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.savefig("churn_by_gender.png")
plt.show()

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[0, 30, 40, 50, 60, 100],
    labels=["Under 30", "30-40", "40-50", "50-60", "60+"]
)

print("\nChurn by Age Group:")
print(df.groupby("AgeGroup", observed=True)["Exited"].sum())
sns.countplot(x="AgeGroup", hue="Exited", data=df)
plt.title("Customer Churn by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Number of Customers")
plt.savefig("churn_by_age_group.png")
plt.show()

df["CreditScoreGroup"] = pd.cut(
    df["CreditScore"],
    bins=[300, 500, 600, 700, 800, 900],
    labels=["300-500", "500-600", "600-700", "700-800", "800-900"]
)

print("\nChurn by Credit Score Group:")
print(df.groupby("CreditScoreGroup", observed=True)["Exited"].sum())

sns.countplot(x="CreditScoreGroup", hue="Exited", data=df)

plt.title("Customer Churn by Credit Score")
plt.xlabel("Credit Score Group")
plt.ylabel("Number of Customers")
plt.savefig("churn_by_credit_score.png")
plt.show()
print("\nChurn by Active Membership:")
print(df.groupby("IsActiveMember")["Exited"].sum())

sns.countplot(x="IsActiveMember", hue="Exited", data=df)
plt.title("Customer Churn by Active Membership")
plt.xlabel("Active Member (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.savefig("churn_by_active_membership.png")
plt.show()
print("\nChurn by Number of Products:")
print(df.groupby("NumOfProducts")["Exited"].sum())

sns.countplot(x="NumOfProducts", hue="Exited", data=df)
plt.title("Customer Churn by Number of Products")
plt.xlabel("Number of Products")
plt.ylabel("Number of Customers")
plt.savefig("churn_by_products.png")
plt.show()

print("\nAverage Balance by Churn Status:")
print(df.groupby("Exited")["Balance"].mean())

sns.boxplot(x="Exited", y="Balance", data=df)
plt.title("Balance vs Customer Churn")
plt.xlabel("Exited (0 = Stayed, 1 = Churned)")
plt.ylabel("Balance")
plt.savefig("churn_by_balance.png")
plt.show()
print("\nChurn by Tenure:")
print(df.groupby("Tenure")["Exited"].sum())

sns.countplot(x="Tenure", hue="Exited", data=df)
plt.title("Customer Churn by Tenure")
plt.xlabel("Tenure (Years)")
plt.ylabel("Number of Customers")
plt.savefig("churn_by_tenure.png")
plt.show()

print("\nAverage Salary by Churn Status:")
print(df.groupby("Exited")["EstimatedSalary"].mean())

sns.boxplot(x="Exited", y="EstimatedSalary", data=df)
plt.title("Estimated Salary vs Customer Churn")
plt.xlabel("Exited (0 = Stayed, 1 = Churned)")
plt.ylabel("Estimated Salary")
plt.savefig("churn_by_salary.png")
plt.show()

print("\nAverage Salary by Churn Status:")
print(df.groupby("Exited")["EstimatedSalary"].mean())

sns.boxplot(x="Exited", y="EstimatedSalary", data=df)
plt.title("Estimated Salary vs Customer Churn")
plt.xlabel("Exited (0 = Stayed, 1 = Churned)")
plt.ylabel("Estimated Salary")
plt.savefig("churn_by_salary.png")
plt.show()

print("\nCorrelation with Churn:")
print(df.corr(numeric_only=True)["Exited"].sort_values(ascending=False))

plt.figure(figsize=(10, 6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

print("\n========== KEY FINDINGS ==========")

print("1. Overall churn rate is 20.37%.")
print("2. Customers aged 40-50 show a high number of churned customers.")
print("3. Germany has a relatively high churn rate compared with other countries.")
print("4. Active members are less likely to churn than inactive members.")
print("5. Customers with certain numbers of products show different churn patterns.")
print("6. Credit score and balance have some relationship with customer churn.")
print("7. Estimated salary shows only a small difference between stayed and churned customers.")

print("\n========== BUSINESS RECOMMENDATIONS ==========")

print("1. Focus on retaining customers who show a high risk of churn.")
print("2. Provide special offers and support to inactive customers.")
print("3. Investigate why customers in Germany have higher churn.")
print("4. Create targeted retention programs for middle-aged customers.")
print("5. Study customer product usage and provide suitable product recommendations.")
print("6. Monitor customer behavior regularly to identify early signs of churn.")