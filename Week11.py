import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("database/Iris.csv")

print(iris.head())
print(iris.info())
print(iris.describe())
print(iris["Species"].value_counts())

# Histograms
iris.drop(columns="Id").hist(figsize=(10, 8), bins=12)
plt.suptitle("Histograms of Iris Measurements")
plt.show()

# Scatter plot coloured by species
colors = {"Iris-setosa": "red", "Iris-versicolor": "green", "Iris-virginica": "blue"}
plt.scatter(iris["SepalLengthCm"], iris["SepalWidthCm"],
            c=iris["Species"].map(colors))
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Sepal Length against Sepal Width")
plt.show()

# Box plot
iris.boxplot(column="PetalLengthCm", by="Species")
plt.title("Petal Length across Species")
plt.suptitle("")
plt.show()
