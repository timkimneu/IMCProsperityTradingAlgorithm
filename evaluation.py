import pandas as pd
import matplotlib.pyplot as plt

csv = "testsubmit4.csv"
df = pd.read_csv(csv)

print("Total profit/loss: " + str(round(df["profit_and_loss"].sum()))
      + "\nAvg profit/loss: " + str(round(df["profit_and_loss"].mean(), 2))
      + "\nMode: " + df["profit_and_loss"].mode().to_string())

starfruit_midprices = list()

for i in range(df["product"].count()):
    if df["product"][i] == "STARFRUIT":
        starfruit_midprices.append(df["mid_price"][i])

starfruit_dict = {"starfruit_mid_price": starfruit_midprices}

df2 = pd.DataFrame(starfruit_dict)

df2["starfruit_mid_price"] = (df2["starfruit_mid_price"] // 10) * 10

ax = df2["starfruit_mid_price"].value_counts(sort=False).plot(kind='barh')

plt.xlabel("Number of Starfruit")
plt.ylabel("Starfruit Mid Price")
plt.show()
