import pandas as pd

csv = "testsubmission.csv"
df = pd.read_csv(csv)
print("Total profit/loss: " + str(round(df["profit_and_loss"].sum()))
      + "\nAvg profit/loss: " + str(round(df["profit_and_loss"].mean(), 2)))
