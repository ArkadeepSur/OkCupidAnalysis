import pandas as pd


def Check_Dataframe(dataframe):
   print("##########COMPLETE INFO#######################")
   print("Head\n", dataframe.head())
   print("\nInfo\n")
   dataframe.info()
   print("\nDescribe\n", dataframe.describe(include='all'))
   print("\nIsNull\n", dataframe[dataframe.isnull().any(axis = 1)])
   print("#################################")

df = pd.read_csv("profiles.csv")
# Set the option to display all columns
#pd.set_option('display.max_columns', None)
Check_Dataframe(df)