'''IMPORTING DATA'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

'''LOAD DATASET'''
df = pd.read_csv('cars.csv')

'''CHECKING THE FIRST FEW LINES OF COLUMN AND INFORMATION'''
df.head()
df.info()

#There are cubicinches and weightlbs are numeric columns but it's datatypes are in the object so I wil convert it into a int

'''CONVERTING TWO COLUMNS IN THE INT FROM THE OBJECT DATATYPES'''
df['cubicinches'] = pd.to_numeric(df['cubicinches'], errors='coerce')
df['weightlbs'] = pd.to_numeric(df['weightlbs'], errors='coerce')

'''IF THERE WERE ANY UNCOVERTIBLE VALUE SUCH AS SPACE IS THERE REPLACE WITH NAN'''
df.replace(" ", np.nan, inplace=True)

'''CHECKING NULL VALUES PRESENT INTO THE DATA'''
df.isnull().sum()

#Here null values are very negligible so I will remove it.

'''DROPING THE NULL VALUES'''
df.dropna(inplace=True)

'''PLOTING BOXPLOTS TO CHECK IS THERE ANY OUTLIERS ARE PRESENT OR NOT'''
for i in df.columns:
    if df[i].dtype!= "object":
        plt.boxplot(df[i])
        plt.title(i)
        plt.show()

#In this boxplot time-to-60 have some outliers but because of small dataset I will not remove it.

'''ENCODING THE DATA WHICH ARE OBJECT IN DATATYPES'''
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])
#here my only last brand column is categorical so i perform label encoder

'''SCALLING THE DATA IN PARTICULAR ONE RANGE'''
sc = StandardScaler()
df[df.columns] = sc.fit_transform(df)

'''CHECKING THE FEW COLUMNS OF THE DATA TO CHECK ALL THE CHANGES ARE DONE OR NOT'''
df.head()

'''CHECKING CORRELATION '''
df.corr()

'''PLOTING HEATMAP OF CORRELATION TO SEE WHICH VALUES OF COLUMNS ARE GREATER THAN 0.5 AND REMOVE IT'''
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(5,2))
sns.heatmap(df.corr(),cmap='Blues',annot=True)

#Here 'cubicinches','cylinders' columns are highly correlated but my dataset is small so removing that columns can cause data loss

#Here all ETL process is done
'''THANK YOU '''

