from Utilities import *
from KNNImpute import *

labels=["country", "sku_id", "title", "category_lvl1","category_lvl2","category_lvl3", "description", "price", "type"]

def getTrainingDataset():
    return pd.read_csv('data_train.csv', header=None, names=labels)  


def NullStatistics(df):
    missing_val = df.isnull().sum()
    print(missing_val)
    total_cells = np.product(df.shape)
    missing_percent = (missing_val.sum()/total_cells) *100
    print(f'\nThe missing data percent is: {missing_percent}')

trainingDataset=getTrainingDataset()
print(trainingDataset)

train_df,Y1,Y2,Y3=Cleaning_Data_Utility(trainingDataset)
print(train_df)

NullStatistics(train_df)

unique_label_c1, unique_label_c2, unique_label_c3 = preserve_label(train_df) #It has all unique values lying in ctg1 , 2 , 3 column

encode(['category_lvl1', 'category_lvl2', 'category_lvl3'],train_df)        #Performed encoding for CTGLVL3 KNN

train_df_imputed = impute(train_df)
train_df_imputed = clean_csv(train_df_imputed,train_df)
NullStatistics(train_df_imputed)