import missingno
import pandas as pd
# --------------------------------------------------------------
# Load and clean train data
# --------------------------------------------------------------

train = pd.read_csv("../../data/raw/train.csv")
test = pd.read_csv("../../data/raw/test.csv")

combine = pd.concat([train, test], axis = 0).reset_index(drop = True)
combine.head()

combine.info()
combine.nunique()
combine.describe()
# missingno.matrix(combine)
# combine.isnull().sum()#.sort_values(ascending = False)


# fill NaN with mostcommonPlanet value
mostcommon_planet = combine.HomePlanet.value_counts().index[0]
combine.HomePlanet = combine.HomePlanet.fillna(mostcommon_planet)

# fill NaN with mostcommon_Destination value
mostcommon_Destination = combine.Destination.value_counts().index[0]
combine.Destination = combine.Destination.fillna(mostcommon_Destination)

# fill NaN with mostcommon_CryoSleep value
mostcommon_CryoSleep = combine.CryoSleep.value_counts().index[0]
combine.CryoSleep = combine.CryoSleep.fillna(mostcommon_CryoSleep)


# fill NaN with mostcommon_VIP value
mostcommon_VIP = combine.VIP.value_counts().index[0]
combine.VIP = combine.VIP.fillna(mostcommon_VIP)

# fill NaN with mostcommon_Cabin value
combine.Cabin = combine.Cabin.fillna('F/82/S')
combine['Cabin'] = combine['Cabin'].astype('string')
combine[['Deck','Number','Side']] = combine['Cabin'].str.split('/', expand=True)
combine.drop(['Number'],axis=1, inplace=True)


# fill name NaN with U for unknown
combine.Name = combine.Name.fillna('U')

# fill mising values with grouped median/mean
grouped = combine.groupby(['HomePlanet', 'CryoSleep', 'Destination','VIP'])
combine['Age'] = grouped['Age'].transform(lambda x: x.fillna(x.median()))
combine['RoomService'] = grouped['RoomService'].transform(lambda x: x.fillna(x.mean()))
combine['FoodCourt'] = grouped['FoodCourt'].transform(lambda x: x.fillna(x.mean()))
combine['ShoppingMall'] = grouped['ShoppingMall'].transform(lambda x: x.fillna(x.mean()))
combine['VRDeck'] = grouped['VRDeck'].transform(lambda x: x.fillna(x.mean()))
combine['Spa'] = grouped['Spa'].transform(lambda x: x.fillna(x.mean()))



missingno.matrix(combine)
combine.isnull().sum()#.sort_values(ascending = False)

# New feature Tota Spent
combine['TotalSpent'] = combine.RoomService + combine.FoodCourt + combine.ShoppingMall + combine.VRDeck + combine.Spa



# --------------------------------------------------------------    
# Export dataset
# --------------------------------------------------------------

combine.to_csv("../../data/interim/combine_cleaned.csv", index=False)
