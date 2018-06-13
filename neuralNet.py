from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


# "PercentMen", "Hispanic", "White", "Black", "Native", "Asian", "Pacific", "Income", "IncomePerCap", "Poverty",
#      "ChildPoverty", "Professional", "Service", "Office", "Construction", "Production", "Drive", "Carpool", "Transit",
#      "Walk", "OtherTransp", "WorkAtHome", "MeanCommute", "PercentEmployed", "PrivateWork", "PublicWork",
#      "SelfEmployed", "FamilyWork", "Unemployment"

# fix random seed for reproducibility
np.random.seed(7)


dataframe = pd.read_csv("/Users/lulucole/PycharmProjects/GunViolencePrediction/GunViolencePrediction_ML/census_county_counts.csv")
dataframe["PercentMen"] = dataframe["Men"] / dataframe["TotalPop"]
dataframe["PercentEmployed"] = dataframe["Employed"] / dataframe["TotalPop"]
#dataframe["IncidentsPerCap"] = dataframe["incident_counts"] / dataframe["TotalPop"]

dataframe["incident_counts_per_cap"] = dataframe["incident_counts"] / dataframe["TotalPop"] * 1000
dataframe['log_incident_counts_per_cap'] = np.log((1 + dataframe['incident_counts_per_cap']))
#
# print(dataframe["incident_counts_per_cap"])

dataframe = dataframe.drop(dataframe.columns[0], axis = 1)
dataframe = dataframe.dropna()

#features = (dataframe.drop(['CensusId', 'State', 'County', 'incident_counts', 'n_killed', 'n_injured','incident_counts_per_cap', 'log_incident_counts_per_cap'], axis = 1))
#print(features.columns)
#num_cols = len(features.columns)
#features = features.values

features = dataframe[['Black', 'SelfEmployed', 'MeanCommute', 'PercentMen', 'Drive', 'Office', 'Hispanic', 'ChildPoverty', 'Walk', 'Production', 'Poverty', 'White', 'PublicWork', 'PrivateWork', 'WorkAtHome', 'Asian', 'Native', 'Pacific', 'OtherTransp']]
num_cols = len(features.columns)
features = features.values
targets = (dataframe['log_incident_counts_per_cap']).values

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)
X_train1, X_test1, y_train1, y_test1 = train_test_split(features, (dataframe["incident_counts"]).values, test_size=0.2, random_state=0)
X_train2, X_test2, y_train2, y_test2 = train_test_split(features, (dataframe["TotalPop"]).values, test_size=0.2, random_state=0)

#scaler = MinMaxScaler(feature_range=(-1,1))
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential()
model.add(Dense(num_cols, input_dim=num_cols, activation='linear'))
# model.add(Dense(int(num_cols/2), activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
model.add(Dense(int(num_cols/2), activation='linear'))
model.add(Dense(int(num_cols/4), activation='linear'))
model.add(Dense(1, activation='linear'))
#model.add(Dense(1, activation='relu'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=100, batch_size=5)

#predictionsTrain = model.predict(X_train_scaled)
predictions = model.predict(X_test_scaled)
i = 0
newNew = []
for x in predictions:
    new = (np.exp(x) - 1) / 1000 * y_test2[i]
    newNew.append(new)
    #print("REAL: " + str(y_test1[i]) + " PREDICTION: " + str(new))
    #print()
    i += 1

#newValsFlat = [item for sublist in newNew for item in sublist]
rmse = math.sqrt(metrics.mean_squared_error(y_test1, newNew))
print('RMSE for raw incident counts: %.3f' % rmse)
