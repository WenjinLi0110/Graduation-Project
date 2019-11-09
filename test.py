import csv
import datetime
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from functools import partial
from itertools import repeat

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection, svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

# creates a nested dictionary

def nested_defaultdict(default_factory, depth=1):
    result = partial(defaultdict, default_factory)
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)
    return result()

#initialise dictionary
data_dict = nested_defaultdict(list,5)
data_dict1= nested_defaultdict(list,5)
folder = './Structure' #change this to filepath if needed
max_len=0
max_data={}
max_len1=0
max_data1={}
line_count = 0
for file in os.listdir(folder):
    
    path = folder +'/'+file
    if file.endswith(".csv"):
        with open(path, encoding='utf-8') as f:
            csv_reader = csv.reader(f)

            column_names = None
            first = True
            #ignore first row of each file
            for row in csv_reader:
                line_count +=1
                if first:
                    first = False

                    continue
                
                #ensure ints are ints
                try:
                    int(row[8])
                    int(row[15])
                except ValueError:
                    print("int conversion error",file)
                    skipped+=1
                    continue

                #create row
                item = []
                item.append(row[2]) #append search datetime
                item.append(int(row[15])) #append price

                # calculate departure date and days to departure based on one way
#                 or return journey
                dep_date = None
                days_to_dep = None
                if row[10] == 'oneway':
                    dep_date = row[5]
                    days_to_dep = int(row[8])
                elif row[10] == 'return (inbound)':
                    try:
                        int(row[7])
                    except ValueError:
                        print("int conversion error",file)
                        skipped+=1
                        continue
                    dep_date = row[6]
                    days_to_dep = int(row[7]) + int(row[8])
                elif row[10] == 'return (outbound)':
                    dep_date = row[5]
                    days_to_dep = int(row[8])
                item.append(dep_date) #append departure date
                item.append(days_to_dep) #append days to departure
                item.append(int(row[14])) #append position of search
                data_dict[row[11]][row[12]][row[9]][row[13]][days_to_dep].append(item)
                data_dict1[row[11]][row[12]][row[9]][row[13]][dep_date].append(item)  
                #add row to dict
                if len(data_dict[row[11]][row[12]][row[9]][row[13]])>max_len:
                    max_len=len(data_dict[row[11]][row[12]][row[9]][row[13]])
                    max_data=[data_dict[row[11]][row[12]][row[9]][row[13]],[row[11],row[12],row[9],row[13]]]
                
                if len(data_dict1[row[11]][row[12]][row[9]][row[13]])>max_len1:
                    max_len1=len(data_dict1[row[11]][row[12]][row[9]][row[13]])
                    max_data1=[data_dict1[row[11]][row[12]][row[9]][row[13]],[row[11],row[12],row[9],row[13]]]

        print(file,"processed lines:" ,line_count)

    else: continue

Holiday=["2018-1-1","2018-1-10","2018-1-26","2018-2-12","2018-2-28",
"2018-3-5","2018-3-6","2018-3-12","2018-3-30","2018-3-31","2018-4-1",
"2018-4-2","2018-4-3","2018-4-25","2018-5-4","2018-5-7","2018-5-28",
"2018-6-4","2018-6-11","2018-6-29","2018-7-6","2018-7-13","2018-7-20",
"2018-7-27","2018-8-6","2018-8-15","2018-9-24","2018-9-28","2018-10-1",
"2018-10-5","2018-10-11","2018-10-19","2018-10-25","2018-11-5","2018-11-6",
"2018-11-30","2018-12-24","2018-12-25","2018-12-26","2018-12-31",

"2019-1-1","2019-1-9","2019-1-28","2019-2-11","2019-2-27",
"2019-3-4","2019-3-5","2019-3-11","2019-4-19","2019-4-20","2019-4-21",
"2019-4-22","2019-4-23","2019-4-25","2019-5-3","2019-5-6","2019-5-27",
"2019-6-3","2019-6-10","2019-6-28","2019-7-5","2019-7-12","2019-7-19",
"2019-7-26","2019-8-5","2019-8-14","2019-9-27","2019-9-30","2019-10-4",
"2019-10-7","2019-10-10","2019-10-18","2019-10-24","2019-11-4","2019-11-5",
"2019-11-29","2019-12-24","2019-12-25","2019-12-26","2019-12-31"]
def get_datetime_week(searchDateTime):
    week=datetime.datetime.strptime(searchDateTime,'%Y-%m-%d').weekday()
    return week
def holiday_judge(day):
    if day in Holiday or get_datetime_week(day)>=5:
        return 1
    else:
        return 0



        

max_data[0]=sorted(max_data[0].items(),key=lambda x:x[0],reverse=True)
test=[]
for i in range(len(max_data[0])):
    if(len(max_data[0][i][1])>10):
        for j in max_data[0][i][1]:
            holiday=holiday_judge(j[2])
            if np.random.rand()<0.2:
                test.append([max_data[0][i][0],holiday,j[1]])
                max_data[0][i][1].remove(j)

                break

train=[]

for i in max_data[0]:
    value_sum=[0,0]
    value_len=[0,0]
    for j in i[1]:
        holiday=holiday_judge(j[2])
        value_sum[holiday]=value_sum[holiday]+j[1]
        value_len[holiday]=value_len[holiday]+1
    for holiday in[0,1]:
        if value_len[holiday]>0:
            value_mean=value_sum[holiday]/value_len[holiday]
            train.append([i[0],holiday,value_mean])
train=np.array(train)
test=np.array(test)
features=train[:,0:2]
labels=train[:,2]
sc_model=StandardScaler()
labels_sc=sc_model.fit_transform(labels.reshape(-1,1))
features_test=test[:,0:2]
labels_test=test[:,2]
# model=MLPRegressor(hidden_layer_sizes=(100,50,50),max_iter=100000)
model=GradientBoostingRegressor()
model.fit(features, labels_sc)
Predict=model.predict(features)
Predict=sc_model.inverse_transform(Predict.reshape(-1,1))
Predict_test=model.predict(features_test)
Predict_test=sc_model.inverse_transform(Predict_test.reshape(-1,1))
plt.plot(labels)
plt.plot(Predict)
plt.show()

plt.plot(labels_test)
plt.plot(Predict_test)
plt.show()
print("ok")
