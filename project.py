# -*- coding: utf-8 -*-
'''
Created on Sun Apr  2 20:47:57 2023

@author: merom
'''


import pickle
import pandas as pd
import numpy as np
import folium
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pyproj import Transformer 

#Cleans data
#Removes columns
#Converts values
def import_data(file_name) -> pd.DataFrame:    
    df = pd.read_csv(file_name)
    
    df = df[ ['Sample Number', 'Sample Date', 'Sample Site', \
              'Residual Free Chlorine (mg/L)', 'Turbidity (NTU)',\
              'Coliform (Quanti-Tray) (MPN /100mL)']]
        
    df = df.rename(columns={'Sample Number' : 'ID', \
                            'Sample Date' : 'Date', \
                            'Residual Free Chlorine (mg/L)' : 'Residual Free Chlorine', \
                            'Turbidity (NTU)' : 'Turbidity', \
                            'Coliform (Quanti-Tray) (MPN /100mL)' : 'Coliforms'
                            })
    df = df.dropna()
    
    #Clean data
    df.loc[df['Turbidity'] == '<0.10', 'Turbidity'] = 0
    df.loc[df['Coliforms'] == '<1', 'Coliforms'] = 0
    df.loc[df['Coliforms'] == '>200.5', 'Coliforms'] = 200.5
    df['Turbidity'] = df['Turbidity'].astype(float)
    df['Coliforms'] = df['Coliforms'].astype(float)
    df.loc[df['Coliforms'] >= 1, 'Coliforms'] = 1
    df['Date']= pd.to_datetime(df['Date']) 
    return df

#cleans data
#converts NGS coordinates to Lat Long
def import_map_data(map_file_name):
    sites = pd.read_csv(map_file_name)
    transformer = Transformer.from_crs( "epsg:2908","epsg:4326",always_xy=False)
    lat = []
    lon = []
    for index, row in sites.iterrows():
        x = (row['X - Coordinate'])
        y = (row['Y - Coordinate'])
        
        #call transformer
        x1, y1 = transformer.transform(x,y)
          
        #append to lists    
        lat.append(x1)
        lon.append(y1)
    sites["Lat"] = lat
    sites["Lon"] = lon
    sites = sites.drop(['Sample Station (SS) - Location Description', \
                        'X - Coordinate', 'Y - Coordinate'], 
                        axis=1)
    return sites
    
#Merges for lat long
def merge_map_data(df, map_df):  
    df = df.merge(map_df, on='Sample Site', how='left')
    df = df.dropna()
    return df

def create_map(df):
    m = folium.Map(location=[40.7128, -74.0060])
    color_map = ["blue", "darkgreen"]
    icon_map = ["glass-water", "toilet"]
    safety = ["Safe", "Unsafe"]
    new_df = df.groupby(["Sample Site","Lat", "Lon"])["Coliforms"].max().reset_index()
    for index, row in new_df.iterrows():
        col= int(row["Coliforms"])
        popup_str = "Sample Site: " + row["Sample Site"] + "<br>"\
                    "Max coliform level: " + safety[col]
        folium.Marker(\
          location = [row["Lat"], row["Lon"]],
          popup = folium.Popup(popup_str, max_width = 500),
          icon = folium.Icon(color=color_map[col], icon=icon_map[col], prefix='fa'),
       ).add_to(m)
    m.save(outfile="myMap.html")
    
def histograms(df, column = 'Residual Free Chlorine'):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    no_col = df[df["Coliforms"] == 0]
    yes_col = df[df["Coliforms"] == 1]
    fig, axes = plt.subplots(1,2)
    no_col.hist(column, bins=50, ax=axes[0], density=True)
    yes_col.hist(column, bins=50, ax=axes[1], density=True)
    
    units = " (NTU)" if column == "Turbidity" else " (mg/L)"
    axes[0].set_xlabel(column + units)
    axes[0].set_ylabel('Probability density')
    axes[0].set_title(column + ' no coliforms')
    axes[1].set_xlabel(column + units)
    axes[1].set_ylabel('Probability density')
    axes[1].set_title(column + ' with coliforms')
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    
def jitter(arr, amt=0.01):
    stdev = amt * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def scatterplots(df, column = 'Residual Free Chlorine'):
    mean = np.mean(df['Coliforms'])
    plt.scatter(x=df[column], 
                 y = jitter(df['Coliforms']),
                 s=10,
                 alpha=0.5,
                 edgecolors=('none'),
                 )

    plt.title("Amount of " + column + " with and without Coliforms")
    plt.xlabel(column)
    plt.ylabel("Coliforms")
    plt.axhline(y = mean, color = 'r', linestyle = '-')
    plt.show()
  
def tpr_graph(mod_pkl, x, y):
    mod = pickle.loads(mod_pkl)
    precision, recall, threshold = (
        metrics.precision_recall_curve(y, mod.predict_proba(x)[:, 1]))
    tpr_df = pd.DataFrame({"threshold":threshold, 
                       "precision":precision[:-1], "recall": recall[:-1], })
    #print(tpr_df)
    precision = precision[:-1]
    recall = recall[:-1]
    plt.plot(threshold, precision, label = "Precision")
    plt.plot(threshold, recall, label = "Recall")
    plt.xlabel('Threshold')
    plt.ylabel('Proportions')
    plt.title('TPR Graph')
    plt.legend()
    plt.show()

def pr_graph(mod_pkl, x, y):
    mod = pickle.loads(mod_pkl)
    precision, recall, threshold = (
        metrics.precision_recall_curve(y, mod.predict_proba(x)[:, 1]))
    precision = precision[:-1]
    recall = recall[:-1]
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall Graph')
    plt.show()
    
def split_data(df, x_cols, y_col_name, test_size=0.25, random_state=2023):
    x_train, x_test, y_train, y_test = train_test_split(df[x_cols],
                                                       df[y_col_name],
                                                       test_size=test_size,
                                                       random_state=random_state) 
    return x_train, x_test, y_train, y_test

def fit_model(x_train, y_train):
    mod = LogisticRegression(solver='saga', penalty='l2', max_iter=5000)
    mod.fit(x_train, y_train)
    return pickle.dumps(mod)
 
def score_model(mod_pkl,xes,yes):
    y_true = yes
    mod = pickle.loads(mod_pkl)
    #y_pred = mod.predict(xes)
    y_pred = threshold_predict(mod, xes)
    confuse_mx = metrics.confusion_matrix(y_true, y_pred)
    
    return confuse_mx    

def cfmx_visual(cm):
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt="g", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Confusion Matrix'
    plt.title(all_sample_title, size = 15)
    plt.show()
    
def build_model(df):
    x_cols = ['Turbidity', 'Residual Free Chlorine']
    y_col = 'Coliforms'
    x_train, x_test, y_train, y_test = split_data(df, x_cols, y_col)
    mod = fit_model(x_train, y_train)
    cfmx = score_model(mod, x_test, y_test)
    tpr_graph(mod, x_test, y_test)
    pr_graph(mod, x_test, y_test)
    return mod, cfmx, x_test

def threshold_predict(model, X, threshold = 0.0016226106743523992): #0.0016226106743523992 0.0021142541893057845
    return np.where(model.predict_proba(X)[:,1] > threshold, 1.0, 0.0)

def main():
    file = "Data.csv"
    map_file = "Sites.csv"
    df = import_data(file)
    map_df = import_map_data(map_file)
    df = merge_map_data(df, map_df)
    
    histograms(df)
    scatterplots(df)

    mod_pkl, cfmx, x_test = build_model(df)
    mod = pickle.loads(mod_pkl)
    
    cfmx_visual(cfmx)
    create_map(df)
    #[intercept] = mod.intercept_
    #[[coef1,coef2]] = mod.coef_
    #print(f'Intercept:           {intercept:.1f}')
    #print(f'Turbidity coefficient: {coef1:.1f}')
    #print(f'Residual Free Chlorine coefficient: {coef2:.1f}')
    #print(cfmx)
    
if __name__ == "__main__":
    main()












