"""
Name:       Viet Dinh
Email:      vdinh2020@gmail.com
Title:      Are There Fecal Bacteria In Your Water?
URL:        https://meromeofbacca.github.io/
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import seaborn as sns
from pyproj import Transformer
from sklearn import metrics
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def import_data(file_name) -> pd.DataFrame:
    """Cleans data"""
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

    df.loc[df['Turbidity'] == '<0.10', 'Turbidity'] = 0
    df.loc[df['Coliforms'] == '<1', 'Coliforms'] = 0
    df.loc[df['Coliforms'] == '>200.5', 'Coliforms'] = 200.5
    df['Turbidity'] = df['Turbidity'].astype(float)
    df['Coliforms'] = df['Coliforms'].astype(float)
    df.loc[df['Coliforms'] >= 1, 'Coliforms'] = 1
    df['Date']= pd.to_datetime(df['Date'])
    return df


def import_map_data(map_file_name):
    """Cleans data and converts NGS coordinates to Lat Long"""
    sites = pd.read_csv(map_file_name)
    transformer = Transformer.from_crs( "epsg:2908","epsg:4326",always_xy=False)
    lat = []
    lon = []
    for index, row in sites.iterrows():
        x_coord = row['X - Coordinate']
        y_coord = row['Y - Coordinate']

        #call transformer
        x_1, y_1 = transformer.transform(x_coord,y_coord)

        #append to lists
        lat.append(x_1)
        lon.append(y_1)
    sites["Lat"] = lat
    sites["Lon"] = lon
    sites = sites.drop(['Sample Station (SS) - Location Description', \
                        'X - Coordinate', 'Y - Coordinate'],
                        axis=1)
    return sites

def merge_map_data(df, map_df):
    """Merges for lat long"""
    df = df.merge(map_df, on='Sample Site', how='left')
    df = df.dropna()
    return df

def create_map(df):
    """Creates folium map for GIS data"""
    gis_map = folium.Map(location=[40.7128, -74.0060])
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
       ).add_to(gis_map)
    gis_map.save(outfile="myMap.html")

def histograms(df, column = 'Residual Free Chlorine'):
    """Density histograms"""
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    no_col = df[df["Coliforms"] == 0]
    yes_col = df[df["Coliforms"] == 1]
    fig, axes = plt.subplots(1,2)
    no_col.hist(column, bins=50, ax=axes[0], density=True)
    yes_col.hist(column, bins=50, ax=axes[1], density=True)

    units = " (NTU)" if column == "Turbidity" else " (mg/L)"
    axes[0].set_xlabel(column + units)
    axes[0].set_ylabel('Density')
    axes[0].set_title(column + ' no coliforms')
    axes[1].set_xlabel(column + units)
    axes[1].set_ylabel('Density')
    axes[1].set_title(column + ' with coliforms')
    fig.tight_layout()
    plt.show()

def scatterplots(df, column = 'Residual Free Chlorine'):
    """Scatterplot visualization with jitter"""
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

def jitter(arr, amt=0.01):
    """Jitter for scatterplot"""
    stdev = amt * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def tpr_graph(precision, recall, threshold):
    """Threshold, precision, recall graph"""
    plt.plot(threshold, precision, label = "Precision")
    plt.plot(threshold, recall, label = "Recall")
    plt.xlabel('Threshold')
    plt.ylabel('Proportions')
    plt.title('TPR Graph')
    plt.legend()
    plt.show()

def pr_graph(precision, recall):
    """Precision vs Recall graph"""
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall Graph')
    plt.show()

def split_data(df, x_cols, y_col_name, test_size=0.25, random_state=2023):
    """Split into test and training data"""
    x_train, x_test, y_train, y_test = train_test_split(df[x_cols],
                                                       df[y_col_name],
                                                       test_size=test_size,
                                                       random_state=random_state)
    return x_train, x_test, y_train, y_test

def fit_model(x_train, y_train):
    """Logistic Regression model"""
    mod = LogisticRegression(solver='saga', penalty='l2', max_iter=5000)
    mod.fit(x_train, y_train)
    return pickle.dumps(mod)

def score_model(mod_pkl,xes,yes):
    """Confusion Matrix using threshold prediction"""
    y_true = yes
    mod = pickle.loads(mod_pkl)
    #y_pred = mod.predict(xes)
    y_pred = threshold_predict(mod, xes)
    confuse_mx = metrics.confusion_matrix(y_true, y_pred)
    return confuse_mx

def threshold_predict(model, xes, threshold = 0.0016226106743523992): #threshold for 100% recall
    """Threshold prediction for scoring model"""
    return np.where(model.predict_proba(xes)[:,1] > threshold, 1.0, 0.0)

def cfmx_visual(cfmx):
    """Sns visualization for confusion matrix"""
    plt.figure(figsize=(9,9))
    sns.heatmap(cfmx, annot=True, fmt="g", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Confusion Matrix'
    plt.title(all_sample_title, size = 15)
    plt.show()

def main():
    """Runs through data cleaning, visualizations and model building"""
    # Data clean up and joins
    file = "Data.csv"
    map_file = "Sites.csv"
    df = import_data(file)
    map_df = import_map_data(map_file)
    df = merge_map_data(df, map_df)

    # Exploratory data visualizations
    histograms(df)
    histograms(df, "Turbidity")
    scatterplots(df)
    scatterplots(df, "Turbidity")

    # Map visualization
    create_map(df)

    # Splitting data for model
    x_cols = ['Turbidity', 'Residual Free Chlorine']
    y_col = 'Coliforms'
    x_train, x_test, y_train, y_test = split_data(df, x_cols, y_col)

    # Fitting model
    mod = fit_model(x_train, y_train)

    # Confusion matrix
    cfmx = score_model(mod, x_test, y_test)

    # Model accuracy plots
    mod = pickle.loads(mod)
    precision, recall, threshold = (
        metrics.precision_recall_curve(y_test, mod.predict_proba(x_test)[:, 1])
    )
    precision = precision[:-1]
    recall = recall[:-1]

    tpr_graph(precision, recall, threshold)
    pr_graph(precision, recall)
    cfmx_visual(cfmx)

if __name__ == "__main__":
    main()
