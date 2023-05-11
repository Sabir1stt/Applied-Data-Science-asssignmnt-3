# sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class baseFunctions():

    def dataReadFunc(file_name):
        try:
            data_df = pd.read_csv(file_name)
        except Exception as e:
            print(e)
            print("these is some kind of error related to path you provided to read the the data.")
            return False
        return data_df
    
    def selectIndicatorFunc(data_df, indicatorName):
        selectIndicationDf = data_df[data_df['Indicator Name'] == indicatorName]
        shape = selectIndicationDf.shape
        if shape[0] == 0:
            print("Empty Dataframe! Please Enter the Valid indicator Name")
            return None
        else:
            return selectIndicationDf
    
    def filterDataFunc(data_df, column, value):
        filteredDf = data_df[data_df[f'{column}'] == value]
        shape = filteredDf.shape
        if shape[0] == 0:
            print("Empty Dataframe! Please Enter the Valid column Name or Enter the valid value")
            return None
        else:
            return filteredDf
    
    def columnsSelectedFunc(data_df, years):
        data_df = data_df[years]
        return data_df
    
    def normalizeDfFunc(data_df):
        scaler = StandardScaler()
        scaler.fit(data_df)
        # standardScaler is a way to preprocess the data and transform/normalize it between 1,-1. 1 means the highest value and -1 meand the lowest value
        scaled_data = scaler.transform(data_df)
        return scaled_data
    
    def findKValueFunc(df, maximum_K):
    
        kmeans_intertia = []
        k_values = []
        
        for k in range(1, maximum_K):
            # choosing KMeans  unsupervised learning models to indentifies the clusters in the data.
            kmeans_model = KMeans(n_clusters = k)
            kmeans_model.fit(df)
            kmeans_intertia.append(kmeans_model.inertia_)
            k_values.append(k)
        return kmeans_intertia, k_values

    def elbowPlotFunc(clusters_centers, k_values):
    
        figure = plt.subplots(figsize = (12, 6))
        plt.plot(k_values, clusters_centers, 'o-', color = 'blue')
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("kmeans Inertia")
        plt.title("Elbow_Plot of kmeans")
        plt.savefig("elbow_plot")
        plt.show()

    def clusteredPlotFunc(data_df, years):
        print(years)
        print(data_df)
        plt.scatter(data_df[[years[0]]], 
            data_df[[years[1]]], 
            c = data_df["pred"])
        plt.title("different clusters for the dataset")
        plt.xlabel(f"{years[0]}")
        plt.ylabel(f"{years[1]}")
        plt.savefig("clustered_image")
        plt.show()