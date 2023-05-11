# sklearn, numpy imports

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from base import baseFunctions
def exponentialGrowth(x, a, b, c):
    # np.exp is used to get the exponential value for the array
    # np.random.normal used to get the Normal Distribution. It is one of the most important distributions that fits the probability distribution of many events, eg. IQ Scores, Heartbeat etc.
    return a * np.exp(b * x) + np.random.normal(0, 0.2, x.shape[0]) + c

def errRanges(popt, cov_matrix, x):
    errors = np.sqrt(np.diag(cov_matrix))
    lowerBound = exponentialGrowth(x, *(popt - errors))
    upperBound = exponentialGrowth(x, *(popt + errors))
    return lowerBound, upperBound

if __name__ == "__main__":

    file_name = "climate_Change_Data.csv"
    data_df   = baseFunctions.dataReadFunc(file_name)
    print(data_df)
    # can change the indicator name here!
    indicatorName       = "Urban population (% of total population)"
    indicatorSelectedDf = baseFunctions.selectIndicatorFunc(data_df = data_df, indicatorName = indicatorName)

    years = ['1990','2015']
    selectedYears = baseFunctions.columnsSelectedFunc(indicatorSelectedDf, years)


    climateChangeDF = selectedYears.dropna()  # entries with one nan are useless

    plt.scatter(climateChangeDF[years[0]], 
                climateChangeDF[years[1]])
    plt.xlabel(f"{years[0]}")
    plt.ylabel(f"{years[1]}")
    plt.title("data visualisation")
    plt.savefig("indicator data visualisation")
    plt.show()
    print("dmeo")

    scaleData = baseFunctions.normalizeDfFunc(climateChangeDF)

    # will show the graph to use to get the best kValue
    baseFunctions.elbowPlotFunc(*(baseFunctions.findKValueFunc(scaleData, 12)))
    print("||||||||||||||||||||||||||||||||||||||||||||||||||")
    KValue = int(input("Enter the perfect K value score ! "))
    print(KValue)

    """ Number of clusters (K): The number of clusters you want to group your data points into, 
    has to be predefined. Initial Values/ Seeds: The choice of the initial cluster centers can have an impact on the final cluster formation. 
    The K-means algorithm is non-deterministic """
    kmeans_model = KMeans(n_clusters = KValue)

    kmeans_model.fit(scaleData)
    print(kmeans_model.labels_)


    climateChangeDF["pred"] = kmeans_model.labels_

    print(climateChangeDF)

    baseFunctions.clusteredPlotFunc(climateChangeDF, years)


    indicatorName = "Urban population (% of total population)"
    indicatorSelectedDf = baseFunctions.selectIndicatorFunc(data_df = data_df, indicatorName = indicatorName)
    indicatorSelectedDf = indicatorSelectedDf.dropna()  # entries with one nan are useless


    columnName      = "Country Name"
    value           = 'Arab World'
    selectedYears   = baseFunctions.filterDataFunc(indicatorSelectedDf, columnName, value)
    x = selectedYears.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
    x = x.values.reshape((x.shape[1]))
    y = exponentialGrowth(x, 2, 0.5, 0)

    popt, pcov = curve_fit(exponentialGrowth, x, y)

    x_predictions = np.linspace(x.min(), x.max() + 10, 20)


    lowerBound, upperBound = errRanges(popt, pcov, x_predictions)


    # Plot the data, best fit, and confidence range
    plt.scatter(x, y, label='Data')
    plt.plot(x_predictions, exponentialGrowth(x_predictions, *popt), label='Best Fit')
    plt.fill_between(x_predictions, lowerBound, upperBound, alpha=1, label='Confidence Range')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("best curveFit line and future Predictions for 20 of years")
    plt.legend()
    plt.savefig("curve_fit")
    plt.show()
    
