# How to handle outliers-the demon points

If you have any confusion in outliers then your search is over. Today, We will talk about outliers in Machine Learning. We will talk about what is outlier, problems with presence of outliers, their cause of occurrence, how to detect them and how to solve the problem of outliers in very simple terms. So let’s start.…

First of all let’s understand what are outliers. So in simple terms, outliers are extreme values that deviate from other observations in data. As we know data have a particular range in which lies all the data points except some. These some points which lie outside the range of data are called outliers.

![image](https://user-images.githubusercontent.com/65160713/131214900-3c86c4f8-8417-45e5-9d05-6bf936420647.png)

_As we can see the red points here are far away from normal data points. These are outliers._

## Problems with outliers in your data —

1. In a data distribution, with extreme outliers, the distribution is skewed in the direction of the outliers which makes it difficult to analyze the data. It means variance and standard deviation of data is affected by outliers.
2. Outliers affect the statistical analysis very much including mean, median and mode. Naive interpretation of statistics derived from data sets that include outliers may be misleading.

![image](https://user-images.githubusercontent.com/65160713/131214955-58642fe3-e8f5-4dee-816d-d60255cbe9a5.png)

_As we can see how removing outliers give a much better fit line for the dataset._

_**Outliers should be rare. If they are not rare then the model or data is not trustworthy. As it is rare so it is not a big problem for large datasets**_

## Types of outliers —

There are three types of outliers which are-

1. **Point outliers**: When single data point lie outside the range of data ,it is called point outlier. These often happen randomly.

![image](https://user-images.githubusercontent.com/65160713/131215081-b5b29205-1a37-466a-9ddb-c2b605c4cc78.png)

_This is graph of deaths in India during COVID-19.We can see a spike at June 16 which has much more value than nearby points. This is point outlier and we can see it just happened on random._

2. **Contextual outliers**: These are also known as conditional outliers as these happens when data point is anomalous in certain condition only. These are noise in data, such as punctuation symbols when realizing text analysis.

![image](https://user-images.githubusercontent.com/65160713/131215115-be14c57c-721d-4006-bc42-d209d23a7636.png)


3. **Collective outliers**: When group of data points fall outside the specific range, then it is called collective outliers.

![image](https://user-images.githubusercontent.com/65160713/131215121-d9a2e1ca-0683-4de3-b5bf-b5afd0dea15c.png)

_We can see that for a long time the subsequence don’t follow the normal pattern. This is collective outliers._

## Causes of occurrence of outliers and their examples:

Some of usual causes for occurence of outliers are:-

1. **Data entry error**- Mistype of a value during making dataset.
2. **Measurement error**- For length data, if measuring instrument is faulty.
3. **Experimental error**- Error in doing experiment. For example, we are conducting an experiment on how much time one student takes to complete exam and one student arrived late during the exam.
4. **Intentional error**- Dummy outliers made to test detection methods.
5. **Data processing errors**- We generally take data from multiple sources during data mining. So it is possible for presence of extraction errors leading to outliers.
6. **Sampling errors**- Taking data from wrong sources.
7. **Natural outlier**- Not an error but just novelties in data. These are outliers which doesn’t occur due to anyone’s mistake but these are naturally present.

## Methods of detecting outliers-

Now we know much about outliers so let’s see how to detect these demon points. We can visually identify outliers by scatter and box plots but let’s see experimental methods to detect outliers.

There are generally four methods which data scientists use to treat detect outliers which are:-

### Z-score method-
It works on principle that if value falls outside of 3 standard deviations from mean, then it is an outlier. We calculate z-score by-

![image](https://user-images.githubusercontent.com/65160713/131215234-c685c5e7-ecdc-4e42-8f7f-9910dff95ad7.png)

The data is standardized first(to a z-score with zero mean and unit variance), so that outlier detection can be performed by using standard z-score cut off values. Then we calculate z-score for each data point after specifying a threshold(common are 2.5,3,3.5) and group outliers and non-outliers.

![image](https://user-images.githubusercontent.com/65160713/131215241-5fadbb36-7d7d-45d4-a03c-e14c8b3ebc60.png)

_Here red points fall below 3 standard deviations from mean and are hence considered as outliers._

But z-score method is not convenient for high-dimensional and large datasets. Hence, we use DBSCAN for it.

### DBSCAN-

DBSCAN stands for Density Based Spatial Clustering of Applications with Noise. It is a density based clustering algorithm. It is focused on finding neighbors by density.

It defines three classes of points-

![image](https://user-images.githubusercontent.com/65160713/131215260-e6928eb4-806c-42b7-8d26-6dd67761533e.png)

1. **Core points**- If its neighborhood(denoted by ε) contains at least some or more number of points than threshold MinPts.
2. **Border points**- If its neighborhood does not contain more points than MinPts but is still density reachable by other points in the cluster.
3. **Noise point or outliers**- If the instance lies in no cluster and id not density reachable to other points.

#### Steps to perform DBSCAN-

1. First we scale the data to classify outliers accurately.
2. Then we choose metric for clustering(Euclidean metric mostly used for 2 or 3 dimension and Manhattan metric for higher dimension.
3. Choose ε(radius of cluster) as it specifies how close points should be to each other to be considered a part of a cluster. It means that if the distance between two points is lower or equal to ε, these points are considered neighbors. It should ne chosen wisely(0.25 to 0.75 commonly used).
4. Choose MinPts wisely as it specifies the minimum number of points to form a dense region. For example, if we set the minPoints parameter as 5, then we need at least 5 points to form a dense region.

We then import DBSCAN from sklearn as-

    from sklearn.cluster import DBSCAN
    
You can check it out here-https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

In it eps parameter refers to ε,min_samples as MinPts and metric parameter for choosing your metric.

DBSCAN is used when distribution of values can’t be assumed and can also be used for higher dimensions. It is good for data which contains clusters of similar density.

### Isolation forests-

This is another technique for handling outliers and is better than others as it doesn’t need scaling like others and is very effective.

It is based on binary decision trees. It’s principle is that outliers are few and far away from rest. Algorithm randomly picks a feature and random split value ranging between maximums and minimum values of the selected feature. To build a forest a tree, ensemble is made averaging all the trees in the forest.

Then for prediction, it compares an observations against that splitting value in a node, that node will have two children on which another random comparisons will be made. The number of splitting made by algorithm is named path length. As expected, outliers will have shorter path lengths than rest observations.

![image](https://user-images.githubusercontent.com/65160713/131215422-3a8931e5-970b-47d3-8afc-2978d80b1362.png)

We can import it from sklearn as-

    from sklearn.ensemble import IsolationForest
 
You can see it’s documentation here-https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

It returns the anomaly score of each sample.

### Interquartile range method-

It is very good and easy for non-Gaussian distribution. It’s principle is that the normal data points would appear in high probability regions of a stochastic model, while outliers would occur in low probability regions.

#### Steps-

1. Calculate Interquartile Range(IQR).
2. Calculate cutoff by multiplying IQR by threshold k(which we choose).Common k value is 1.5. It decides the new Interquartile Range for data points commonly known as decision range.
3. Define lower and upper limit by-

    Lower = q25-cutoff
    Upper = q75+cutoff
    
_where,_

_q25=25th percentile_

_q75=75th percentile_

![image](https://user-images.githubusercontent.com/65160713/131215562-7b5008e5-5797-4daf-aa48-cfae854714e4.png)

If data points falls below lower limit or higher than upper limit, then it is an outlier.

Now we know how to detect outliers, so now let’s learn how to handle outliers.

## Handling outliers-

Most parametric statistics like mean, standard deviation, correlation, etc . and various algorithms like Linear Regression are sensitive to outliers. But still it is not always convenient to drop outliers. We can try the following to handle outliers:-

1. If it is obvious that outlier is due to incorrectly entered or measured data, you should drop the outlier. For example, If you have dataset of woman’s weight and it has 17kg, then it is physically impossible, so you should drop the outlier.
2. If the outlier is not an influential point then you should drop it. If the parameter estimates change a great deal when a point is removed from the calculations, the point is said to be influential.
3. Otherwise you should run the model with outlier included and excluded and test accuracy and drop it if accuracy increases on dropping it, otherwise keep it.

If we don’t drop an outlier, we can try the following to handle outliers-
1. Try a transformation.
2. Try a different model which is not sensitive to outliers.
3. We can treat them separately if they are significantly large in number and make two models and then combine them.

## Conclusion:

So far we have seen, what are outliers, what problems they create, how they are caused, methods to detect them and how to handle them.

Hope you found this article useful. Thanks. Have a good day!
