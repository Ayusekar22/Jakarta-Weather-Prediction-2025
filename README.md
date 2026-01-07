# Weather Prediction Using Two Stage Model (Random Forest Classifier + XGBRegressor) In DKI Jakarta 2025

Indonesia is a country with a tropical climate which has two main seasons, such as the dry season and the rainy season. Rainfall fluctuations, especially during the rainy season, are often difficult to predict. Therefore, this research aims to build a prediction model that can predict the occurrence of rain the next day (tomorrow) based on historical meteorological data obtained from the BMKG data bank.

---

## 📌 Quick Overview 
Predicting tropical rainfall is tricky due to the high frequency of non-rainy days (zero-inflated data). This project implements a **two-stage hybrid architecture** to separately predict the *occurrence* and *intensity* of rain.

**Key Result:** 
- **MAE Reduction:** Improved from 10.68 mm to 6.76 mm (**36% more accurate**).
- **Classification:** 71% Accuracy in predicting "Will it rain tomorrow?".

---


## Table of Contents
1. [Dataset](#dataset)
2. [Flowchart](#flowchart)
3. [Data Understanding](#data-understanding)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Development](#model-development)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [License](#license)
10. [Contact](#contact)

---

## Dataset 
The dataset used in this research is daily historical meteorological data sourced from the official online data portal of the Indonesian Meteorology, Climatology and Geophysics Agency (BMKG) with a location selection in Central Jakarta. 
- Source: dataonline.bmkg.go.id
- Time Range: Data is taken for the period 1 January 2015 to 07 December 2025.
- Data Volume: This dataset consists of 4019 rows of data (observations), where each row represents daily weather data.  

This dataset has 11 original attribute that record various weather parameters with the following description :

| Description | Unit |
| :--- | :--- |
| Waktu pencatatan data | - |
| Temperatur Minimum | °C |
| Temperatur Maksimum | °C |
| Temperatur Rata-rata | °C |
| Kelembapan Rata-rata | % |
| Curah Hujan | mm |
| Lamanya Penyinaran Matahari | Hour |
| Kecepatan Angin Maksimum | m/s |
| Arah Angin saat Kecepatan Maks. | ° |
| Kecepatan Angin Rata-rata | m/s |
| Arah Angin Terbanyak | ° |

[⬆ Back to Top](#table-of-contents)

---

## Flowchart

![Alur Project](Gambar/Flowchart.png)

[⬆ Back to Top](#table-of-contents)

---

## Data Understanding

### Rainfall Distribution Analysis

![Distribusi Hujan](Gambar/Distribusi_Curah.png)
This graphic shows the dominance of the value 0 mm, which indicates that most days do not experience rain. Meanwhile, non-zero rainfall values ​​have a distribution that is very skewed to the right, with a long tail (heavy-tailed) representing moderate to extreme rainfall events.


![Boxplot Hujan](Gambar/Boxplot_RR.png)
The visualization reveals a small number of extreme outliers with values significantly exceeding the rest of the dataset. This confirms that the rainfall data follows a zero-inflated and heavy-tailed distribution.


### Correlation Analysis

![Heatmap](Gambar/Heatmap.png)

Based on the Spearman correlation heatmap on raw meteorological data, it can be seen that the relationship between rainfall and other weather variables on the same day is generally at a weak to moderate level. Rainfall shows a moderate positive correlation with average humidity, as well as a negative correlation with temperature and duration of sunlight, which is in line with the physical characteristics of the rainfall process. However, no very strong correlation was found with any single variable, indicating that the direct (simultaneous) relationship between variables is not sufficient to explain rainfall variability as a whole.

### Baseline Models

The baseline is used as an initial reference.
The classification model achieved an accuracy of 66%, while the regression model produced an MAE of 10,681 mm and an RMSE of 20,818 mm, indicating that there is still room for performance improvement.
RMSE: 20,818 mm

### Key Findings from Data Understanding
- Based on data distribution analysis, daily rainfall shows zero-inflated characteristics, where most observations have a value of zero, as well as a heavy-tailed distribution on days with rain. This indicates that there is a difference in the process between the formation of rainfall events (occurrence) and the mechanisms that determine the amount of rainfall (intensity).

- Furthermore, Spearman correlation analysis on raw data shows that the simultaneous relationship between rainfall and other meteorological variables tends to be weak to moderate. This indicates that the information contained in raw data on the same day is not sufficient to represent rainfall dynamics comprehensively, so temporal-based feature engineering is needed to capture historical patterns and time dependencies.

- Based on the characteristics of the zero-inflated distribution and limited information on raw meteorological features, rainfall modeling requires an approach that is able to handle rainfall events and rainfall intensity separately, while utilizing temporal information. Therefore, a two-stage modeling approach was chosen, by separating predictions of rainfall events (classification) and rainfall magnitudes (regression), so that it is hoped to produce predictions that are more stable, robust against the dominance of zero values, and more representative than the one-stage approach.

[⬆ Back to Top](#table-of-contents)

---

## Data Preprocessing 
- Combine Historical Data into 1 File
- Check Missing Values
- Carrying out Data Mapping 
- Fill in missing values ​​using Random Forest 
- Perform Feature Engineering

[⬆ Back to Top](#table-of-contents)

---

## Feature Engineering
1. Sort Data by Date 

    Data is sorted chronologically based on the Date column to ensure the time series structure is maintained and prevent data leakage in the modeling process.

2. Create Lag Features
   These features were created to help the model learn from previous days conditions. The goal is to capture Temporal Dependencies and the Persistence Effect aknowledging that tomorrow's weather is heavily influenced by today's atmospheric state. The variables used for these lag features include:
 
    * Average Humidity
    * Average Temperature
    * Maximum Temperature
    * Minimum Temperature
    * Duration of sunlight

    With the lag used, namely from 1 to 4 days in advance. 

3. Create a Rolling Mean (Moving Average)

    Rolling mean was carried out for variables that were highly correlated with rainfall using time gaps of 3 and 7 days. After that, the rolling mean is shifted one day back so that it only uses historical information and there is no leakage.

4. Create an Exponentially Weighted Moving Average (EWMA)

    EWMA was created specifically for the Rainfall variable (mm) with periods of 3 and 7 days. This method gives greater weight to the most recent observations so it is more sensitive to changes in rainfall patterns than the usual rolling mean.

5. Rolling Standard Deviation

    Rolling standard deviation is used to measure the level of variation in rainfall within a certain time window (7 days).

6. Cylical Encoding 

    Cyclical encoding is applied to time information to capture seasonal patterns of rainfall. The Date column is reduced to a dayofyear feature, then represented using the sine and cosine functions. This approach is used so that the model understands that time is cyclical.

7. Extreme Rain Event Feature
    
    The extreme rainfall event feature was created to capture unusual rainfall conditions. The extreme rain threshold is determined using the 90th quantile of the rainfall distribution. This feature aims to help the model recognize the influence of previous extreme rain events on rain events the following day.

[⬆ Back to Top](#table-of-contents)

---

## Model Development

### Data Modelling

The modeling process is carried out using the **two-stage hybrid model** approach: 
1. **Classification** to predict rain events (rain/no rain).
2. **Regression** to predict rainfall intensity, which is **only trained using data with rainy days**.

This approach helps reduce bias from the dominance of zero data (days without rain) and allows the regression model to focus on studying rainfall intensity patterns.

### Data Splitting 
The data is divided into 80% train and 20% test in a time-ordered manner (without shuffle) to maintain the temporal structure of the time series and prevent data leakage.

### Classification
The first stage aims to predict rain events as a binary classification problem, with labels 0 for no rain and 1 for rain. The model used is **Random Forest Classifier** with the parameter `class_weight="balanced"` to handle class imbalance. The model is trained using all training data (both rainy and non-rainy days). The model produces a probability of rain (0-1), which is then converted into a class prediction using a threshold of 0.5.

### Regression

The second stage predicts rainfall intensity with a different approach. **The regression model is ONLY trained using data from rainy days** (rain days only), namely samples where actual rainfall is > 0 mm. The filtering process is carried out as follows:

1. Data filtering: From the entire dataset, only rows are taken where `Rain_mm_t+1 > 0` (based on ground truth/actual data)
2. Target transformation: Rainfall is transformed using `log1p` to reduce distribution skewness

The model used is **XGBoost Regressor**. By only training on rainy days, the model can focus on learning rain intensity patterns without being distracted by dominant zero values, resulting in more accurate intensity predictions.

### Hybrid Prediction

In the prediction stage, the two models are combined sequentially:
- If the classifier predicts **no rain** (probability < 0.5) → final rainfall = 0 mm
- If the classifier predicts **rain** (probability ≥ 0.5) → the regressor is called to predict the intensity, and the final result = probability × predicted intensity

[⬆ Back to Top](#table-of-contents)

---

## Results

### Baseline vs Final Model 


| Task           | Baseline Model | Final Model |
|----------------|----------------|-------------|
| Classification | 66% Accuracy   | **71% Accuracy** |
| Regression     | 10.681 mm RMSE | **6.764 mm RMSE** |


### Impact of Temporal Feature Engineering

- Application of time-based feature engineering produces consistent performance improvements compared to a baseline that only uses raw meteorological features. In the classification stage, accuracy increased from 66% to 71%, with an ROC-AUC value of 77%, which shows the model's better ability to distinguish between rainy and non-rainy days.

- At the regression stage, rainfall intensity modeling resulted in a decrease in MAE from 10,681 mm to 6,764 mm and RMSE from 20,818 mm to 16,452 mm. This improvement indicates that temporal features such as lag features, rolling statistics, and EWMA provide relevant historical context, so that the model is able to capture rainfall dynamics that are not visible in static daily features.

### Major factors contributing to rainfall

![Top10Feature](Gambar/Top10FeatureImportances.pngg)

- The results of the feature importance analysis show that rainfall predictions are dominated by features resulting from temporal-based feature engineering, especially those that represent humid atmospheric conditions and historical patterns of rainfall. The feature with the highest contribution is the average humidity of the previous day (lag-1), followed by various representations of historical rainfall such as exponentially weighted moving average (EWMA), rolling mean, and rolling standard deviation.

- The dominance of these features indicates that rain is not determined by instant weather conditions, but rather by the accumulation and persistence of atmospheric conditions in the previous few days. Feature engineering allows the model to capture these temporal dynamics, which cannot be adequately represented by raw same-day meteorological variables.

### Model Limitations and Performance

Rainfall prediction remains challenging due to the complex and localized nature of rainfall events. This project relies on daily surface-level meteorological data, which limits the model’s ability to capture short-lived or highly localized rain events. Further improvements would likely require higher-resolution or additional data sources such as weather radar or satellite imagery.

[⬆ Back to Top](#table-of-contents)

---

## Conclusion 

This project demonstrates that daily rainfall prediction in tropical regions is better approached using a two-stage model rather than a single regression model. By separating rainfall occurrence and intensity, and incorporating temporal feature engineering, the model is able to better capture historical weather patterns and reduce bias from zero-inflated data. While performance is constrained by the use of daily surface-level data, the proposed approach provides a more stable and interpretable framework for short-term rainfall prediction.

[⬆ Back to Top](#table-of-contents)

---

## License

This project is licensed under the MIT License.

![License](Gambar/License.svg)

---

## Contact

If you have any questions or feedback, feel free to reach out:
- **Email:** Ayusekar1822@gmail.com
- **LinkedIn:** https://www.linkedin.com/in/ayusekar22/
- **GitHub:** https://github.com/Ayusekar22

[⬆ Back to Top](#table-of-contents)