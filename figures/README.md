# Figures

## [Actual vs Predicted](https://gitlab.com/90477_mls_4ds/pr27_data/-/blob/main/figures/Actual_vs_Predicted.png){style="text-decoration: none;"}

The **hybrid model** predictions for Canada, France, India, South Korea and the UK, made by the authors are displayed below. The plot shows in [persian green]{style="color: #00a896;"} the actual data, while the in [paradise pink]{style="color: #ef476f;"} the predictions running from 20th January until 4th April of 2020.

The predictions made by the authors seems to smoothly follow the pattern of the actual values, showing just a short delay.

In order to plot this graph, we downloaded the data from [**Our World in Data**](https://ourworldindata.org/explorers/coronavirus-data-explorer?zoomToSelection=true&time=2020-04-04..2020-04-14&facet=none&pickerSort=asc&pickerMetric=location&Metric=Confirmed+cases&Interval=7-day+rolling+average&Relative+to+Population=true&Color+by+test+positivity=false&country=GBR~CAN~IND~FRA~KOR).

## [Forecast of Covid-19 Cases](https://gitlab.com/90477_mls_4ds/pr27_data/-/blob/main/figures/Forecast_of_COVID-19_cases.png)

Predictions made upon data of Covid-19 cases showing the number of daily new cases according to the *ARIMA-WBF model.*

## [CFR Dataset](https://gitlab.com/90477_mls_4ds/pr27_data/-/blob/main/figures/CFR_Dataset.png)

This table shows the dataset we have used for the analysis. The data was available from the authors' repository. We have displayed it with style to highlight the number of cases and a in-table barplot to easily compare the number of deaths for country.

The dependent variable of this study is the ***CFR***, the second-last column of this table, marked in bold.

## [RT Main](https://gitlab.com/90477_mls_4ds/pr27_data/-/blob/main/figures/RT_main.png)

The regression tree (**RT**) displayed shows the relationship between the important causal variables and *CFR*. The RT starts with the total number of COVID-19 cases as the most crucial causal variable in the parent
node. In each box, the top most numerical values suggest the aver-
age CFR estimates based on the tree.

One of the key findings of the tree is the following rule: When the number of cases of a country is greater than 14,000 having a population between 14 and 75 million are having second highest case fatality rate, approximately of 10%.

## [Alternative RT Main](https://gitlab.com/90477_mls_4ds/pr27_data/-/blob/main/figures/Alternative_RT_main.png)

Similarly, one can see all the rules generated by RT to get additional information about the relationships between control parameters and the response CFR variable with this visualization; it shows the distribution of decision feature in the each node and the mean of the leaf's response in the case of regression tasks.

We have decided to implement this type of visualization for a better understanding of the technique, indeed we can see the way the ***CART*** algorithm splits the data.

## [**Variable Importance RT Main**](https://gitlab.com/90477_mls_4ds/pr27_data/-/blob/main/figures/Variable_Importance_RT_main.png)

The following plot shows how much the single variables affect the model. We can see that the most important variable is the number of cases of each country, followed by the total population, the number of beds in hospitals and the number of days since shutdown.

## [RT Europe](https://gitlab.com/90477_mls_4ds/pr27_data/-/blob/main/figures/RT_Europe.png)

The idea was to perform a regression tree for every continent but unfortunately the observations were few. The continent with most observation is Europe (23) so we performed a RT only for this continent otherwise it would have been meaningless.

In this case the variable which most influences the model is Population/Km2. The most important findings:

I.  when the population density is greater than 255.5/km\^2,

    -   the case fatality rate is approximately 10%
    -   if the cases are less than 16 thousands, the CFR is 11%

II. whereas if the population density is less than 255.5/km\^2,

    -   and the cases are more than 98.7 thousands, the CFR is 11%

    -   otherwise, if the cases are less than 98.7 thousands and the time of arrival is greater than 67 days, the CFR is 5.7%

## [**Variable Importance RT Europe**](https://gitlab.com/90477_mls_4ds/pr27_data/-/blob/main/figures/Variable_Importance_RT_Europe.png)

This plot again shows how much the single variables affect our model. We can see that the most important variable is still the number of cases of each country, followed by the population density, the time of arrival and the number of days since shutdown.