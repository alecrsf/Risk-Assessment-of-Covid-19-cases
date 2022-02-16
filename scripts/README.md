# Real-time forecasts and risk assessment of novel coronavirus (COVID-19) cases: A data-driven analysis

The main focus of this paper is two-fold: (a) generating short term (real-time) forecasts of the future COVID-19 cases for multiple countries; (b) risk assessment (in terms of case fatality rate) of the novel COVID-19 for some profoundly affected countries by finding various important demographic characteristics of the countries along with some disease characteristics.

## Real-time forecasting of COVID-19 cases

To solve the first problem, we presented a hybrid approach based on autoregressive integrated moving average model and Wavelet-based forecasting model that can generate short-term (ten days ahead) forecasts of the number of daily confirmed cases for Canada, France, India, South Korea, and the UK. The predictions of the future outbreak for different countries will be useful for the effective allocation of health care resources and will act as an early-warning system for government policymakers.

For the COVID-19 datasets, we propose a hybridization of stationary ARIMA and nonstationary WBF model to reduce the individual biases of the component models. The $ARIMA(p,d,q)$ model can be mathematically expressed as follows: $$y_t = \theta_0 + \phi_1y_{t-1} + \phi_2y_{t-2} +...+\phi_py_{t-p}+\epsilon_t - \theta_1\epsilon_{t-1}-\theta_2\epsilon_{t-2}-...-\theta_q\epsilon_{t-q}$$

where:

-   $y_t$ denotes the actual value of the variable under consideration at time $t$.
-   $ε_t$ is the random error at time $t$.
-   $φ_i$ and $\theta_i$ are the coefficients of the $ARIMA$ model.
-   $p$ and $q$ are the order of the $AR$ model and the $MA$ model respectively, and $d$ is the level of differencing.

The ARIMA model fails to produce random errors or even stationary residual series. Thus, we choose the wavelet function to model the remaining series. Firstly, an ARIMA model is built to model the linear components of the epidemic time series, and a set of out-of-sample forecasts are generated. In the second phase, the ARIMA residuals (oscillatory residual series) are remodeled using a mathematically-grounded WBF model. Here, WBF models the left-over autocorrelations in the residuals which ARIMA could not model. The algorithmic presentation of the proposed hybrid model is given in [Algorithm 1](#Algorithm1).

![](Algorithm1.png "Algorithm 1"){#Algorithm1}

The proposed model can be looked upon as an error remodeling approach in which we use ARIMA as the base model and remodel its error series by wavelet-based time series forecasting technique to generate more accurate forecasts.

As the WBF model is fitted on the residual time series, predictions are generated for the next ten time steps (5 April 2020 to 14 April 2020). Further, both the ARIMA forecasts and WBF residual forecasts are added together to get the final out-of-sample forecasts for the next ten days (5 April 2020 to 14 April 2020)

## **Risk assessment of COVID-19 cases**

In the second problem, we applied an optimal regression tree algorithm to find essential causal variables that significantly affect the mortality for different countries.

Mortality is crudely estimated using the **case fatality rate** (**CFR**), which divides the number of known deaths by the total number of identified cases.

A key differentiation among the CFR of different countries can be found by determining an exhaustive list of causal variables that significantly affect CFR. In this work, we put an effort to identify critical parameters that may help to assess the risk (in terms of CFR) using an optimal regression tree model

The regression tree has a built-in variable selection mechanism from high dimensional variable space and can model arbitrary decision boundaries.

The regression tree combines case estimates, epidemiological characteristics of the disease, and health-care facilities to assess the risks of major outbreaks for profoundly affected countries.

Such assessments will help to anticipate the expected morbidity and mortality due to COVID-19 and provide some critical information for the planning of health care systems in various countries facing this epidemic.

# Conclusions

This paper, of which we have remade the analysis, was cited by the authors of the paper **Applications of machine learning and artificial intelligence for Covid-19 (SARS-CoV-2) pandemic: A review**[^1], within the section *2.3* *"ML and AI technology in SARS-CoV-2 prediction and forecasting"*. The authors suggested that this paper, among several others has provided evidence that AI and ML can significantly improve
treatment, forecasting and contact tracing for the Covid-19 pandemic and reduce the human intervention in medical practice.

[^1]: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7315944/>

After having re-proposed the model implemented in the reference *47*[^2]*,* we agreed to what they claim, specially because the model was quiet reliable. Also, that most of the models are not deployed enough to show their real-world operation, but they are still up to the mark to tackle the pandemic.

[^2]: [`Real-time forecasts and risk assessment of novel coronavirus (COVID-19) cases: A data-driven analysis`](https://www.sciencedirect.com/science/article/pii/S0960077920302502)
