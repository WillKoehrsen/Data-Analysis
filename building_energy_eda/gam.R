# Prophet time series work
# Exploring general additive model decomposition

library(tidyverse)
library(prophet)

nve <- read_csv('data/f-NVE_weather.csv')
ts <- dplyr::select(nve, timestamp, forecast)
names(ts) <- c('ds', 'y')
ts_model <- prophet(ts)
future <- prophet::make_future_dataframe(ts_model, periods = 365)
head(future)

forecast <- predict(ts_model, future)

ggplot(forecast, aes(ds, yhat)) + geom_line() + xlab('Date') + 
  ylab('Forecast (kWh)') + ggtitle('Prophet Forecasting')

plot(ts_model, forecast)

prophet::prophet_plot_components(ts_model, forecast)

library(prophet)

ts <- data.frame(y = energy_data$forecast,
                 ds = energy_data$timestamp)

ts_mode <- prophet::prophet(ts)

future <- make_future_dataframe(ts_mode, periods = 180)

forecast <- predict(ts_mode, future)

plot(ts_mode, forecast)

prophet_plot_components(ts_mode, forecast)

