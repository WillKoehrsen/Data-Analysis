# Retrieve appropriate weather data for Progressive Buildings
# Data will be used for training and testing machine learning models (Random Forest)

library(tidyverse)
library(lubridate)

correct_meta <- read_csv('metadata/progressive_corrected.csv')

location_df <- read_csv('metadata/progressive_ksu_starbucks_meta.csv')
date_df <- read_csv('metadata/progressive_dates.csv')
names(correct_meta) <- c('bldg', 'start', 'end', 'lati', 'long', 'tz')

correct_meta$s_name <- c('bq03nu9', 'mx6g86v', 'q3a0qa5', 'vik7gzo', 'hf4x6f3', 'kj8wq87', 'nzpi6ta', 'gcuy3bd')
location_df <- filter(location_df, partner == 'Progressive')
location_df$building <- sapply(location_df$buna, function(x) {strsplit(x, '-|_')[[1]][2]})

reference_df <- merge(location_df, date_df, by = 'building')

for (i in 1:nrow(reference_df)) {
  # Extract the s_name and reference name
  s_name <- reference_df[[i, 's_name']]
  name <- reference_df[[i, 'building']]
  
  start <- reference_df[[i, 'time_start']] - years(2)
  end <- reference_df[[i, 'time_end']] 
  
  latitude <- reference_df[[i, 'lati']]
  longitude <- reference_df[[i, 'long']]
  
  weather_data <- cradlesgis::fetch_solar(latitude, longitude, start, end)
  weather_data <- BuildingRClean::colfix(weather_data, type = 'weather')
  
  write_csv(weather_data, sprintf('metadata/%s_weather.csv', name))
}
