# RF temperature modeling
# 
# Read in data
library(tidyverse)
library(lubridate)

# Read in the data as a dataframe
temps <- read_csv('raw_temps.csv')

# Make sure all readings are from same station
temps <- dplyr::filter(temps, NAME == 'SEATTLE TACOMA INTERNATIONAL AIRPORT, WA US')

# Create month, day, and week columns
temps <- mutate(temps, year = lubridate::year(DATE), 
                month = lubridate::month(DATE), 
                day = lubridate::day(DATE), 
                week = lubridate::wday(DATE, label = TRUE)) %>% 
  arrange(DATE)

# Create the past max temperature columns
temps$temp_1 <- c(NA, temps$TMAX[1:{nrow(temps) - 1}])
temps$temp_2 <- c(NA, NA, temps$TMAX[1:{nrow(temps) - 2}])
# Shift the average wind speed, precipitation, and snow depth
temps$AWND <- c(NA, temps$AWND[1:{nrow(temps) - 1}])
temps$PRCP <- c(NA, temps$PRCP[1:{nrow(temps) - 1}])
temps$SNWD <- c(NA, temps$SNWD[1:{nrow(temps) - 1}])

# Read in the averages as a dataframe
averages <- read_csv('hist_averages.csv')

# Create columns for the month and day
averages$month <- as.numeric(substr(averages$DATE, 5, 6))
averages$day <- as.numeric(substr(averages$DATE, 7, 8))

# Join the averages to the temperature measurements
temps <- merge(temps, averages[, c('month', 'day', 'DLY-TMAX-NORMAL')], 
               by = c('month', 'day'), all.x = TRUE) %>% arrange(DATE)

# Select and order relevant columns
temps <- dplyr::select(temps, year, month, day, week, AWND, PRCP, SNWD,
                       temp_2, temp_1, `DLY-TMAX-NORMAL`, TMAX)

# Rename columns
names(temps) <- c('year', 'month', 'day', 'weekday', 'ws_1', 'prcp_1', 'snwd_1', 
                  'temp_2', 'temp_1', 'average', 'actual')

# Friend predictions
temps$friend <- sapply(temps$average, function(x) 
  round(runif(1, min = x - 20, max = x + 20)))

# Remove first two rows
temps <- temps[-c(1,2), ]

# Remove na
temps <- temps[complete.cases(temps), ]

# Summary of data
summary(temps)

# Write to csv file
write_csv(temps, 'temps_extended.csv')
