# RF temperature modeling
# 
# Read in data
library(tidyverse)
library(lubridate)

temps <- read_csv('1159640.csv')
temps <- mutate(temps, month = lubridate::month(DATE), 
                day = lubridate::day(DATE), week = lubridate::wday(DATE, label = TRUE))
temps$temp_1 <- c(45, temps$TMAX[1:{nrow(temps) - 1}])
temps$temp_2 <- c(45, 44, temps$TMAX[1:{nrow(temps) - 2}])

averages <- read_csv('1159653.csv')
averages <- dplyr::filter(averages, STATION_NAME == 'SEATTLE TACOMA INTERNATIONAL AIRPORT WA US')
averages$month <- as.numeric(substr(averages$DATE, 5, 6))
averages$day <- as.numeric(substr(averages$DATE, 7, 8))

temps <- merge(temps, averages[, c('month', 'day', 'DLY-TMAX-NORMAL')], by = c('month', 'day'), 
               all.x = TRUE)

temps <- temps[, c('month', 'week', 'day', 'DATE', 'TMAX', 'temp_1', 'temp_2', 'DLY-TMAX-NORMAL')]
temps <- dplyr::rename(temps, date = DATE, actual = TMAX)
temps <- dplyr::rename(temps, average = 'DLY-TMAX-NORMAL')
temps$year <- 2016
temps <- temps[, -which(names(temps) == 'date')]

write_csv(temps, 'mod_temps.csv')
