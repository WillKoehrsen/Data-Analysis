#' Plots the daily electricity consumption patterns for each day of the week
#' grouped by each season and year.
#'
#' This graph provides a comprehensive overview of the daily and weekly
#' patterns within the data. It can also be used to make comparisons
#' across seasons.
#'
#' @param energydata Building energy consumption dataframe. Each row of the
#' dataframe should be an observation and the columns contain the energy and weather
#' variables.
#'
#' @param unit unit of measurement, default is "kWh"
#'
#' @param col which column to go through variablity calculations
#'
#' @param interval the time interval to calculate the values
#' default is 60.
#'
#' @param type the type of output plot. It can be "all_in_one", "all", or "last"
#' "all_in_one" plots all the boxplots in one graph
#' "all" plots winters and summmers in two separate graph
#' "last" plots the last full summer and winter
#'
#'
#' @param plot plots the variability function if set to TRUE. Default is TRUE.
#'
#' @param func function of the result calculation. The default is IQR. It could
#' be any statistical function such as mean, median, sd, etc
#'
#' @return  Plots a graph displaying daily patterns for each day of the week, for each year and season.
#' Returns a dataframe consisting of the desired statistical parameter, for
#' each day of each season of the year.
#'
#'
#' @export




energy_pattern_snapshot <- function(energydata , unit = "kWh", col = "elec_cons",interval = 60,
                         type = "all_in_one", plot = TRUE, func = IQR){

  require(dplyr)
  require(tidyverse)

  energydata <- as.data.frame(energydata)
  energydata$timestamp <- as.POSIXct(energydata$timestamp)
  energydata$elec_cons <- energydata[,col]

  # respline on desired time interval
  new_time <- as.data.frame(seq(from = energydata$timestamp[1],
                                to = energydata$timestamp[length(energydata$timestamp)],
                                by = paste(interval, "min")))

  names(new_time)[1] <- "timestamp"

  energydata <- BuildingRClean::merge_ts(new_time, energydata)



  energydata$day_of_week <- lubridate::wday(energydata$timestamp,
                                            label = TRUE,
                                            abbr = TRUE)
  energydata$day_of_week <- substr(energydata$day_of_week, 1, 3)

  # number of datapoints in a day
  daypoints <- 24*60/BuildingRClean::time_frequency(energydata)
  ## Calculate just business days. ( This is not used in this code)

  # energydata <- energydata[which(energydata$biz_day==1),]

  # extracting year month and day of the data
  energydata <- energydata %>%
    mutate(year = as.numeric(substr(timestamp, 1, 4))) %>%
    mutate(month = as.numeric(substr(timestamp, 6, 7))) %>%
    mutate(day = as.numeric(substr(timestamp, 9, 10)))
  # calculate the numerical results
  variability_data <- hourly_variability(energydata = energydata, daypoints = daypoints, interval = interval, func = func)


  if(plot == TRUE){

  # summer subset

  energydata_summer <- energydata[which(energydata$month > 5 &
                                          energydata$month < 9),]

  # winter subset
  energydata_winter <- energydata[which(energydata$month < 3 |
                                          energydata$month > 11),]
  #assigning the winter of december with year of january and feburary
  energydata_winter[which(energydata_winter$month == 12),]$year <- energydata_winter[which(energydata_winter$month == 12),]$year+1

  #
  summer_years <- unique(energydata_summer$year)
  winter_years <- unique(energydata_winter$year)




  days <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
  if (type == "last"){
  #taking last full summer
  for (i in length(summer_years):1){
    energy_one_summer <- energydata_summer[which(energydata_summer$year == summer_years[i]),]

  if(length(energy_one_summer$elec_cons) > (70*daypoints)){
    break

  }
  }
#taking last full winter
  for (i in length(winter_years):1){
    energy_one_winter <- energydata_winter[which(energydata_winter$year == winter_years[i]),]

    if(length(energy_one_winter$elec_cons) > (70*daypoints)){
      break

    }
  }

  energy_one_summer$season <- "Summer"
  energy_one_winter$season <- "Winter"
  energy_one_summer_winter <- rbind(energy_one_summer, energy_one_winter)

#geting non holiday weekdays and weekends all togethor
  energy_one_summer_winter <- get_biz_wknd(energy_one_summer_winter)
  t_srt <- min(energy_one_summer_winter$num_time)



p <- ggplot(energy_one_summer_winter, aes(x=as.factor(num_time),
                                          y=elec_cons))+
  geom_boxplot(fill ='red', outlier.size = NA)+
  facet_grid(as.factor(season)~factor(day_of_week, levels = days),
             margins =FALSE)+
  stat_summary(fun.y = mean, geom = "line", aes(group = 1), col = 'blue')+
  scale_x_discrete(name = "Time (h)",
                   breaks = c(t_srt, t_srt+6, t_srt + 12, t_srt +18),
                   labels =c(0,6,12,18))+
  ylab(paste0("Consumption (", unit, ")"))+
  theme(axis.title = element_text(size = 16,face= 'bold')) +
  theme(plot.title = element_text(size = 16,face= 'bold')) +
  theme(strip.text = element_text(size = 12,   face = "bold")) +
  theme(legend.text=element_text(size=12))+
  theme(axis.text  = element_text(size = 12, face = 'bold'))


print(p)
}else if(type == "all"){
  # calculates all of the summer winters

  #taking all full summers
  for(i in summer_years){
    summer_length <- length(energydata_summer[which(energydata_summer$year == i),]$year)
    if(summer_length < 70*daypoints){
      energydata_summer <- energydata_summer[which(energydata_summer$year != i),]

    }

  }

  energydata_summer <- get_biz_wknd(energydata_summer)
  #taking all full winters

  for(i in winter_years){
    winter_length <- length(energydata_winter[which(energydata_winter$year == i),]$year)
    if(summer_length < 70*daypoints){
      energydata_winter <- energydata_winter[which(energydata_winter$year != i),]

    }

  }

  energydata_winter <- get_biz_wknd(energydata_winter)
  energydata_winter$year <- energydata_winter$year - 1
  t_srt <- min(energydata_winter$num_time)

  p <- ggplot(energydata_summer, aes(x=as.factor(num_time), y=elec_cons))+
    geom_boxplot(fill ='red', outlier.size = NA)+
    facet_grid(as.factor(year)~factor(day_of_week, levels = days), margins =FALSE)+
    stat_summary(fun.y = mean, geom = "line", aes(group = 1), col = 'blue')+
    scale_x_discrete(name = "Time (h)",
                     breaks = c(t_srt, t_srt+6, t_srt + 12, t_srt +18),
                     labels =c(0,6,12,18))+
    ylab(paste0("Consumption (", unit, ")"))+
    labs(title = "Summer")+
    theme(axis.title = element_text(size = 16,face= 'bold')) +
    theme(plot.title = element_text(size = 16,face= 'bold')) +
    theme(strip.text = element_text(size = 12,   face = "bold")) +
    theme(legend.text=element_text(size=12))+
    theme(axis.text  = element_text(size = 12, face = 'bold'))
  print(p)

  q <- ggplot(energydata_winter, aes(x=as.factor(num_time), y=elec_cons))+
    geom_boxplot(fill ='red', outlier.size = NA)+
    facet_grid(as.factor(year)~factor(day_of_week, levels = days), margins =FALSE)+
    stat_summary(fun.y = mean, geom = "line", aes(group = 1), col = 'blue')+
    scale_x_discrete(name = "Time (h)",
                     breaks = c(t_srt, t_srt+6, t_srt + 12, t_srt +18),
                     labels =c(0,6,12,18))+
    ylab(paste0("Consumption (", unit, ")"))+
    labs(title = "Winter")+
    theme(axis.title = element_text(size = 16,face= 'bold')) +
    theme(plot.title = element_text(size = 16,face= 'bold')) +
    theme(strip.text = element_text(size = 12,   face = "bold")) +
    theme(legend.text=element_text(size=12))+
    theme(axis.text  = element_text(size = 12, face = 'bold'))
  print(q)


}else if(type == "all_in_one"){



  # calculates all of the summer winters

  #taking all full summers
  for(i in summer_years){
    summer_length <- length(energydata_summer[which(energydata_summer$year == i),]$year)
    if(summer_length < 70*daypoints){
      energydata_summer <- energydata_summer[which(energydata_summer$year != i),]

    }

  }

  energydata_summer <- get_biz_wknd(energydata_summer)
  energydata_summer$season <- "summer"
  #taking all full winters

  for(i in winter_years){
    winter_length <- length(energydata_winter[which(energydata_winter$year == i),]$year)
    if(summer_length < 70*daypoints){
      energydata_winter <- energydata_winter[which(energydata_winter$year != i),]

    }

  }

  energydata_winter <- get_biz_wknd(energydata_winter)
  energydata_winter$season <- "winter"
  t_srt <- min(energydata_winter$num_time)
  energydata_winter$year <- energydata_winter$year-1

  energy_data <- rbind(energydata_summer, energydata_winter)

  energy_data$year_season <- paste0(energy_data$year, "_", energy_data$season)
  energy_data$year_season <- substr(energy_data$year_season, 3, 11)

  aggdata <-aggregate(energy_data$elec_cons,
                      by=list(energy_data$num_time,
                              factor(energy_data$day_of_week, levels =days),
                              energy_data$year_season),
                      FUN=mean, na.rm=TRUE)

  q <- ggplot(energy_data, aes(x=as.factor(num_time), y=elec_cons))+
    geom_boxplot(fill ='red', outlier.size = NA)+
    facet_grid(as.factor(year_season)~factor(day_of_week, levels = days),
               margins =FALSE)+
   stat_summary(fun.y = mean, geom = "line", aes(group = 1), col = 'blue')+
    scale_x_discrete(name = "Time (h)",
                     breaks = c(t_srt, t_srt+6, t_srt + 12, t_srt +18),
                     labels =c(0,6,12,18))+
    ylab(paste0("Consumption (", unit, ")"))+
    labs(title = "Variability plot")+
    theme(axis.title = element_text(size = 16,face= 'bold')) +
    theme(plot.title = element_text(size = 16,face= 'bold')) +
    theme(strip.text = element_text(size = 12,   face = "bold")) +
    theme(legend.text=element_text(size=12))+
    theme(axis.text  = element_text(size = 12, face = 'bold'))
  print(q)



}
}
return(variability_data)

  }
