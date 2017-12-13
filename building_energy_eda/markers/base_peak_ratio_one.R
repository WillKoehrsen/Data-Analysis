base_peak_ratio <- function(energydata, energy_col="elec_cons"){
  
  energydata$elec_cons <- energydata[,c(energy_col)]
  
  # start
  energydata$timestamp <- as.POSIXct(energydata$timestamp)
  daypoints <- 24*60/BuildingRClean::time_frequency(energydata)
  # Calculate just business days
  energydata <- energydata[which(energydata$biz_day==1),]
  
  #extracting year month and day of the data
  energydata <- energydata %>%
    mutate(year = as.numeric(substr(timestamp, 1, 4))) %>%
    mutate(month = as.numeric(substr(timestamp, 6, 7))) %>%
    mutate(day = as.numeric(substr(timestamp, 9,10)))
  
  #summer subset
  energydata_summer <- energydata[which(energydata$month>5 & 
                                          energydata$month <8),]
  
  #winter subset
  energydata_winter <- energydata[which(energydata$month <3),]
  
  
  summer_years <- unique(energydata_summer$year)
  base_peak_stat <- c()
  
  # caculations for summer
  # aggregation and average by a day frame
  # calculate the minimum and maximum and ratio of the average day
  # calculation of the ratios
  # decision of 0.7 as the threshold due to pnnl report
  
  
  for (i in summer_years){
    energy_one_summer <- energydata_summer[which(energydata_summer$year==i),]  
    
    if(length(energy_one_summer$elec_cons)>(20*daypoints)){
      aggdata <-aggregate(energy_one_summer$elec_cons, by=list(energy_one_summer$num_time),
                          FUN=mean, na.rm=TRUE)
      
      average_peak <- round(max(aggdata$x), 2)
      average_base <- round(min(aggdata$x), 2)
      base_peak_ratio <- round(average_base/average_peak, 2)
      reduction_from_peak <- round((average_peak-average_base)/average_peak,2)
      base_saving_op <- "No"
      if(reduction_from_peak <=0.7){
        base_saving_op <-"Yes"
      }
      base_peak_gather <- cbind(i, "summer", average_peak,
                                average_base, base_peak_ratio, 
                                reduction_from_peak, base_saving_op)
      
      base_peak_stat <- rbind(base_peak_stat, base_peak_gather)
    }
  }
  
  
  
  winter_years <- unique(energydata_winter$year)
  
  # the same process for winter
  
  for (i in winter_years){
    energy_one_winter <- energydata_winter[which(energydata_winter$year==i),]  
    
    if(length(energy_one_winter$elec_cons)>(20*daypoints)){
      aggdata <-aggregate(energy_one_winter$elec_cons, 
                          by=list(energy_one_winter$num_time),
                          FUN=mean, na.rm=TRUE)
      
      average_peak <- round(max(aggdata$x), 2)
      average_base <- round(min(aggdata$x), 2)
      base_peak_ratio <- round(average_base/average_peak, 2)
      reduction_from_peak <- round((average_peak-average_base)/average_peak,2)
      base_saving_op <- "No"
      if(reduction_from_peak <=0.7){
        base_saving_op <-"Yes"
      }
      
      #i-1 instead of i beacuse winter is considered jan and feb which
      # seats on next year
      base_peak_gather <- cbind(i-1, "winter", average_peak, 
                                average_base, base_peak_ratio, 
                                reduction_from_peak, base_saving_op)
      
      base_peak_stat <- rbind(base_peak_stat, base_peak_gather)
    }
  }
  
  
  # dataframe fixes, colnames and order
  base_peak_stat <- as.data.frame(base_peak_stat)
  colnames(base_peak_stat)[1:2]<- c("year", "season")
  base_peak_stat <- base_peak_stat[with(base_peak_stat, order(year, season)), ]
  return(base_peak_stat)
  
  
  
  
  
}
