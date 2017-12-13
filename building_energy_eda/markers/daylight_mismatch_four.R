

setwd("/mnt/projects/CSE_MSE_RXF131/hpc-members/axk846/jci-new-processed-files/")
energy_data <-data.table::fread("p-jci-5145-30131.csv")

energy_data <-cisco_reader(cisco_list[1])



daylight_mismatch <- function(energy_data){

energy_data$timestamp <- as.POSIXct(energy_data$timestamp)
actual_data <- energy_data[which(!energy_data$pow_dem == 0), ]

time_freq <- BuildingRClean::time_frequency(actual_data[1:1000, ])

freq <- 24 * 60 / time_freq


# find the daylight times
actual_data$daylight <- lubridate::dst(actual_data$timestamp)

suppressWarnings(actual_data[which(actual_data$daylight==TRUE),]$daylight <- 1)

D <- actual_data[which(actual_data$daylight == 0), ]

actual_data$daylightdif <- NA
suppressWarnings(actual_data[2:length(actual_data$daylight), ]$daylightdif <- 
                   diff(actual_data$daylight))

daylight_date <- actual_data[which(!actual_data$daylightdif == 0), ]

raw <-actual_data
raw$elec_cons<-forecast::na.interp(raw$elec_cons)

##column names of result dataframe
col.names<- c("ACF",
              "lag",
              "time_diff_minutes",
              "daylight_date",
              "daylight_mismatch")

daylight_results <- read.table(text = "",
                               col.names = col.names)

## for each daylight saving time analyse to see if we have a mismatch or not

for (i in 1:length(daylight_date$elec_cons)){

raw1 <- raw[which(raw$timestamp < daylight_date$timestamp[i]), ]
raw1 <- raw1[(length(raw1$elec_cons) - 30 * freq):length(raw1$elec_cons), ]
raw2 <- raw[which(raw$timestamp > daylight_date$timestamp[i]), ]
raw2 <- raw2[1:(30*freq), ]



days <- c("Mon", 
          "Tue", 
          "Wed", 
          "Thu", 
          "Fri", 
          "Sat",
          "Sun")


grouped1 <- dplyr::group_by(raw1, 
                     day_of_week,
                     num_time)

raw1_mean_week <- dplyr::summarise(grouped1, 
                            mean=mean(elec_cons), 
                            mean_power_dem=mean(pow_dem))

raw1_mean_week <-raw1_mean_week[order(match(raw1_mean_week$day_of_week,days)), ]

grouped2 <- dplyr::group_by(raw2, 
                     day_of_week,
                     num_time)

raw2_mean_week <- dplyr::summarise(grouped2, 
                            mean=mean(elec_cons), 
                            mean_power_dem=mean(pow_dem))

raw2_mean_week <-raw2_mean_week[order(match(raw2_mean_week$day_of_week,
                                            days)),]

if((length(raw2_mean_week$num_time) > 100) &
   (length(raw1_mean_week$num_time) > 100)){
  
merged_data <- as.data.frame(merge(raw1_mean_week, 
                                   raw2_mean_week, 
                     by =c("day_of_week", 
                           "num_time"), 
                     all=TRUE))

merged_data <-merged_data[order(match(merged_data$day_of_week,
                                      days)), ]


merged_data$mean_power_dem.x <- forecast::na.interp(merged_data$mean_power_dem.x)
merged_data$mean_power_dem.y <- forecast::na.interp(merged_data$mean_power_dem.y)

## plot mean power for a month before and after daylight saving

plot(merged_data$mean.x, 
     type = "l", 
     xaxt  = 'n',
     xlab="", 
     ylab = "Consumption (kWh)")

lines(merged_data$mean.y,
      type = "l",
      col ='red')


axis(1, 
     at=seq(0, 7*freq, by=freq), 
     labels = FALSE)
text(seq(freq/2, 6.5*freq, by=freq), 
     par("usr")[3] - 0.2, 
     labels = days,  
     pos = 1, 
     xpd = TRUE)

legend(
  "topright",
  cex =0.65,
  lty=c(1,1),
  col=c("black", "red"),
  legend = c("Before Daylight Saving Date", 
             "After Daylight Saving Date")
)

## plot power demand before and after daylight saving
plot(merged_data$mean_power_dem.x,
     type = "l", 
     xaxt  = 'n',
     xlab="", 
     ylab = "Energy Diff")

lines(merged_data$mean_power_dem.y,
      type = "l", 
      col ='red')

axis(1, 
     at=seq(0, 7*freq, by=freq),
     labels = FALSE)

text(seq(freq/2, 6.5*freq, by=freq), 
     par("usr")[3] - 0.2, 
     labels = days,  
     pos = 1, 
     xpd = TRUE)

legend(
  "topright",
  cex =0.65,
  lty=c(1,1),
  col=c("black", "red"),
  legend = c("Before Daylight Saving Date", 
             "After Daylight Saving Date")
)


## Cross Correlation calulation of dataset
cross_cor <-ccf(merged_data$mean_power_dem.x,
                merged_data$mean_power_dem.y)

cross_cor$acf <- round(cross_cor$acf,4)

cross_cor_info <- as.data.frame(cbind(cross_cor$acf,
                                     cross_cor$lag))
max_cros <- max(cross_cor_info$V1)

## generating the daylight saving time daytime saving lag information

cross_lag <- cross_cor_info[which(cross_cor_info$V1 == max_cros), ]
cross_lag$time_difference <- cross_lag$V2*time_freq
cross_lag$daylight_date <- as.Date(daylight_date$timestamp[i])

cross_lag$daylight_mismatch <- "Yes"
if((cross_lag$time_difference < 15) & 
   (cross_lag$time_difference > -15) ){
  cross_lag$daylight_mismatch <- "No"
}


names(cross_lag) <- col.names

daylight_results <- rbind(daylight_results, 
                          cross_lag)

}


}

## final result
daylight_results <- daylight_results[, c(4,1:3,5)]

return(daylight_results)

}
