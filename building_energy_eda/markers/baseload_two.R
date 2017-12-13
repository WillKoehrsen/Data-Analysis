# baseload function to calculate the baseload of energy consumption
# The input should be energy dataset with timestamp and cleaned_energy column
# Also, the energy column can be changed as specifying in energy_coliumn input
# for example energy_column ="elec_cons".

 

  baseload <- function(energy_data, energy_column = "cleaned_energy"){
  df = subset(energy_data, 
              select = c('timestamp',
                        energy_column))
  
  names(df)[2] <- c("cleaned_energy")
  df = data.frame(df)
  df[,1] = as.POSIXct(df[, 1])
  
  df$Date = as.Date(df$timestamp, 
                    tz = "UTC")
 
  df$cleaned_energy <- forecast::na.interp(df$cleaned_energy)
  df$Day <- (format(df$Date,'%A'))
  df$Month <- (format(df$Date,'%B'))
 
  df$Time <- format(df$timestamp,"%H:%M:%S")
  df$Hours <- format(as.POSIXct(strptime(df$timestamp,
                                         "%d/%m/%Y %H:%M:%S",
                                         tz="")) ,
                     format = "%H:%M:%S")

# filter to smooth the dataset
  bf <- signal::butter(3, 0.1)
  b11 <- signal::filtfilt(bf, 
                  df$cleaned_energy)

  df$smooth <- b11

  names(df)[2] <- "Energy"
  names(df)[8] <- "Filtered"
  Energy <- df[10:(length(df$Energy) - 10),
             c(1, 2, 8)]
 

  Energy$day <- format(Energy$timestamp, 
                     format = "%Y-%m-%d")

  agg4 <- aggregate(Energy$Filtered,
                    list(id1 = Energy$day),
                    min)

  agg4.ord <- agg4[order((agg4$x)),]

     
  agg4$id1 <- as.POSIXct(agg4$id1)
  
#extreme point removal  
   A <- outliers:: outlier(agg4.ord$x)

  L <- extremevalues::getOutliersI(agg4.ord$x, rho=c(1,1), 
                    FLim = c(0.05, 0.95),
                    distribution = "normal")
    agg4.rm.outlier <- agg4.ord[which(agg4.ord$x!= A), ]
  agg4.rm.outlier <- agg4.rm.outlier[which(agg4.rm.outlier$x > 0.1), ]
  
  if(L$nOut[1]!= 0){
  agg4.rm.outlier <- agg4.rm.outlier[which(agg4.rm.outlier$x > L$limit[1]), ]
}
  agg4.outlier <- agg4.ord[which(agg4.ord$x == A), ]

  agg4.outlier <- agg4.outlier[which(agg4.outlier$x > 0), ]
  baseload <- mean(agg4.rm.outlier[1:10, 2])
  baseload <- round(baseload, 2)

return(baseload)
}

