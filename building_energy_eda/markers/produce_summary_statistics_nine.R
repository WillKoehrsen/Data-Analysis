#This function produces summary statistics for a buildings dataset. 
#The summary statistics include the mean, median, minimum, maximum, 
#first quartile and third quartile of the electrcitiy consumption and, 
#if available, for weather data which includes, cleaned energy,
#temperature, global horizontal irradiance, relative humidity and wind speed
#for the winters and summers for years for which data are available. 
#Summer consists of all the days in June, July and August.
#Winter consists of all the days in December, January, and February.
A <- data.table::fread("/mnt/projects/CSE_MSE_RXF131/hpc-members/kjn33/merged_data/APS_weather.csv")
#Convert timestamp to POSIXct
#A <- progressive_reader(progressive_list[1])    #read in data from spreadsheet
ProduceSummaryStatistics <- function(A){
  
  #Load libraries
  library(BuildingRClean)
  library(tidyr)
  library(dplyr)
  
  #Define functions
  CalcElec_consStatistics <- function(years, i, season){
    
    if (toupper(season) == "SUMMER"){
      if (is.data.frame(Summer) && nrow(Summer)!=0){
        
        #Mean
        meanec <- round(mean(Summer$elec_cons, na.rm = TRUE), 2)
        assign(years[i], meanec)          
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_mean_elec_cons", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_mean_elec_cons", sep = "_")
        
        #Median
        medianec <- round(median(Summer$elec_cons, na.rm = TRUE), 2)
        assign(years[i], medianec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_median_elec_cons", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_median_elec_cons", sep = "_")
        
        #Standard deviation
        sdec <- round(sd(Summer$elec_cons, na.rm = TRUE), 2)
        assign(years[i], sdec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_sd_elec_cons", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_sd_elec_cons", sep = "_")
        
        #Minimum
        minec <- round(min(Summer$elec_cons, na.rm = TRUE), 2)
        assign(years[i], minec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_min_elec_cons", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_min_elec_cons", sep = "_")
        
        #Maximum
        maxec <- round(max(Summer$elec_cons, na.rm = TRUE), 2)
        assign(years[i], maxec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_max_elec_cons", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_max_elec_cons", sep = "_")
        
        #First quartile
        firstqec <- 
          round(as.numeric(strsplit(as.character(quantile(Summer$elec_cons, 
                                                          na.rm = TRUE)[2]),
                                    "%")), 2)
        assign(years[i], firstqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_first_quartile_elec_cons", sep = "_"),
               x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_first_quartile_elec_cons", sep = "_")
        
        #Third quartile
        thirdqec <- 
          round(as.numeric(strsplit(as.character(quantile(Summer$elec_cons,
                                                          na.rm = TRUE)[4]),
                                    "%")), 2)
        assign(years[i], thirdqec)
        x[i] <- get(years[i]) 
        assign(paste(years[i], "summer_third_quartile_elec_cons", sep = "_"),
               x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_third_quartile_elec_cons", sep = "_")
        
        
      }
    }
    if (toupper(season) == "WINTER"){
      if (is.data.frame(Winter) && nrow(Winter)!=0){
        
        #Mean
        meanec <- round(mean(Winter$elec_cons, na.rm = TRUE), 2)
        assign(years[i], meanec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_mean_elec_cons", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_mean_elec_cons", sep = "_")
        
        #Median
        medianec <- round(median(Winter$elec_cons, na.rm = TRUE), 2)
        assign(years[i], medianec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_median_elec_cons", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_median_elec_cons", sep = "_")
        
        #Standard deviation
        sdec <- round(sd(Winter$elec_cons, na.rm = TRUE), 2)
        assign(years[i], sdec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_sd_elec_cons", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_sd_elec_cons", sep = "_")
        
        #Minimum
        minec <- min(range(Winter$elec_cons, na.rm = TRUE), 2)
        assign(years[i], minec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_min_elec_cons", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_min_elec_cons", sep = "_")
        
        #Maximum
        maxec <- round(max(Winter$elec_cons, na.rm = TRUE), 2)
        assign(years[i], maxec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_max_elec_cons", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_max_elec_cons", sep = "_")
        
        #First quartile
        firstqec <- 
          round(as.numeric(strsplit(as.character(quantile(Winter$elec_cons,
                                                          na.rm = TRUE)[2]),
                                    "%")), 2)
        assign(years[i], firstqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_first_quartile_elec_cons", sep = "_"),
               x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_first_quartile_elec_cons", sep = "_")
        
        #Third quartile
        thirdqec <- 
          round(as.numeric(strsplit(as.character(quantile(Winter$elec_cons,
                                                          na.rm = TRUE)[4]),
                                    "%")), 2)
        assign(years[i], thirdqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_third_quartile_elec_cons", sep = "_"),
               x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_third_quartile_elec_cons", sep = "_")
      }
    }
    return(mainfile)
  }
  CalcGHIStatistics <- function(years, i, season){
    #This function calculates the mean, median, standard deviation, minimum,
    #maximum, first quartile and third quartile for the global horizontal
    #irradiance either for the summer or winter. The arguments for the function
    #are: 'years', 'i', 'season'. 'years' is a vector containing all the years
    #for which data is available. 'i' is a counter which is used to index the 
    #'years' vector and extract data for a particular year. Can be implemented
    #using a for loop. 'season' must equal 'winter' or 'summer' (case insensitive)
    #Statistics for the data from the season selected will be computed accordingly
    #This function is part of the function, 'ProduceSummaryStatistics'
    #This function excludes all zero values for all the computations
    if (toupper(season) == "SUMMER"){
      if (is.data.frame(Summer) && nrow(Summer)!=0){
        
        #Mean
        meanec <- round(mean(Summer$ghi[Summer$ghi!=0], na.rm = TRUE), 2)
        assign(years[i], meanec)          
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_mean_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_mean_ghi", sep = "_")
        
        #Median
        medianec <- round(median(Summer$ghi[Summer$ghi!=0], na.rm = TRUE), 2)
        assign(years[i], medianec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_median_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_median_ghi", sep = "_")
        
        #Standard deviation
        sdec <- round(sd(Summer$ghi[Summer$ghi!=0], na.rm = TRUE), 2)
        assign(years[i], sdec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_sd_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_sd_ghi", sep = "_")
        
        #Minimum
        minec <- round(min(Summer$ghi[Summer$ghi!=0], na.rm = TRUE), 2)
        assign(years[i], minec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_min_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_min_ghi", sep = "_")
        
        #Maximum
        maxec <- round(max(Summer$ghi[Summer$ghi!=0], na.rm = TRUE), 2)
        assign(years[i], maxec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_max_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_max_ghi", sep = "_")
        
        #First quartile
        firstqec <- 
          round(as.numeric(strsplit(as.character(quantile(Summer$ghi[Summer$ghi!=0],
                                                          na.rm = TRUE)[2]),
                                    "%")), 2)
        assign(years[i], firstqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_first_quartile_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_first_quartile_ghi", sep = "_")
        
        #Third quartile
        thirdqec <- 
          round(as.numeric(strsplit(as.character(quantile(Summer$ghi[Summer$ghi!=0],
                                                          na.rm = TRUE)[4]),
                                    "%")), 2)
        assign(years[i], thirdqec)
        x[i] <- get(years[i]) 
        assign(paste(years[i], "summer_third_quartile_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_third_quartile_ghi", sep = "_")
        
        
      }
    }
    if (toupper(season) == "WINTER"){
      if (is.data.frame(Winter) && nrow(Winter)!=0){
        
        #Mean
        meanec <- round(mean(Winter$ghi[Winter$ghi!=0], na.rm = TRUE), 2)
        assign(years[i], meanec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_mean_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_mean_ghi", sep = "_")
        
        #Median
        medianec <- round(median(Winter$ghi[Winter$ghi!=0], na.rm = TRUE), 2)
        assign(years[i], medianec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_median_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_median_ghi", sep = "_")
        
        #Standard deviation
        sdec <- round(sd(Winter$ghi[Winter$ghi!=0], na.rm = TRUE), 2)
        assign(years[i], sdec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_sd_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_sd_ghi", sep = "_")
        
        #Minimum
        minec <- min(range(Winter$ghi[Winter$ghi!=0], na.rm = TRUE), 2)
        assign(years[i], minec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_min_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_min_ghi", sep = "_")
        
        #Maximum
        maxec <- round(max(Winter$ghi[Winter$ghi!=0], na.rm = TRUE), 2)
        assign(years[i], maxec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_max_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_max_ghi", sep = "_")
        
        #First quartile
        firstqec <- 
          round(as.numeric(strsplit(as.character(quantile(Winter$ghi[Winter$ghi!=0], 
                                                          na.rm = TRUE)[2]),
                                    "%")), 2)
        assign(years[i], firstqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_first_quartile_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_first_quartile_ghi", sep = "_")
        
        #Third quartile
        thirdqec <- 
          round(as.numeric(strsplit(as.character(quantile(Winter$ghi[Winter$ghi!=0],
                                                          na.rm = TRUE)[4]),
                                    "%")), 2)
        assign(years[i], thirdqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_third_quartile_ghi", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_third_quartile_ghi", sep = "_")
      }
    }
    return(mainfile)
  }
  CalcCleaned_energyStatistics <- function(years, i, season){
    #This function calculates the mean, median, standard deviation, minimum,
    #maximum, first quartile and third quartile for the cleaned energy
    #either for the summer or winter. The arguments for the function
    #are: 'years', 'i', 'season'. 'years' is a vector containing all the years
    #for which data is available. 'i' is a counter which is used to index the 
    #'years' vector and extract data for a particular year. Can be implemented
    #using a for loop. 'season' must equal 'winter' or 'summer' (case insensitive)
    #Statistics for the data from the season selected will be computed accordingly
    #This function is part of the function, 'ProduceSummaryStatistics'
    if (toupper(season) == "SUMMER"){
      if (is.data.frame(Summer) && nrow(Summer)!=0){
        
        #Mean
        meanec <- round(mean(Summer$cleaned_energy, na.rm = TRUE), 2)
        assign(years[i], meanec)          
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_mean_cleaned_energy", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_mean_cleaned_energy", sep = "_")
        
        #Median
        medianec <- round(median(Summer$cleaned_energy, na.rm = TRUE), 2)
        assign(years[i], medianec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_median_cleaned_energy", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_median_cleaned_energy", sep = "_")
        
        #Standard deviation
        sdec <- round(sd(Summer$cleaned_energy, na.rm = TRUE), 2)
        assign(years[i], sdec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_sd_cleaned_energy", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_sd_cleaned_energy", sep = "_")
        
        #Minimum
        minec <- round(min(Summer$cleaned_energy, na.rm = TRUE), 2)
        assign(years[i], minec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_min_cleaned_energy", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_min_cleaned_energy", sep = "_")
        
        #Maximum
        maxec <- round(max(Summer$cleaned_energy, na.rm = TRUE), 2)
        assign(years[i], maxec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_max_cleaned_energy", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_max_cleaned_energy", sep = "_")
        
        #First quartile
        firstqec <- 
          round(as.numeric(strsplit(as.character(quantile(Summer$cleaned_energy,
                                                          na.rm = TRUE)[2]),
                                    "%")), 2)
        assign(years[i], firstqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_first_quartile_cleaned_energy", sep = "_"),
               x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_first_quartile_cleaned_energy", sep = "_")
        
        #Third quartile
        thirdqec <- 
          round(as.numeric(strsplit(as.character(quantile(Summer$cleaned_energy,
                                                          na.rm = TRUE)[4]),
                                    "%")), 2)
        assign(years[i], thirdqec)
        x[i] <- get(years[i]) 
        assign(paste(years[i], "summer_third_quartile_cleaned_energy", sep = "_"),
               x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_third_quartile_cleaned_energy", sep = "_")
        
        
      }
    }
    if (toupper(season) == "WINTER"){
      if (is.data.frame(Winter) && nrow(Winter)!=0){
        
        #Mean
        meanec <- round(mean(Winter$cleaned_energy, na.rm = TRUE), 2)
        assign(years[i], meanec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_mean_cleaned_energy", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_mean_cleaned_energy", sep = "_")
        
        #Median
        medianec <- round(median(Winter$cleaned_energy, na.rm = TRUE), 2)
        assign(years[i], medianec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_median_cleaned_energy", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_median_cleaned_energy", sep = "_")
        
        #Standard deviation
        sdec <- round(sd(Winter$cleaned_energy, na.rm = TRUE), 2)
        assign(years[i], sdec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_sd_cleaned_energy", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_sd_cleaned_energy", sep = "_")
        
        #Minimum
        minec <- min(range(Winter$cleaned_energy, na.rm = TRUE), 2)
        assign(years[i], minec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_min_cleaned_energy", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_min_cleaned_energy", sep = "_")
        
        #Maximum
        maxec <- round(max(Winter$cleaned_energy, na.rm = TRUE), 2)
        assign(years[i], maxec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_max_cleaned_energy", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_max_cleaned_energy", sep = "_")
        
        #First quartile
        firstqec <- 
          round(as.numeric(strsplit(as.character(quantile(Winter$cleaned_energy,
                                                          na.rm = TRUE)[2]),
                                    "%")), 2)
        assign(years[i], firstqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_first_quartile_cleaned_energy", sep = "_"),
               x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_first_quartile_cleaned_energy", sep = "_")
        
        #Third quartile
        thirdqec <- 
          round(as.numeric(strsplit(as.character(quantile(Winter$cleaned_energy, 
                                                          na.rm = TRUE)[4]),
                                    "%")), 2)
        assign(years[i], thirdqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_third_quartile_cleaned_energy", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_third_quartile_cleaned_energy", sep = "_")
      }
    }
    return(mainfile)
  }
  CalcRHStatistics <- function(years, i, season){
    #This function calculates the mean, median, standard deviation, minimum,
    #maximum, first quartile and third quartile for the relative humidity
    #either for the summer or winter. The arguments for the function
    #are: 'years', 'i', 'season'. 'years' is a vector containing all the years
    #for which data is available. 'i' is a counter which is used to index the 
    #'years' vector and extract data for a particular year. Can be implemented
    #using a for loop. 'season' must equal 'winter' or 'summer' (case insensitive)
    #Statistics for the data from the season selected will be computed accordingly
    #This function is part of the function, 'ProduceSummaryStatistics'
    if (toupper(season) == "SUMMER"){
      if (is.data.frame(Summer) && nrow(Summer)!=0){
        
        #Mean
        meanec <- round(mean(Summer$rh, na.rm = TRUE), 2)
        assign(years[i], meanec)          
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_mean_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_mean_rh", sep = "_")
        
        #Median
        medianec <- round(median(Summer$rh, na.rm = TRUE), 2)
        assign(years[i], medianec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_median_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_median_rh", sep = "_")
        
        #Standard deviation
        sdec <- round(sd(Summer$rh, na.rm = TRUE), 2)
        assign(years[i], sdec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_sd_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_sd_rh", sep = "_")
        
        #Minimum
        minec <- round(min(Summer$rh, na.rm = TRUE), 2)
        assign(years[i], minec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_min_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_min_rh", sep = "_")
        
        #Maximum
        maxec <- round(max(Summer$rh, na.rm = TRUE), 2)
        assign(years[i], maxec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_max_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_max_rh", sep = "_")
        
        #First quartile
        firstqec <- round(as.numeric(strsplit(as.character(quantile(Summer$rh,
                                                                    na.rm = TRUE)[2]),
                                              "%")), 2)
        assign(years[i], firstqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_first_quartile_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_first_quartile_rh", sep = "_")
        
        #Third quartile
        thirdqec <- 
          round(as.numeric(strsplit(as.character(quantile(Summer$rh,
                                                          na.rm = TRUE)[4]),
                                    "%")), 2)
        assign(years[i], thirdqec)
        x[i] <- get(years[i]) 
        assign(paste(years[i], "summer_third_quartile_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_third_quartile_rh", sep = "_")
        
        
      }
    }
    if (toupper(season) == "WINTER"){
      if (is.data.frame(Winter) && nrow(Winter)!=0){
        
        #Mean
        meanec <- round(mean(Winter$rh, na.rm = TRUE), 2)
        assign(years[i], meanec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_mean_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_mean_rh", sep = "_")
        
        #Median
        medianec <- round(median(Winter$rh, na.rm = TRUE), 2)
        assign(years[i], medianec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_median_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_median_rh", sep = "_")
        
        #Standard deviation
        sdec <- round(sd(Winter$rh, na.rm = TRUE), 2)
        assign(years[i], sdec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_sd_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_sd_rh", sep = "_")
        
        #Minimum
        minec <- min(range(Winter$rh, na.rm = TRUE), 2)
        assign(years[i], minec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_min_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_min_rh", sep = "_")
        
        #Maximum
        maxec <- round(max(Winter$rh, na.rm = TRUE), 2)
        assign(years[i], maxec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_max_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_max_rh", sep = "_")
        
        #First quartile
        firstqec <- 
          round(as.numeric(strsplit(as.character(quantile(Winter$rh, 
                                                          na.rm = TRUE)[2]),
                                    "%")), 2)
        assign(years[i], firstqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_first_quartile_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_first_quartile_rh", sep = "_")
        
        #Third quartile
        thirdqec <- 
          round(as.numeric(strsplit(as.character(quantile(Winter$elec_cons,
                                                          na.rm = TRUE)[4]),
                                    "%")), 2)
        assign(years[i], thirdqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_third_quartile_rh", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_third_quartile_rh", sep = "_")
      }
    }
    return(mainfile)
  }
  CalcTempStatistics <- function(years, i, season){
    #This function calculates the mean, median, standard deviation, minimum,
    #maximum, first quartile and third quartile for the temperature
    #either for the summer or winter. The arguments for the function
    #are: 'years', 'i', 'season'. 'years' is a vector containing all the years
    #for which data is available. 'i' is a counter which is used to index the 
    #'years' vector and extract data for a particular year. Can be implemented
    #using a for loop. 'season' must equal 'winter' or 'summer' (case insensitive)
    #Statistics for the data from the season selected will be computed accordingly
    #This function is part of the function, 'ProduceSummaryStatistics'
    if (toupper(season) == "SUMMER"){
      if (is.data.frame(Summer) && nrow(Summer)!=0){
        
        #Mean
        meanec <- round(mean(Summer$temp, na.rm = TRUE), 2)
        assign(years[i], meanec)          
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_mean_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_mean_temp", sep = "_")
        
        #Median
        medianec <- round(median(Summer$temp, na.rm = TRUE), 2)
        assign(years[i], medianec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_median_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_median_temp", sep = "_")
        
        #Standard deviation
        sdec <- round(sd(Summer$temp, na.rm = TRUE), 2)
        assign(years[i], sdec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_sd_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_sd_temp", sep = "_")
        
        #Minimum
        minec <- round(min(Summer$temp, na.rm = TRUE), 2)
        assign(years[i], minec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_min_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_min_temp", sep = "_")
        
        #Maximum
        maxec <- round(max(Summer$temp, na.rm = TRUE), 2)
        assign(years[i], maxec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_max_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_max_temp", sep = "_")
        
        #First quartile
        firstqec <- round(as.numeric(strsplit(as.character(quantile(Summer$temp,
                                                                    na.rm = TRUE)[2]),
                                              "%")), 2)
        assign(years[i], firstqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_first_quartile_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_first_quartile_temp", sep = "_")
        
        #Third quartile
        thirdqec <- 
          round(as.numeric(strsplit(as.character(quantile(Summer$temp,
                                                          na.rm = TRUE)[4]),
                                    "%")), 2)
        assign(years[i], thirdqec)
        x[i] <- get(years[i]) 
        assign(paste(years[i], "summer_third_quartile_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_third_quartile_temp", sep = "_")
        
        
      }
    }
    if (toupper(season) == "WINTER"){
      if (is.data.frame(Winter) && nrow(Winter)!=0){
        
        #Mean
        meanec <- round(mean(Winter$temp, na.rm = TRUE), 2)
        assign(years[i], meanec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_mean_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_mean_temp", sep = "_")
        
        #Median
        medianec <- round(median(Winter$temp, na.rm = TRUE), 2)
        assign(years[i], medianec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_median_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_median_temp", sep = "_")
        
        #Standard deviation
        sdec <- round(sd(Winter$temp, na.rm = TRUE), 2)
        assign(years[i], sdec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_sd_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_sd_temp", sep = "_")
        
        #Minimum
        minec <- min(range(Winter$temp, na.rm = TRUE), 2)
        assign(years[i], minec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_min_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_min_temp", sep = "_")
        
        #Maximum
        maxec <- round(max(Winter$temp, na.rm = TRUE), 2)
        assign(years[i], maxec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_max_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_max_temp", sep = "_")
        
        #First quartile
        firstqec <- 
          round(as.numeric(strsplit(as.character(quantile(Winter$temp, 
                                                          na.rm = TRUE)[2]),
                                    "%")), 2)
        assign(years[i], firstqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_first_quartile_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_first_quartile_temp", sep = "_")
        
        #Third quartile
        thirdqec <- 
          round(as.numeric(strsplit(as.character(quantile(Winter$temp,
                                                          na.rm = TRUE)[4]),
                                    "%")), 2)
        assign(years[i], thirdqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_third_quartile_temp", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_third_quartile_temp", sep = "_")
      }
    }
    return(mainfile)
  }
  CalcWSStatistics <- function(years, i, season){
    #This function calculates the mean, median, standard deviation, minimum,
    #maximum, first quartile and third quartile for the wind speed
    #either for the summer or winter. The arguments for the function
    #are: 'years', 'i', 'season'. 'years' is a vector containing all the years
    #for which data is available. 'i' is a counter which is used to index the 
    #'years' vector and extract data for a particular year. Can be implemented
    #using a for loop. 'season' must equal 'winter' or 'summer' (case insensitive)
    #Statistics for the data from the season selected will be computed accordingly
    #This function is part of the function, 'ProduceSummaryStatistics'
    if (toupper(season) == "SUMMER"){
      if (is.data.frame(Summer) && nrow(Summer)!=0){
        
        #Mean
        meanec <- round(mean(Summer$ws, na.rm = TRUE), 2)
        assign(years[i], meanec)          
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_mean_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_mean_ws", sep = "_")
        
        #Median
        medianec <- round(median(Summer$ws, na.rm = TRUE), 2)
        assign(years[i], medianec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_median_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_median_ws", sep = "_")
        
        #Standard deviation
        sdec <- round(sd(Summer$ws, na.rm = TRUE), 2)
        assign(years[i], sdec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_sd_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_sd_ws", sep = "_")
        
        #Minimum
        minec <- round(min(Summer$ws, na.rm = TRUE), 2)
        assign(years[i], minec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_min_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_min_ws", sep = "_")
        
        #Maximum
        maxec <- round(max(Summer$ws, na.rm = TRUE), 2)
        assign(years[i], maxec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_max_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_max_ws", sep = "_")
        
        #First quartile
        firstqec <- round(as.numeric(strsplit(as.character(quantile(Summer$ws,
                                                                    na.rm = TRUE)[2]),
                                              "%")), 2)
        assign(years[i], firstqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "summer_first_quartile_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_first_quartile_ws", sep = "_")
        
        #Third quartile
        thirdqec <- 
          round(as.numeric(strsplit(as.character(quantile(Summer$ws,
                                                          na.rm = TRUE)[4]),
                                    "%")), 2)
        assign(years[i], thirdqec)
        x[i] <- get(years[i]) 
        assign(paste(years[i], "summer_third_quartile_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "summer_third_quartile_ws", sep = "_")
        
        
      }
    }
    if (toupper(season) == "WINTER"){
      if (is.data.frame(Winter) && nrow(Winter)!=0){
        
        #Mean
        meanec <- round(mean(Winter$ws, na.rm = TRUE), 2)
        assign(years[i], meanec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_mean_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_mean_ws", sep = "_")
        
        #Median
        medianec <- round(median(Winter$ws, na.rm = TRUE), 2)
        assign(years[i], medianec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_median_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_median_ws", sep = "_")
        
        #Standard deviation
        sdec <- round(sd(Winter$ws, na.rm = TRUE), 2)
        assign(years[i], sdec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_sd_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_sd_ws", sep = "_")
        
        #Minimum
        minec <- min(range(Winter$ws, na.rm = TRUE), 2)
        assign(years[i], minec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_min_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_min_ws", sep = "_")
        
        #Maximum
        maxec <- round(max(Winter$ws, na.rm = TRUE), 2)
        assign(years[i], maxec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_max_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_max_ws", sep = "_")
        
        #First quartile
        firstqec <- 
          round(as.numeric(strsplit(as.character(quantile(Winter$ws, 
                                                          na.rm = TRUE)[2]),
                                    "%")), 2)
        assign(years[i], firstqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_first_quartile_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_first_quartile_ws", sep = "_")
        
        #Third quartile
        thirdqec <- 
          round(as.numeric(strsplit(as.character(quantile(Winter$ws,
                                                          na.rm = TRUE)[4]),
                                    "%")), 2)
        assign(years[i], thirdqec)
        x[i] <- get(years[i])
        assign(paste(years[i], "winter_third_quartile_ws", sep = "_"), x[i])
        mainfile <- cbind(mainfile, get(years[i]))
        colnames(mainfile)[which(names(mainfile) == "get(years[i])")] <- 
          paste(years[i], "winter_third_quartile_ws", sep = "_")
      }
    }
    return(mainfile)
  }
  
  A$timestamp <- as.POSIXct(A$timestamp)       #convert timestamps to POSIXct
  #Separate columns for year and month
  if("temp" %in% colnames(A)){                 #if merged weather data is present
    B <-A%>%
      mutate(timestamp=as.character(timestamp)) %>%        
      mutate(year = substring(timestamp,1,4))%>%           
      mutate(month= substring(timestamp,6,7))%>%          
      select(timestamp, cleaned_energy, temp, ghi, rh, ws, elec_cons, year, month)
  } else {       #if only building data is present
    B <-A%>%
      mutate(timestamp=as.character(timestamp)) %>%
      mutate(year = substring(timestamp,1,4))%>%
      mutate(month= substring(timestamp,6,7))%>%
      select(timestamp, elec_cons, year, month)
      #mutate(timestamp=as.POSIXct(timestamp))
  }
  years <- unique(B$year)                      #extract unique years
  x <- 0      #pre-define x
  #Data frame that will contain the summary statistics
  mainfile <- building_summary(A)
  for (i in 1:length(years)){                  #iterate from 1 to number of years
    C <- B[which(B$year==years[i]),]           #extract data for ith year
  
    ##Summer
    #Extract data from Jun, Jul and Aug for that year
    Summer <- C[which(C$month==c("07","06","08")),]
    #Check if data for summer months for that year exists
    if (is.data.frame(Summer) && nrow(Summer)!=0){
      mainfile <- CalcElec_consStatistics(years, i, season = "summer")
      if ("temp" %in% colnames(Summer)){       #if merged weather data is present
        mainfile <- CalcCleaned_energyStatistics(years, i, season = "summer")
        mainfile <- CalcRHStatistics(years, i, season = "summer")
        mainfile <- CalcGHIStatistics(years, i, season = "summer")
        mainfile <- CalcTempStatistics(years, i, season = "summer")
        mainfile <- CalcWSStatistics(years, i, season = "summer")
      }
    
    
    }   
    ##Winter
    # Extract data from Dec, Jan, and Feb for that year
    Winter <- C[which(C$month==c("12","01","02")),]
    #Check if data for winter months for that year exists
    if (is.data.frame(Winter) && nrow(Winter)!=0){
      mainfile <- CalcElec_consStatistics(years, i, season = "winter")
      if ("temp" %in% colnames(Winter)){       #if merged weather data is present
        mainfile <- CalcCleaned_energyStatistics(years, i, season = "winter")
        mainfile <- CalcRHStatistics(years, i, season = "winter")
        mainfile <- CalcGHIStatistics(years, i, season = "winter")
        mainfile <- CalcTempStatistics(years, i, season = "winter")
        mainfile <- CalcWSStatistics(years, i, season = "winter")
      }
    
    }
    ssdata <- t(mainfile)                     #tranpose data frame
  }
  return(ssdata)
}