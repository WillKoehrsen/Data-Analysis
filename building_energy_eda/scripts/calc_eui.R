calculate_eui <- function(buildingdata, squarefeet, interval = 15){
  #Load libraries
  suppressMessages(library(data.table))
  library(ggplot2)
  
  
  #Create empty table with following coumns
  col.names<- c("SquareFootage","Annual_Energy(MBTU","Annual_Energy(MWh","EUI(kBTU/sf)")
  eui_info <- read.table(text = "", col.names = col.names)
  
  EUI_info <-c()
  
  #Transferring a few variables
  jci_data <- buildingdata
  
  #Total number of data points in a day
  Length <- (60/interval)*24
  #Extract data from the last year
  year_jci_data <- tail(jci_data, n = Length*365)
  # summation of electricty consumption
  year_cons <- sum(year_jci_data$forecast, rm.na = TRUE)
  year_cons_kbtu <- year_cons*3.41

  ## EUI equation
  eui <- year_cons_kbtu/squarefeet
  
  ## A table with the information calculated above
  eui_info[1,1] <- squarefeet
  eui_info[1,2] <- year_cons_kbtu/1000
  eui_info[1,3] <-year_cons/1000
  eui_info[1,4] <- eui
  
  
  return(eui_info)
}
