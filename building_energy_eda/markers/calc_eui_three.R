CalcEUI <- function(buildingdata, filename = "building", squarefootagedata, unit = "kWh"){
  #Load libraries
  library(data.table)
  library(BuildingRClean)
  library(ggplot2)


  #Create empty table with following coumns
  col.names<- c("Facility","SquareFootage","Annual_Energy(MBTU","Annual_Energy(MWh","EUI(kBTU/sf)")
  eui_info <- read.table(text = "", col.names = col.names)

  EUI_info <-c()

  #Transferring a few variables
  jci_filename <- filename
  jci_data <- buildingdata
  squarefootage <- squarefootagedata

  #Find the interval of the data in minutes
  interval <- time_frequency(jci_data)
  #Total number of data points in a day
  Length <- (60/interval)*24
  #Extract data from the last year
  year_jci_data <- tail(jci_data, n=Length*365)
  # summation of electricty consumption
  year_cons <- sum(year_jci_data$elec_cons, rm.na = TRUE)
  # converting kWh to kBTU
  if(toupper(unit)=="KWH"){
    year_cons_kbtu <- year_cons*3.41
  }
  #converting kW to kBTU
  if(toupper(unit)=="KW"){
    year_cons_kbtu <- year_cons*(interval/60)*3.41
  }
  ## EUI equation
  eui <- year_cons_kbtu/squarefootage

  ## A table with the information calculated above
  jci_filename1 <-gsub("-01.csv","",jci_filename)
  eui_info[1,1] <- jci_filename1
  eui_info[1,2] <- squarefootage
  eui_info[1,3] <- year_cons_kbtu/1000
  eui_info[1,4] <-year_cons/1000
  eui_info[1,5] <- eui

  EUI_info <-rbind(EUI_info,eui_info)

  ## just a plot function to get the eui values in a box plot


  ggplot(EUI_info,aes( Facility,EUI.kBTU.sf. ))+geom_bar(stat = "identity")+scale_y_log10()+
    theme(axis.text = element_text(angle = 90, hjust = 1,vjust = 0.5))+
    xlab("Facility ID")+ylab(bquote(EUI~(kBTU/ft^2)))

  return(EUI_info)
}
