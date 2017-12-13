# Working to convert zip codes and lat lon to climate zones
# 
# 

library(zipcode)
climate_zones = data.table::fread('Koeppen-Geiger-ASCII.txt')

unique_lati <- unique(climate_zones$Lat)
unique_long <- unique(climate_zones$Lon)

find_climate <- function(x, type = "zip") {
  data(zipcode)
  zipcode$zip <- as.integer(zipcode$zip)
  x <- as.integer(x)
  print(x)
 
  if (!is.na(x)) { 
    if (type == "zip") {
      lati <- zipcode[which(zipcode$zip == x), "latitude"]
      long <- zipcode[which(zipcode$zip == x), "longitude"]
      
      lati <- unique_lati[which.min(abs(unique_lati - lati))]
      long <- unique_long[which.min(abs(unique_long - long))]
      
      kg_climate <- dplyr::filter(climate_zones, Lat == lati & Lon == long)[['Cls']]
    }
  } else {
    return(NA)
  }
  kg_climate
}


