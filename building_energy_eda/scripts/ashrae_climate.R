library(httr)

latitude = 41.5
longitude = -81.7

url <- sprintf('http://data.fcc.gov/api/block/find?format=json&latitude=%s&longitude=%s&showall=true', latitude, longitude)

r <- GET(url, enco)

content <- content(r, "parsed")
county <- content$County[['name']]
state <- content$State[['code']]
