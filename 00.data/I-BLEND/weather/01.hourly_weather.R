library("data.table")
data2013 = fread("delhi_weatherData_year2013_FULL.csv")
data2014 = fread("delhi_weatherData_year2014_FULL_15mins.csv")
data2015 = fread("delhi_weatherData_year2015_FULL_15mins.csv")
data2016 = fread("delhi_weatherData_year2016_FULL.csv")
data2017 = fread("delhi_weatherData_year2017_FULL.csv")

names(data2014) = names(data2013)
names(data2015) = names(data2013)
names(data2016) = names(data2013)
names(data2017) = names(data2013)

datosWeather = rbind(data2013, data2014, data2015, data2016, data2017)
Sys.setenv(TZ='Asia/Kolkata')
datosWeather$timestamp = as.POSIXct(datosWeather$timestamp, format = '%Y-%m-%d %H:%M:%S')
occ = datosWeather
occ$hour <- format(occ$timestamp, format = '%H')
occ$day<- format(occ$timestamp, format = '%Y-%m-%d')
pivot <- aggregate(occ$TemperatureC, by = list(occ$day, occ$hour), FUN = mean)
pivot$timestamp = paste0(pivot$Group.1, " ", pivot$Group.2,":00:00")
pivot$timestamp = as.POSIXct(pivot$timestamp, format = '%Y-%m-%d %H:%M:%S', tz = "Asia/Kolkata", origin = "1970-01-01")
final = pivot[c("timestamp", "x")]
names(final)[2] = "temp"
write.table(final, "temperatures.csv", sep=";")

library(zoo)

cons = read.table("dataprocessed/hourly/ACB-AcademicBuilding.csv", sep=";")
occ = read.table("IIITD_occupancy_dataset/IIITD_occupancy_dataset/ACB.csv", sep=",", head=T)
occ$timestamp <- as.POSIXct(occ$timestamp, tz = "Asia/Kolkata", origin = "1970-01-01")
occ$hour <- format(occ$timestamp, format = '%H')
occ$day<- format(occ$timestamp, format = '%Y-%m-%d')
pivot <- aggregate(occ$occupancy_count, by = list(occ$day, occ$hour), FUN = mean)
cons$timestamp = as.numeric(as.POSIXct(rownames(cons)))
pivot$timestamp2 = paste0(pivot$Group.1, " ", pivot$Group.2,":00:00")
pivot$timestamp = as.numeric(as.POSIXct(pivot$timestamp2, format = '%Y-%m-%d %H:%M:%S'))
OCC = pivot

merge = merge(x = OCC, y = cons, by = "timestamp", all = F)
merge$datH = as.numeric(na.approx(zoo(merge$datH)))
str(merge)

# Este es el formato que tiene que tener:
#cons = pd.read_csv('https://bitbucket.org/aurorax/datangi/raw/'+lastCommit+'/processed/study1-energy/'+building+'-tsCons.csv', sep=';', index_col=0)

occFinal = merge[c("timestamp2", "x")]
names(occFinal) = c("timestamp", "occ")

consFinal = merge[c("timestamp2", "datH")]
names(consFinal) = c("timestamp", "cons")

write.table(consFinal, "dataprocessed/cons/ACB-AcademicBuilding.csv", sep=";", row.names = F)
write.table(occFinal, "dataprocessed/occ/ACB-AcademicBuilding.csv", sep=";", row.names = F)



buildings = dir("dataprocessed/hourly/")
buildings = buildings[-2] # al parecer, BH-02 contiene todo lo que tiene BH-01 y algo más
bOcc = c("ACB.csv", "BH.csv", "GH.csv", "GH.csv", "LB.csv", "LB.csv", "LCB.csv") 
for(i in 1:length(buildings)){
  name = buildings[i]
  print(name)
  name2 = bOcc[i]
  cons = read.table(paste0("dataprocessed/hourly/",name), sep=";")
  occ = read.table(paste0("IIITD_occupancy_dataset/IIITD_occupancy_dataset/",name2), sep=",", head=T)
  Sys.setenv(TZ='Asia/Kolkata')
  occ$timestamp <- as.POSIXct(occ$timestamp, tz = "Asia/Kolkata", origin = "1970-01-01")
  occ$hour <- format(occ$timestamp, format = '%H')
  occ$day<- format(occ$timestamp, format = '%Y-%m-%d')
  pivot <- aggregate(occ$occupancy_count, by = list(occ$day, occ$hour), FUN = mean)
  cons$timestamp = as.numeric(as.POSIXct(rownames(cons)))
  pivot$timestamp2 = paste0(pivot$Group.1, " ", pivot$Group.2,":00:00")
  pivot$timestamp = as.numeric(as.POSIXct(pivot$timestamp2, format = '%Y-%m-%d %H:%M:%S'))
  OCC = pivot
  
  merge = merge(x = OCC, y = cons, by = "timestamp", all = F)
  merge[ncol(merge)] = as.numeric(na.approx(zoo(merge[ncol(merge)])))
  str(merge)
  
  # Este es el formato que tiene que tener:
  #cons = pd.read_csv('https://bitbucket.org/aurorax/datangi/raw/'+lastCommit+'/processed/study1-energy/'+building+'-tsCons.csv', sep=';', index_col=0)
  
  occFinal = merge[c("timestamp2", "x")]
  names(occFinal) = c("timestamp", "occ")
  
  consFinal = merge[c("timestamp2", "datH")]
  names(consFinal) = c("timestamp", "cons")
  
  write.table(consFinal, paste0("dataprocessed/cons/",name), sep=";", row.names = F)
  write.table(occFinal, paste0("dataprocessed/occ/",name), sep=";", row.names = F)
  
}
