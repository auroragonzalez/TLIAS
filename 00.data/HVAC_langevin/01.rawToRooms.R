library(lattice)
library(ggplot2)
library(reshape2)
library(stats)
library(plyr)
library(dplyr)
library(xts)
library(lubridate)
library("data.table")

# Leemos los datos del repositorio
link = "https://bitbucket.org/aurorax/datangi/raw/162a9cc53e77c6dab67d88e6807b2160c262d1db/LANGEVIN_DATA.txt"
df = fread(link)

# Se crea una copia de los datos y se añade una nueva columna indicando el numero de fila
n <- 1:nrow(df)
df1 <- cbind(n, df)
names(df1$n) <- "N"
trabajadores <- 24

# Funcion auxiliar para hacer resampling por horas de las variables continuas
resample.dates <- function(x, FUN, units= "hours",...) {
  ep <- endpoints(x, units)
  x <- period.apply(x, ep, FUN, ...)
  index(x) <- round_date(index(x),unit = units)
  return (x)
}

# Contruimos un vector de fechas para hacer resampling
dates <- seq(ymd_hm('2013-07-01 00:00'), ymd_hm('2014-07-01 00:00'), by = '15 mins')
dates
# Contruimos un vector con las horas para la columna de timestamp
# hours <- data.frame(seq(ymd_hm('2013-07-01 00:00'), ymd_hm('2014-07-01 00:00'), by = '1 h'))
# hours

# Funcion que devuelve un vector con el timestamp para el resampling
ciclos <- function(maximo){
  aux <- NULL
  num <- 1
  while(num <= maximo){
    aux <- rbind(aux, df1$V1[num])
    num <- num+4
  }
  return (aux)
}

tiempo <- ciclos(nrow(df1)/trabajadores)

# Para cada trabajador hacemos resampling de las variables continuas y se guardan en un fichero (formato: Timestamp + variables continuas por horas)
for (i in 1:trabajadores) {
  filtroPersonas <- filter(df1, df1$V2 == i)        # Filtramos las filas de cada trabajador (variable V2 determina el trabajador)
  varCont <- filtroPersonas[,c(7:19,60:63,67,119)]  # Nos quedamos con las columnas de las variables continuas
  
  # Resampling datos
  data.xts <- xts(varCont, dates, tzone = Sys.getenv("TZ"))
  resamp <- resample.dates(data.xts, mean, units = "hours")

  # Creamos el dataframe con el timestamp y los valores de las variables continuas
  resultado <- data.frame(tiempo, resamp)
  names(resultado)[1] <- c("Timestamp")
  
  write.table(resultado, paste("./Trabajador",i,".csv", sep = ""), row.names = FALSE, na = "NA", col.names = TRUE, sep = ";", dec = ".")
}



