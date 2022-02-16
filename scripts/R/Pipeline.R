library(tidyverse)
owid_covid_data <- read_csv(
	"~/Desktop/owid-covid-data.csv",
	col_types = cols(iso_code = col_skip(), 
										 continent = col_skip(),
										 location = col_factor(),
										 date = col_datetime(
										 	format = "%Y-%m-%d")))  %>% 
	select(location, date, new_cases) %>% 
	filter(location %in% c('Canada','France','India', 'South Korea', 'United Kingdom'),
					 date >= as.Date('2020-01-20'),
					 date <= as.Date('2020-04-04')) %>% 
	arrange(date) %>% 
	pivot_wider(
		names_from = 'location', 
		values_from = 'new_cases') %>% 
	replace(is.na(.), 0) %>% 
	write_csv("~/Desktop/ML Systems for Data Science/Project/data/Realtime_cases_plot.csv")

