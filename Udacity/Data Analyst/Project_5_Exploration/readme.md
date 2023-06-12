# Renting Bike Exploration - Fordgobike and Divvy Trip Systems

## Dataset

The dataset consists of two datasets of two different cities; San Francisco and Chicago. 
There are 268465 individuals rides recorded from the Ford GoBike System (San Francisco) and the Divvy Trips System (Chicago). 
The data gives insights about the duration of a ride, the gender, user type and the birth year. 
Furthermore there are information about the trip distance in the San Francisco dataset.

There two different user types:

- Customer and Subscriber

Gender in:

- Male, Female and other

The columns are as followed:

 0.   trip_duration_seconds       
 1.   start_time             
 2.   end_time               
 3.   start_station_id             
 4.   start_station_name      
 5.   end_station_id          
 6.   end_station_name         
 7.   user_type              
 8.   member_birth_year     
 9.   member_gender          
 10.  city                    
 11.  start_coordinates       
 12.  end_coordinates
 13.  distance

The dataset only gives information about ride behavoiur of February 2019.

One dataframe is provided directly from Udacity. But the divvy trip data can be found [here](https://divvy-tripdata.s3.amazonaws.com/Divvy_Trips_2019_Q1.zip)

## Summary of Findings

In the exploration I found that the main users are male and subscriber. Round about 77% of the users are male and 93%
of the users are subscriber. 65 % of the records are from San Francisco.

The most trips are taken on workdays with the highest peek on thursdays and there is less traffic on weekends. Probably many users
renting a bike to get to their workplace. The highest traffic is obviously at rush hour times in the morning and afternoon.
A trip takes usually 6-12 minutes and is 1.0 to 1.6km long.

The differences in the proportion of the bike usage separated in gender is really small, so it doesn't provide so much information
about the usage. Male drivers usually take a shorter trip than women or others. 

The differences in the usertype is much bigger. The average trip duration for customers is much higher than subscribers. 
Subscribers takes in average a ride about 10 mins on all days. 
Customers takes a ride 13-15 mins on workdays and at the weekend over 17,5 mins.

Obviously the trip distance shows a strong positive relationship to the trip duration. But this chart/scatterplot shows that some people
rent the bike at one station and returned it to same station.

## Key Insights for Presentation

In this investigation I wanted to look for differences between gender and user type in the usage of the bike sharing platform. 
I focused on the proportions over weekdays and day time and compared the differences in the average trip durations.