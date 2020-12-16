# predicting-house-prices-in-bengaluru
MachineHack Hackathon: https://www.machinehack.com/hackathons/predicting_house_prices_in_bengaluru/overview

## Achievement:
Rank 18 on the leaderboard (as of 15-December-2020)

## Overview
What are the things that a potential home buyer considers before purchasing a house? The location, the size of the property, vicinity to offices, schools, parks, restaurants, hospitals or the stereotypical white picket fence? What about the most important factor -- the price? Now with the lingering impact of demonetization, the enforcement of the Real Estate (Regulation and Development) Act (RERA), and the lack of trust in property developers in the city, housing units sold across India in 2017 dropped by 7 percent. In fact, the property prices in Bengaluru fell by almost 5 percent in the second half of 2017, said a study published by property consultancy Knight Frank. For example, for a potential homeowner, over 9,000 apartment projects and flats for sale are available in the range of ₹42-52 lakh, followed by over 7,100 apartments that are in the ₹52-62 lakh budget segment, says a report by property website Makaan. According to the study, there are over 5,000 projects in the ₹15-25 lakh budget segment followed by those in the ₹34-43 lakh budget category. Buying a home, especially in a city like Bengaluru, is a tricky choice. While the major factors are usually the same for all metros, there are others to be considered for the Silicon Valley of India. With its help millennial crowd, vibrant culture, great climate and a slew of job opportunities, it is difficult to ascertain the price of a house in Bengaluru.   So what determines the property prices in Namma Bengaluru? 

## Data  
The train and test data will consist of various features that describe that property in Bengaluru. This is an actual data set that is curated over months of primary & secondary research by our team. Each row contains fixed size object of features. There are 9 features and each feature can be accessed by its name. Features Area_type - describes the area Availability - when it can be possessed or when it is ready(categorical and time-series) Location - where it is located in Bengaluru Price - Value of the property in lakhs(INR) Size - in BHK or Bedroom (1-10 or more) Society - to which society it belongs Total_sqft - size of the property in sq.ft Bath - No. of bathrooms Balcony - No. of the balcony Problem Statement With the given 9 features(categorical and continuous) build a model to predict the price of houses in Bengaluru.

## Evaluation Metrics
1 - np.sqrt(np.square(np.log10(y_pred +1) - np.log10(y_true +1)).mean())


