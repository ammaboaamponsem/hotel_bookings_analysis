# Load the necessary libraries
import pandas as pd
import numpy as np


# Load and display the dataset
hotel_bookings_df = pd.read_csv('hotel_bookings.csv')
print(hotel_bookings_df)


# 1. Data Preprocessing/Cleaning Stage


# Summary statistics of dataset
print(hotel_bookings_df.describe())

# Check the data types of the variables
print(hotel_bookings_df.dtypes)

# Check for missing values
print(hotel_bookings_df.isnull().sum())

# Replace the missing values of numerical variables with their means
numerical_vars = ['lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
                  'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
                  'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
                  'booking_changes', 'agent', 'company', 'days_in_waiting_list', 'adr',
                  'required_car_parking_spaces', 'total_of_special_requests']

for col in numerical_vars:
    hotel_bookings_df[col] =hotel_bookings_df[col].fillna(hotel_bookings_df[col].mean())

# Replace the missing values of categorical variables with their modes
categorical_vars = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel',
                    'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type',
                    'reservation_status', 'reservation_status_date']

for col in categorical_vars:
    hotel_bookings_df[col] =hotel_bookings_df[col].fillna(hotel_bookings_df[col].mode()[0])

# Confirm if missing values have been replaced
print(hotel_bookings_df.isnull().sum())


# Check for duplicates
duplicates = hotel_bookings_df[hotel_bookings_df.duplicated()]

# Print the duplicate rows
print("Duplicate Rows:")
print(duplicates)

# Remove duplicates from the DataFrame and overwrite the original DataFrame
hotel_bookings_df.drop_duplicates(inplace=True)

# Print the shape of the DataFrame after removing duplicates
print("Shape of DataFrame after removing duplicates:", hotel_bookings_df.shape)
