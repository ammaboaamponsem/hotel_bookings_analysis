# Question 1
# Create a bar plot for cancellation rate by hotel type
sns.countplot(x='hotel', hue='is_canceled', data=hotel_bookings_df)
plt.title('Cancellation Rate by Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('No. of Bookings')
plt.legend(title='Cancellation Status', labels=['Not Canceled', 'Canceled'])
plt.show()

# Question 2
# Create a line plot for seasonal booking trends
bookings_by_month = hotel_bookings_df.groupby('arrival_date_month')['hotel'].count().reset_index()
sns.lineplot(x='arrival_date_month', y='hotel', data=bookings_by_month)
plt.title('Seasonal Booking Trends')
plt.xlabel('Month')
plt.ylabel('No. of Bookings')
plt.xticks(rotation=45)
plt.show()

# Question 3
# Create a box plot for the lead time distribution by cancellation status
sns.boxplot(x='is_canceled', y='lead_time', data=hotel_bookings_df)
plt.title('Lead Time Distribution by Cancellation Status')
plt.xlabel('Cancellation Status')
plt.ylabel('Lead Time')
plt.xticks([0, 1], ['Not Canceled', 'Canceled'])
plt.show()

