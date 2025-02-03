import matplotlib.pyplot as plt
import seaborn as sns


# 2. EDA Stage


# Select only numerical columns for correlation analysis
numerical_columns = hotel_bookings_df.select_dtypes(include=['number'])

# Compute the correlation matrix
correlation_matrix = numerical_columns.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlation Matrix Plot')
plt.show()

# Create a bar plot for cancellation rate by hotel type
sns.countplot(x='hotel', hue='is_canceled', data=hotel_bookings_df)
plt.title('Cancellation Rate by Hotel Type')
plt.xlabel('Hotel Type')
plt.ylabel('No. of Bookings')
plt.legend(title='Cancellation Status', labels=['Not Canceled', 'Canceled'])
plt.show()

# Create a line plot for seasonal booking trends
bookings_by_month = hotel_bookings_df.groupby('arrival_date_month')['hotel'].count().reset_index()
sns.lineplot(x='arrival_date_month', y='hotel', data=bookings_by_month)
plt.title('Seasonal Booking Trends')
plt.xlabel('Month')
plt.ylabel('No. of Bookings')
plt.xticks(rotation=45)
plt.show()

# Create a box plot for the lead time distribution by cancellation status
sns.boxplot(x='is_canceled', y='lead_time', data=hotel_bookings_df)
plt.title('Lead Time Distribution by Cancellation Status')
plt.xlabel('Cancellation Status')
plt.ylabel('Lead Time')
plt.xticks([0, 1], ['Not Canceled', 'Canceled'])
plt.show()
