# Animation Fun
# Try to see what can be done with animation


#  Need both of these libraries as well as ImageMagick installed on computer.
#  Make sure ImageMagick is on your computer's path or set the path directly
#  in the call to gganimate
library(tidyverse)
library(gganimate)


# Read in a local file
read_data <- function(filename) {
  df <- as.data.frame(suppressMessages(read_csv(filename)))
  df$day_of_week <- as.factor(df$day_of_week)
  df$week_day_end <- as.factor(df$week_day_end)
  df$sun_rise_set <- as.factor(df$sun_rise_set)
  return(df)
}

# Example dataframe
energy_df <- read_data('data/f-SRP_weather.csv')

# Create a day for grouping
energy_df <- mutate(energy_df, day = lubridate::ymd(as.Date(timestamp)))

# Select the relevant columns for plotting
temp_energy <- select(energy_df, num_time, cleaned_energy, biz_day, temp, day)

# Convert the dataframe to long format
temp_energy <- gather(temp_energy, key = 'variable', value = 'value', cleaned_energy, temp) %>%
  arrange(day, num_time)


# The call to ggplot is the same as normal except aes(frame) is added. The frame 
# is the variable over which you want to iterate. In this case, I am 
# iterating over the days.
# 
# This plot is energy and temperature. I created a secondary axis for temperature
# but left the scale the exact same as for energy because the two
# were very simliar with this data
p <- ggplot(temp_energy[1:100000, ], 
            aes(x = num_time, y = value, col = variable, frame = day)) + 
  geom_line() + xlab('Time of Day (hrs)') + 
  scale_y_continuous(sec.axis = sec_axis(~.*1, name = "Temp (C)")) + 
  labs(col = 'legend') + ylab('Energy (kWh)') + 
  scale_x_continuous(lim=c(0, 24), breaks = seq(0, 24, 4)) + 
  scale_color_manual(values = c('black', 'red')) + 
  theme(axis.text.y.right = element_text(color = 'red'),
        axis.text.y = element_text(color = 'black'))

# I had to specify the path to ImageMagick on my computer using ani.options
# p is the plot, the second argument is the file location to save the output (
# different output types can be used), the interval is the time in seconds between frames,
# ani.options might have to be set to the location of ImageMagick convert
# on your computer, title_frame sets the title to the name of each frame, saver
# is the type of saver to use (can change for different file types)
gganimate(p, 'energy_temp.mp4', interval = 0.5, 
          ani.options(convert = 'C:/ImageMagick-7.0.7-Q16/convert.exe'), 
          title_frame = TRUE, saver = 'mp4')
