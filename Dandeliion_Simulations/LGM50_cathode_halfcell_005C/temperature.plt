#!/gnuplot

# Plots the total voltage (V) against time (s)

file_name = 'temperature.dat'

# Read the number of columns
stats file_name skip 1 nooutput
max_col = STATS_columns  # Maximum number of columns

# Resize the window
set terminal pngcairo background "#ffffff" enhanced font "Verdana, 10" fontscale 1 size 640, 400 linewidth 1 dashlength 1

set output 'temperature.png'

set grid
set key nobox opaque
set key inside vert right bottom height 1.1
set key title "{/:Bold Points:}" enhanced
set border linewidth 1.25
set tics scale 0.75
set style line 12 lc rgb 'grey' lt 1 lw 0.5
set grid xtics ytics ls 12
set title '{/:Bold Temperature}'
set xlabel 'time (s)'
set ylabel 'Temperature (K)'

# Average temperature is in the last column
plot file_name u 1:2 w l lt 6 lw 2 title "coldest", file_name u 1:(column(max_col-1)) w l lt 7 lw 2 title "hottest", file_name u 1:(column(max_col)) w l lt 1 lw 2 title "average"
