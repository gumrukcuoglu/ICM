#!/gnuplot

# Plots concentration in solid across the electrode

# Resize the window
set terminal pngcairo background "#ffffff" enhanced font "Verdana, 10" fontscale 1 size 640, 400 linewidth 1 dashlength 1

set output 'potential_solid.png'

set grid
set key box opaque
set key ins vert cent right
set key width 1
set border linewidth 1.25
set tics scale 0.75
set style line 12 lc rgb 'grey' lt 1 lw 0.5
set grid xtics ytics ls 12

file_name = 'potential_solid.dat'
set title '{/:Bold Potential in solid}'
set xlabel 'x ({/Symbol m}m)'
set ylabel 'Potential (V)'

# Redefine line styles
set style line 1 \
    linecolor rgb '#ff0000' \
    linetype 1 linewidth 3 \
    pointtype 7 pointsize 0.75
set style line 2 \
    linecolor rgb '#009900' \
    linetype 1 linewidth 3 \
    pointtype 7 pointsize 0.75
set style line 4 \
    linecolor rgb '#000000' \
    linetype 1 linewidth 1.5 \
    pointtype 7 pointsize 0.5

# Read the number of columns
stats 'potential_solid.dat' skip 1 nooutput
max_col = STATS_columns  # Maximum number of columns

# Potential in solid plot
plot file_name u 1:($2) ls 2 title columnheader, \
     for [i=3:max_col-1] file_name u 1:(column(i)) ls 4 title columnheader, \
     file_name u 1:(column(max_col)) ls 1 title columnheader(max_col)
