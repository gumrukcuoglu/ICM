#!/gnuplot

# Plots concentration and potential in liquid, concentration in solid in Anode and Cathode,
# and total voltage vs capacity

# Resize the window
set terminal pngcairo background "#ffffff" enhanced font "Verdana, 10" fontscale 1 size 840, 600 linewidth 1 dashlength 1
set size 1, 1
set origin 0, 0

set output 'multiplot.png'

# Redefine line styles
set style line 1 \
    linecolor rgb '#ff0000' \
    linetype 1 linewidth 3 \
    pointtype 7 pointsize 0.5
set style line 2 \
    linecolor rgb '#0000ff' \
    linetype 1 linewidth 3 \
    pointtype 7 pointsize 0.5
set style line 3 \
    linecolor rgb '#0000ff' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 0.5
set style line 4 \
    linecolor rgb '#000000' \
    linetype 1 linewidth 1.5 \
    pointtype 7 pointsize 0.5

# Read the number of columns
stats 'concentration_liquid.dat' skip 1 nooutput
max_col = STATS_columns  # Maximum number of columns

# Set up multiplot canvas
set multiplot layout 2, 2 rowsfirst downwards scale 1.0, 1.0  # or margins 0.1, 0.9, 0.1, 0.9 spacing 0.1

# Tweak some of the global parameters
set border linewidth 1.25
set tics scale 0.75
set grid
#set key nobox opaque
#set key outside vert right center height 1.1
#set key title "{/:Bold Key:}" enhanced
set key off
set style line 12 lc rgb 'grey' lt 1 lw 0.5
set grid xtics ytics ls 12

# Concentration in liquid plot
file_name = 'concentration_liquid.dat'
set title '{/:Bold Concentration in liquid}'
set xlabel '{/:Normal x ({/Symbol m}m)}'
set ylabel '{/:Normal Concentration (mol/dm^3)}'
set xtics font ':Normal'
set ytics font ':Normal'
plot file_name u 1:($2*0.001) w l ls 2 title columnheader, \
     for [i=3:max_col-1] file_name u 1:(column(i)*0.001) w l ls 4 title columnheader, \
     file_name u 1:(column(max_col)*0.001) w l ls 1 title columnheader(max_col)

# Potential in liquid plot
file_name = 'potential_liquid.dat'
set title '{/:Bold Potential in liquid}'
set xlabel '{/:Normal x ({/Symbol m}m)}'
set ylabel '{/:Normal Potential (mV)}'
set xtics font ':Normal'
set ytics font ':Normal'
plot file_name u 1:($2*1000) w l ls 2 title columnheader, \
     for [i=3:max_col-1] file_name u 1:(column(i)*1000) w l ls 4 title columnheader, \
     file_name u 1:(column(max_col)*1000) w l ls 1 title columnheader(max_col)

# Concentration in solid in Anode plot
file_name = 'cs_solid_anode_xrel=0.500000.dat'
set title '{/:Bold Concentration in solid in the middle of Anode}'
set xlabel '{/:Normal r ({/Symbol m}m)}'
set ylabel '{/:Normal Concentration (mol/dm^3)}'
set xtics font ':Normal'
set ytics font ':Normal'
plot file_name u 1:($2*0.001) w l ls 2 title columnheader, \
     for [i=3:max_col-1] file_name u 1:(column(i)*0.001) w l ls 4 title columnheader, \
     file_name u 1:(column(max_col)*0.001) w l ls 1 title columnheader(max_col)

# Concentration in solid in Cathode plot
file_name = 'cs_solid_cathode_xrel=0.500000.dat'
set title '{/:Bold Concentration in solid in the middle of Cathode}'
set xlabel '{/:Normal r ({/Symbol m}m)}'
set ylabel '{/:Normal Concentration (mol/dm^3)}'
set xtics font ':Normal'
set ytics font ':Normal'
plot file_name u 1:($2*0.001) w l ls 2 title columnheader, \
     for [i=3:max_col-1] file_name u 1:(column(i)*0.001) w l ls 4 title columnheader, \
     file_name u 1:(column(max_col)*0.001) w l ls 1 title columnheader(max_col)

unset multiplot
