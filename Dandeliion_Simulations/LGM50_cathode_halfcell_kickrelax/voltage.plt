#!/gnuplot

# Plots the total voltage (V) and the current against time (s)

file_voltage = 'voltage.dat'
file_current = 'current_total.dat'

# Redefine line styles
set style line 3 \
    linecolor rgb '#0000ff' \
    linetype 1 linewidth 2 \
    pointtype 7 pointsize 0.5
set style line 4 \
    linecolor rgb '#000000' \
    linetype 1 linewidth 1.5 \
    pointtype 7 pointsize 0.5

# Resize the window
set terminal pngcairo background "#ffffff" enhanced font "Verdana, 10" fontscale 1 size 640, 480 linewidth 1 dashlength 1

set output 'voltage.png'

set key nobox
set border linewidth 1.25
set tics scale 0.75
set style line 12 lc rgb 'grey' lt 1 lw 0.5
#set grid xtics ytics ls 12
set title '{/:Bold Total voltage and current vs time}'
set xlabel 'time (s)'
set ylabel 'Voltage (V)'

# Double y axis
set ytics nomirror autofreq tc ls 3
set y2tics nomirror autofreq tc lt 4
set y2label 'Current (A)'

plot file_current u 1:2 w l lt 4 notitle axis x1y2, file_voltage u 1:3 w l ls 3 notitle axis x1y1

# With dots
# plot file_name u 1:3 linestyle 4 notitle, \
#      file_name u 1:3 w l linestyle 3 notitle
