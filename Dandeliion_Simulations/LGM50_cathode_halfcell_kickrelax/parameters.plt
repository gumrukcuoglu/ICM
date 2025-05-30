#!/gnuplot

# Plots parameters: Ueq and Ds in Anode and Cathode

# Resize the window
set terminal pngcairo background "#ffffff" enhanced font "Verdana, 10" fontscale 1 size 840, 600 linewidth 1 dashlength 1
set size 1, 1
set origin 0, 0

set output 'parameters.png'

# Set up multiplot canvas
set multiplot layout 2, 2 rowsfirst downwards scale 1.0, 1.0  # or margins 0.1, 0.9, 0.1, 0.9 spacing 0.1

# Tweak some of the global parameters
set border linewidth 1.25
set tics scale 0.75
set grid
set key off
set style line 12 lc rgb 'grey' lt 1 lw 0.5
set grid xtics ytics ls 12

file_name = 'parameters.dat'

# Ueq in Anode
set title '{/:Bold U_{eq} in Anode}'
set xlabel '{/:Normal SOC}'
set ylabel '{/:Normal U_{eq} (V)}'
set xtics font ':Normal'
set ytics font ':Normal'
plot file_name u 1:2 w l lw 2 lt 1

# Ueq in Cathode
set title '{/:Bold U_{eq} in Cathode}'
set xlabel '{/:Normal SOC}'
set ylabel '{/:Normal U_{eq} (V)}'
set xtics font ':Normal'
set ytics font ':Normal'
plot file_name u 1:3 w l lw 2 lt 1

# Ds in Anode
set title '{/:Bold Diffusivity in Anode}'
set xlabel '{/:Normal SOC}'
set ylabel '{/:Normal D_s (cm^2/s)}'
set format y '10^{%L}'
set logscale y
plot file_name u 1:($4*1e4) w l lw 2 lt 4

# Ueq in Cathode
set title '{/:Bold Diffusivity in Cathode}'
set xlabel '{/:Normal SOC}'
set ylabel '{/:Normal D_s (cm^2/s)}'
set format y '10^{%L}'
set logscale y
plot file_name u 1:($5*1e4) w l lw 2 lt 4

unset multiplot
