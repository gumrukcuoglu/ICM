#!/gnuplot

# Plots parameters: kappa, D in Electrolyte, Ueq and Ds in Anode and Cathode

set samples 1000

# Resize the window
set terminal pngcairo background "#ffffff" enhanced font "Verdana, 10" fontscale 1 size 400, 280 linewidth 1 dashlength 1

# Tweak some of the global parameters
set border linewidth 1.25
set tics scale 0.75
set grid
set key off
set style line 12 lc rgb 'grey' lt 1 lw 0.5
set grid xtics ytics ls 12

# Conductivity
set title "Conductivity in the electrolyte"
set xlabel "Li concentration (mol/L)"
set ylabel "{/Symbol k} (S/m)"
