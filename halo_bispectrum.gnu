set terminal pdfcairo enhanced color font ',14'
#set output "cutoff_-12.pdf"
set output "dustopt_halo.pdf"
set border linewidth 1
#set size square (1,1)

set format y "10^{%L}"
#set format y "%2.0t{/Symbol \327}10^{%L}"
#set format y "{%T}"
set title "Bispectrum of Optimistic Dust Model in Halos"
#set title "Magnification for M_{lens} = 10^{-12}M_{solar} r=0.001"
#set title "iic"
set ylabel r"{/Symbol D}^2_{eq}(k)"
set xlabel r"k [h Mpc^{-1}]" 

set logscale yx
set yrange [10**(-2):10**(5)]
#set xtics
#set style line 2 lt 1 lc rgb "dark-green" lw 4 pt 6 ps 1 

set key top left
plot "dustopt_vs_halo_vs_perturb_equilateral.txt" u 1:($2/sqrt(6119846.547113527)) w l lt rgb "blue" title "PPP_{dust}", \
"dustopt_vs_halo_vs_perturb_equilateral.txt" u 1:($3/sqrt(6119846.547113527)) w l lt rgb "red" title "Phh_{dust}", \
"dustopt_vs_halo_vs_perturb_equilateral.txt" u 1:($4/sqrt(6119846.547113527)) w l lt rgb "violet" title "hhh_{dust}", \
"dustopt_vs_halo_vs_perturb_equilateral.txt" u 1:5 w l lt rgb "green" title "{dust}", \
"dustopt_vs_halo_vs_perturb_equilateral.txt" u 1:6 w l lt rgb "brown" title "halo", \
"dustopt_vs_halo_vs_perturb_equilateral.txt" u 1:7 w l lt rgb "black" title "PT"
