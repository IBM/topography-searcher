# Reconverge molecular landscape

In this example we show how to reconverge a landscape of stationary points within a new potential. We first load the minima and transition states
of ethanol computed using a low level of density functional theory. From these stationary points we generate the landscape for a higher
level of density functional theory by reconverging all within the new potential. We are able to produce the landscape at the higher level of theory with a much reduced cost. This reconvergence of conformations could be applied to any set of conformations, for example MD snapshots within a force field, provided the structures are written into the required input format. The script can generally be applied to any molecule by changing the xyz file input, but be aware the computational cost will increase quickly with molecule size. **This script is slow, and will take several hours to complete on 8 threads**.
