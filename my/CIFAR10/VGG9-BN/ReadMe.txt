BWN:
withnoxnorgrad:89.5%
withxnorgrad:89.66%

#vectorize all filters in one layer 
withnoxnorgrad & shrink angle:89.30% : beta 0.02
withnoxnorgrad & shrink angle:89.68% : beta 0.01
withnoxnorgrad & shrink angle:89.91% : beta 0.001
                              89.84%
withnoxnorgrad & shrink angle:89.75% : beta 0.0001
   			      
#vectorize one filter in one layer 
                              89.51% beta 0.001

VGG9-BN on CIFAR-10 dataset
FWN: 91.19%
TWN: 90.31%
BWN: 89.5%
BWN_INQ_Relax:last89.89%  best89.93%  coe:0.0001
BWN_INQ_Relax:last89.96%  best90.2%   coe:0.001
BWN_INQ_Relax:last90.03%  best90.14%  coe:0.01
BWN_INQ_Relax:last88.26%  best88.83%  coe:0.05


