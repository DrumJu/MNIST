//Model:
layer1: convolution layer  kernal_size=5   channels: 1=>6
layer2:MaxPooling layer kernal_size=2
layer3:convolution layer kernal_size=5 channels 6=>16
layer4 MaxPooling layer kernal_size=2
layer5: Flatten
layer6: linear layer 16*5*5 to 120
layer7: linear layer 120 to 84
layer8: linear layer 84 to 10//
Coder:Drum Ju
