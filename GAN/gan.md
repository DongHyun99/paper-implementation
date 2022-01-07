# GAN

latent vector의 dimention은 10, 생성 이미지의 크기는 24x24x3으로 구현했다.  

GAN은 Convolution하는 구간이 없고 오로지 FC Layer와 Activation Function만으로 이루어져있다.  

## Generator  

Generator는 latent vector의 차원 크기 만큼의 Random Noise를 생성하여 FC Layer와 leaky ReLU, Batch Normalization을 반복하는구조를 가진다. 마지막에는 생성하려는 이미지의 크기만큼 크기를 늘려 tanh를 통과한뒤 이미지 크기대로 reshape 해주는 구조다.  

10 -> 256 -> 512 -> 1024 -> 1728 -> 24x24x3로 차원수가 변화한다.