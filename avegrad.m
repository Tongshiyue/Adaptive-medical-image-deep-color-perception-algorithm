function AVEGRAD=avegrad(img)

img=double(img);
% img=rgb2gray(img); 
% img=double(img);
[M,N,C]=size(img);
gradval=zeros(M,N,C); %%% save the gradient value of single pixel
diffX=zeros(M,N,C);    %%% save the differential value of X orient
diffY=zeros(M,N,C);    %%% save the differential value of Y orient

tempX=zeros(M,N,C);
tempY=zeros(M,N,C);
tempX(1:M,1:(N-1))=img(1:M,2:N);
tempY(1:(M-1),1:N)=img(2:M,1:N);

diffX=img-tempX;
diffY=img-tempY;
diffX(1:M,N)=0;       %%% the boundery set to 0
diffY(M,1:N)=0;
diffX=diffX.*diffX;
diffY=diffY.*diffY;
AVEGRAD=sum(sum(diffX+diffY));
AVEGRAD=sqrt(AVEGRAD);
AVEGRAD=AVEGRAD/((M-1)*(N-1));
AVEGRAD=(AVEGRAD(1,1,1)+AVEGRAD(1,1,2)+AVEGRAD(1,1,3))/3;
end    