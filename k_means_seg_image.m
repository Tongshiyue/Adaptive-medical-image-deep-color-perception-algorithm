%clear;clc;close all;MRI-00822.jpg
%image_name ='./27.jpg' %'./datasets/胶质瘤/MRI-018.jpg'
%Apath = k_means_seg_image1(image_name)


function savepath = k_means_seg_image(image_path)
data=imread(image_path);
%imshow(data)
[m,n,c]=size(data);
if (c==3)
    data=rgb2gray(data)
end
[mu,pattern]=k_mean_Seg(data,4);
for x=1:m
    for y=1:n
        if pattern(x,y,1)==1
            data(x,y,1)=0;
            data(x,y,2)=0;
            data(x,y,3)=255;
        elseif pattern(x,y,1)==2
            data(x,y,1)=0;
            data(x,y,2)=255;
            data(x,y,3)=0;
        elseif pattern(x,y,1)==3
            data(x,y,1)=255;
            data(x,y,2)=0;
            data(x,y,3)=0;
%        elseif pattern(x,y,1)==4
%            data(x,y,1)=255;
%            data(x,y,2)=255;
%            data(x,y,3)=255;
        else
            data(x,y,1)=255;
            data(x,y,2)=255;
            data(x,y,3)=0;
        end
    end
end

%figure;
%imshow(data);  
path = strsplit(image_path,'.jpg')
savepath = char(strcat(path(1),'_seg.jpg'))
imwrite(data,savepath);
end
 

function [num,mask]=k_mean_Seg(src,k)
src=double(src);
img=src;       
src=src(:);     
mi=min(src);    
src=src-mi+1;   
L=length(src);
m=max(src)+1;
hist=zeros(1,m);
histc=zeros(1,m);
for i=1:L
  if(src(i)>0)
      hist(src(i))=hist(src(i))+1;
  end;
end
ind=find(hist);
hl=length(ind);
num=(1:k)*m/(k+1);
while(true)
  prenum=num;
  for i=1:hl
      c=abs(ind(i)-num);
      cc=find(c==min(c));
      histc(ind(i))=cc(1);
  end
  for i=1:k
      a=find(histc==i);
      num(i)=sum(a.*hist(a))/sum(hist(a));
  end
  if(num==prenum)
      break;
  end;
end
L=size(img);
mask=zeros(L);
for i=1:L(1),
for j=1:L(2),
  c=abs(img(i,j)-num);
  a=find(c==min(c)); 
  mask(i,j)=a(1);
end
end
num=num+mi-1;   
end
