function [SAM_index,SAM_map] = SAM(I1,I2)

[M,N,~] = size(I2);
prod_scal = dot(I1,I2,3);
norm_orig = dot(I1,I1,3);
norm_fusa = dot(I2,I2,3);
prod_norm = sqrt(norm_orig.*norm_fusa);
prod_map = prod_norm;
prod_map(prod_map ==0)=eps;
SAM_map = acos(prod_scal./prod_map);
prod_scal = reshape(prod_scal,M*N,1);
prod_norm = reshape(prod_norm,M*N,1);
z=find(prod_norm ==0);
prod_scal(z) = [];prod_norm(z)=[];
angolo = sum(sum(acos(prod_scal./prod_norm)))/(size(prod_norm,1));
SAM_index = real(angolo)*180/pi;

end
