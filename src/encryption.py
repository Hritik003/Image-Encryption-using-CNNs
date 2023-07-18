%----------------tent map sequence for the first image------------%
for i=1:s-1
    if x1(i)<(1/2)
        x1(i+1) = mu*x1(i);
       
    else
        x1(i+1)= mu*(1-x1(i));
    end
end

%-----------------------------3D chaos Generation----------------------------%
img1=reshape(new1,row,col);
img2=reshape(new2,row,col);
m=size(img1,1);
n=size(img1,2);
x(1)=0.2350;
y(1)=0.3500;
z(1)=0.7350;
a=0.0125;
b=0.0157;
c=3.7700;
for i=1:1500000
    x(i+1)=c*x(i)*(1-x(i))+b*((y(i))^2)*x(i)+a*z(i)^3;
    y(i+1)=c*y(i)*(1-y(i))+b*((z(i))^2)*y(i)+a*x(i)^3;
    z(i+1)=c*z(i)*(1-z(i))+b*((x(i))^2)*z(i)+a*y(i)^3;
end
x = mod(floor(x*100000),n);
y = mod(floor(y*100000),m);
z = mod(floor(z*100000),256);

%---------------------------------virtual planet xorring----------------%

enc_planet1 = reshape(bin_seq1,3,[]);

enc_planet2=enc_planet2.';

enc_planet_dec1 = bin2dec(enc_planet1).';

[temp2,idx_x2] = sort(x(1:size(enc_planet_dec2,2)));

for i=1:size(enc_planet_dec1,2)
   enc_planet_xor2(i) = bitxor(enc_planet_dec2(i),enc_planet_dec2(idx_x2(i)));
end  

enc_planet_bin1 = dec2bin(enc_planet_xor1,3);

enc_planet_bin2 = reshape(enc_planet_bin2.',1,[]);

enc_planet_bin1=enc_planet_bin1(1:end-t);

enc_8bit_bin2=reshape(enc_planet_bin2.',8,[]);

enc_8bit_bin1=enc_8bit_bin1.';

enc_8bit_dec2=bin2dec(enc_8bit_bin2);

enc_8bit_dec1 = enc_8bit_dec1.';

enc_final2=reshape(enc_8bit_dec2,m,[]);
