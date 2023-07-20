%----------------------------DECRYPTION--------------------------------%

dec_heatkey = bitxor(encrypted_image1,keyspace);

decy_dec = reshape(dec_heatkey,1,[]);

decy_dec = decy_dec.';

decy_bin = dec2bin(decy_dec);

decy_bin = decy_bin.';

decy_planet_seq = reshape(decy_bin,1,[]);

t=0;
while mod(size(decy_planet_seq,2),3)~=0
    decy_planet_seq = strcat(decy_planet_seq,'0');
    t=t+1;
end

decy_planet_seq = reshape(decy_planet_seq,[],1);

decy_planet_seq = decy_planet_seq.';

decy_planet_3bit = reshape(decy_planet_seq,3,[]).';

decy_planet_dec = bin2dec(decy_planet_3bit).';

 for i=1:size(enc_planet_dec1,2)
   decy_planet_dec1(i) = bitxor(decy_planet_dec(i),enc_planet_dec1(idx_x1(i)));
 end  

decy_planet1 = (dec2bin(decy_planet_dec1.')).';

decy_bin_seq = reshape(decy_planet1,1,[]);

decy_bin_seq=decy_bin_seq(1:end-t);

decy_z1=decy_bin_seq-'0';

decy_y1 = reshape(decy_z1,8,[]).';

decy_y1 = char(decy_y1+'0');

decy_dec_img = bin2dec(decy_y1);

decy_newimg3 = decy_dec_img.';

decy_newimg3 = reshape(decy_newimg3,n,m).';

         %---------col reverse rotation----------%

for i=1:n

    Y1 = decy_newimg3(:,i);
    % Y2 = new_img11(:,i);

    shift1 = n-(mod(2*i+chaos2_img1(i),n)+1);
    % shift2 = n-(mod(2*i+chaos2_img2(i),n)+1);

    Y1= circshift(Y1,shift1);
    % Y2= circshift(Y2,shift2);

    decy_newimg1(:,i)=Y1;
    % new_img22(:,i)=Y2;

end

        %----------row reverse rotation----------%

for i=1:m
    Y1 = decy_newimg1(i,:);
    % Y2 = img2(i,:);

    shift1 = m-(mod(i+chaos1_img1(i),n)+1);
    % shift2 = mod(i+chaos1_img2(i),n)+1;

    Y1= circshift(Y1,shift1);
    % Y2= circshift(Y2,shift2);

    decy_new1(i,:)=Y1;
    % new_img11(i,:)=Y2;
end

decy_new1 = reshape(decy_new1,1,s);

for i=1:s
    decy_img1(i) = bitxor(w1(i),decy_new1(i));
    % new2(i) = bitxor(w2(i),img2(i));
end


decrypted_image = reshape(decy_img1,n,m).';
