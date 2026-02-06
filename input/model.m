

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nx=845;
nz=502;

fp=fopen('acc_vp502_845.bin','rb');
vp0=fread(fp,[nz nx],'float');
fclose(fp);


vp1=ones(nz+200,nx)*vp0(1,1);
vp1(201:nz+200,:)=vp0;
vp=vp1;%(:,400:600);

figure;
imagesc(vp);


fp=fopen('acc_vp.bin','wb');
fwrite(fp,vp,'float');
fclose(fp);