% Hace laberintos

clear all;
dimx=401;canal=25;
aa=zeros(dimx);
for i=canal+1:canal:dimx-canal
    for j=1:canal:dimx-canal
        aa(i-1:i+1,j:j+canal)=round(0.25+rand);
    end
end
aa(1:dimx,1:3)=1;aa(1:dimx,dimx-3:dimx)=1;
aa(1:3,1:dimx)=1;aa(dimx-3:dimx,1:dimx-canal)=1;
figure(2);pcolor(aa);shading flat;axis equal;
save laberinto.dat aa -ASCII -DOUBLE

% Hace laberintos 2

clear all;
dimx=301;canal=30;extra=1;peso=-0.30;
aa=zeros(dimx);
for i=0:canal:dimx-canal
    for j=0:canal:dimx-canal
        if round(peso+rand)==1 
            aa(i+1:i+3,j+1:j+canal+extra)=1;
        end
        if round(peso+rand)==1 
            aa(i+1:i+canal+extra,j+1:j+3)=1;
        end
    end
end
aa(1:dimx,1:3)=1;aa(1:dimx,dimx-3:dimx)=1;
aa(1:3,1:dimx)=1;aa(dimx-3:dimx,1:dimx-canal)=1;
figure(2);pcolor(aa);shading flat;axis equal;
save laberinto_301x301_complicado.dat aa -ASCII -DOUBLE

% Hace laberintos

clear all;
dimx=1003;canal=50;extra=0;peso=-0.25;
aa=zeros(dimx);
for i=1:canal:dimx-canal
    for j=1:canal:dimx-canal
        contador=6;
        while contador>2
        numerosi(i,j)=round(peso+rand);
        numerosd(i,j)=round(peso+rand);
        numerosr(i,j)=round(peso+rand);
        numerosb(i,j)=round(peso+rand);
        contador=numerosi(i,j)+numerosd(i,j)+numerosr(i,j)+numerosb(i,j)
        end
    end
end
 
for i=1:canal:dimx-canal
    for j=1:canal:dimx-canal
        if numerosi(i,j)==1
           aa(i:i+2,j:j+canal+extra)=1;
        end
        if numerosb(i,j)==1
           aa(i:i+canal+extra,j:j+2)=1;
        end
        if numerosd(i,j)==1
           aa(i+canal:i+canal+2,j:j+canal+extra)=1;
        end
        if numerosr(i,j)==1
           aa(i:i+canal+extra,j+canal:j+canal+2)=1;
        end
    end
end

aa(1:dimx,1:3)=1;aa(1:dimx,dimx-3:dimx)=1;
aa(1:3,1:dimx)=1;aa(dimx-3:dimx,1:dimx)=1;
figure(2);pcolor(aa);shading flat;axis equal;

save laberinto_1003x1003_10.dat aa -ASCII -DOUBLE


% Hace laberintos


clear all;
dimx=803;canal=40;extra=1;peso=0.50;
aa=zeros(dimx);
for i=0:2*canal:dimx-2*canal
    for j=0:2*canal:dimx-2*canal
        if round(peso+2*rand)==1 
            aa(i+1:i+3,j+1:j+2*canal+extra)=1;
            aa(i+1+canal:i+3+canal,j+1:j+2*canal+extra)=1;
        elseif round(peso+2*rand)==2 
            aa(i+1:i+2*canal+extra,j+1:j+3)=1;
            aa(i+1:i+2*canal+extra,j+1+canal:j+3+canal)=1;
        end
    end
end
aa(1:dimx,1:3)=1;aa(1:dimx,dimx-3:dimx)=1;
aa(1:3,1:dimx)=1;aa(dimx-3:dimx,1:dimx)=1;
figure(2);pcolor(aa);shading flat;axis equal;

save laberinto2_601x601.dat aa -ASCII -DOUBLE
