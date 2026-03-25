%%%%%%%%%%%%%%%%%%%%%%%%%%
%RESOLUCIÓN DE LABERINTOS%
%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1.-Creación de laberintos.


% El programa generará un laberinto aleatorio en base a tres parametros.
% L=Longitud de la pared del laberinto.
% l=Tamaño del canal (anchura de pasillo).
% n=Numero para el escalón aleatorio (se crea pared o no)

%Un objetivo interesante sería encontrar el numero crítico nc (proceso
%percolativo) que asegura que exista camino.


close all; clc;clear; %Para mi salud mental
L=203;
l=25;


n=0.67;%Valor critico aproximado.

muro=3;%Tamaño de muro
A=zeros(L,L);%Matriz del laberinto

%Paredes

%Izquierda
for i=1:1:L
    for j=1:1:muro
        A(i,j)=1;
    end
end

%Superior
for i=L-muro:1:L
    for j=1:1:L
        A(i,j)=1;
    end
end

%Derecha
for i=1:1:L
    for j=L-muro:1:L
        A(i,j)=1;
    end
end

%Inferior
for i=1:1:muro
    for j=1:1:L
        A(i,j)=1;
    end
end



%Laberinto

%Recorro la matriz en pasos de tamaño l y coloco paredes derecha y superior
%al azar

for i=1:l:L-l
    for j=1:l:L-l
        
        n1=rand();
        n2=rand();
        
        %Derecha
        if n1>n
            for p=i:1:i+l
               A(p,j+l)=1; 
            end
        end
        
        %Superior
        if n2>n
            for p=j:1:j+l
               A(i+l,p)=1; 
            end
        end
    end
end

pcolor(A);shading flat %ODIO pcolor pero queda mejor que imagesc
% imagesc(A);grid


