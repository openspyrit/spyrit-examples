function [] = verify_radon_matrix(Q_list, D_list,theta_list)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

%Vecteur des normes
erreur = zeros(1, numel(Q_list)*numel(D_list));
c = 1;

% Vecteur d'angle
theta = 0:180;
nb_angles = numel(theta);

%Pour toutes les tailles d'images et de pixels sur le récépteur
for q = 1:numel(Q_list)
    for d = 1:numel(D_list)
        
        % Importation de la matrice de Voxel
        objet = phantom('Modified Shepp-Logan',Q_list(q));
        %objet = imread('b.png');        
        %objet = rgb2gray(objet);
        % Réorganisation en vecteur
        alpha = zeros(Q_list(q)*Q_list(q),1);
        for i = 1:Q_list(q)
            alpha((i-1)*Q_list(q)+1:(i-1)*Q_list(q)+Q_list(q)) = objet(i,:);
        end
        
        % Importation de la matrice
        if numel(theta_list) == 181
            path = "measurement_matrix/Q" + string(Q_list(q)) + "_D" + string(D_list(d)) +".mat";
        else
            path = "measurement_matrix_missing_angles/Q" + string(Q_list(q)) + "_D" + string(D_list(d)) +".mat";
        end
        struct = load(path,'A');    
        A = struct.A;
        figure
        imagesc(A);
        colorbar
        title('Radon matrice A avec Q' + string(Q_list(q)) + "D" + string(D_list(d)))
        
        
        %Transformée de radon avec la matrice
        P = A*alpha;
        
        %Réorganisation de la matrice P pour l'affichage
        P_affichage = zeros(D_list(d),nb_angles);
        for angle = 1:nb_angles
            P_affichage(:,angle) = P(D_list(d)*(angle-1)+1:D_list(d)*(angle-1)+D_list(d));
        end
        
        %%Transformée de Radon avec MATLAB
        [R,xp] = radon(objet,theta);
        R_resize = zeros(D_list(d),nb_angles);
        
        xp = xp(1):(xp(end)-xp(1))/D_list(d):xp(end);
        
        for angle = 1:nb_angles
            if ismember(angle-1,theta_list)
                R_resize(:,angle) = resize(R(:,angle),D_list(d));
            end
        end
        
        %%Différence des matrices
        E = R_resize-P_affichage;
        erreur(c)=norm(E)/norm(R_resize);
        c = c+1;
        
        %%Affichage des résultats
        %TR avec MATLAB
        figure
        subplot(1,2,1)
        imshow(R_resize,[],'Xdata',theta,'Ydata',xp,'InitialMagnification','fit')
        xlabel('\theta (degrees)')
        ylabel('x''')
        title('Radon MATLAB Q' + string(Q_list(q)) + "D" + string(D_list(d)))
        colormap(gca,gray), colorbar

        %TR avec Matrice
        subplot(1,2,2)
        imshow(P_affichage,[],'Xdata',theta,'Ydata',xp,'InitialMagnification','fit')
        xlabel('\theta (degrees)')
        ylabel('x''')
        title('Radon matrice Q' + string(Q_list(q)) + "D" + string(D_list(d)))
        colormap(gca,gray), colorbar
        
    end
end

%Vérification qu'aucune erreur est faite
figure
hold on
for q = 1:numel(Q_list)
    plot(D_list,erreur((q-1)*numel(D_list)+1:(q-1)*numel(D_list)+numel(D_list)), '-o', 'DisplayName', "Q" + string(Q_list(q)));
end
grid on
title("Norme de la différence des transformés par Matlab et matrice en fonction du nombre de pixel sur le récepteur D")
xlabel('Nombre de pixel sur le détecteur D')
ylabel("||Pmatlab - Pmatrice||_2")
legend('show');



end

