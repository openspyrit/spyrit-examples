function [] = verify_radon_backprojection(Q_list, D_list,theta_list)

%Pour toutes les tailles d'images et de pixels sur le récépteur


for q = 1:numel(Q_list)
    
    figure
    
    for t = 1:numel(theta_list)
        theta = 0:180/theta_list(t):180;
        nb_angles = numel(theta);
        
        for d = 1:numel(D_list)
            % Importation de la matrice de Voxel
            %objet = phantom('Modified Shepp-Logan',Q_list(q));
            
            
            objet2 = imread('test2.png');
            objet2 = rgb2gray(objet2);
            objet = double(objet2)/255;
            
            [~,xp] = radon(objet,theta);

            %objet = imread('b.png');        
            %objet = rgb2gray(objet);
            % Réorganisation en vecteur
            alpha = zeros(Q_list(q)*Q_list(q),1);
            for i = 1:Q_list(q)
                alpha((i-1)*Q_list(q)+1:(i-1)*Q_list(q)+Q_list(q)) = objet(i,:);
            end

            % Importation de la matrice
            path = "measurement_matrix/Q" + string(Q_list(q)) + "_D" + string(D_list(d)) +".mat";
            struct = load(path,'A');    
            A = struct.A;

            %Calculer A reduced avec moins d'angles  

            Areduced = zeros(D_list(d) * nb_angles , Q_list(q)*Q_list(q));
            for angles=1:(nb_angles)
                Areduced((angles-1) * D_list(d) + 1 : (angles-1) * D_list(d) + D_list(d), :) = A(floor(theta(angles)*D_list(d)+1) : floor(theta(angles) * D_list(d) + D_list(d)), :);
            end      
            A_pinv_reduced = pinv(Areduced);

            m = Areduced*alpha;
            f = A_pinv_reduced*m;

            %Réorganisation de la matrice P pour l'affichage
            m_affichage = zeros(D_list(d),nb_angles);
            for angle = 1:nb_angles
                m_affichage(:,angle) = m(D_list(d)*(angle-1)+1:D_list(d)*(angle-1)+D_list(d));
            end

            f_affichage = zeros(Q_list(q),Q_list(q));
            for i = 1:Q_list(q)
                f_affichage(i,:) = f((i-1)*Q_list(q)+1:(i-1)*Q_list(q)+Q_list(q));
            end        


            %Afficher les résultats
            subplot(numel(theta_list),numel(D_list),(t-1)*numel(D_list) + d);
            imshow(f_affichage);
            title("D = " + D_list(d) + " T = " + nb_angles)

        end
    end
end

end