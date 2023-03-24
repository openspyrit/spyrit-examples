function [] = compute_radon_matrix(Q_list, D_list,theta_list)
%compute_radon_matrix this function takes a Q_list vector containing size
%of image of Q*Q pixels. D_list vector containing number of pixels on
%the sensor. theta_list is a vector of measurement angles. The result is
%Q_list * D_list .mat files where the radon matrix is saved for each
%combinaison of Q and D. If theta_list is [0:180], the radon matrix 
%contains every angles. In the other cases this will creates holes in
%sinograms. 

%Angles vector
nb_angles = numel(theta_list);
total_angles = 181;

mkdir 'measurement_matrix'
mkdir 'measurement_matrix_missing_angles'

%Début de la boucle d'export des matrices A

    for q = 1:numel(Q_list)
        for d = 1:numel(D_list)

            % Matrice d'acquisition
            A = zeros(D_list(d)*total_angles, Q_list(q)*Q_list(q));

            % Calcul de la Matrice

            for angle = 1:nb_angles
                for i = 1:Q_list(q)
                   for j = 1:Q_list(q)
                      maquette = zeros(Q_list(q),Q_list(q));      
                      maquette(i,j) = 1;
                      P_theta_temp = radon(maquette,theta_list(angle));
                      P_theta_resize = resize(P_theta_temp, D_list(d));          
                      for index = 1:D_list(d)
                         A(D_list(d)*(theta_list(angle)) + index,(i-1)*Q_list(q)+j) = P_theta_resize(index);
                      end
                   end
                end
            end
            if numel(theta_list) == 181
                path = "measurement_matrix/Q" + string(Q_list(q)) + "_D" + string(D_list(d)) +".mat";
            else
                path = "measurement_matrix_missing_angles/Q" + string(Q_list(q)) + "_D" + string(D_list(d)) +".mat";
            end
            save(path,'A','-v7.3')
        end
    end
end

