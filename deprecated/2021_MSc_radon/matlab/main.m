% longueur / largeur matrice voxel
% Q = [64 128 256];
Q = [64];

% Resolution de la mesure
%D = [8, 16, 32, 64];
D = [64];

% Liste d'angles theta
theta = 0:180;

compute_radon_matrix(Q,D,theta);
%compress_radon_matrix(Q,D,theta);
compute_pinv_radon_matrix(Q,D,theta);
%verify_radon_matrix(Q,D,theta);
%compare_backprojection(Q,D,theta);