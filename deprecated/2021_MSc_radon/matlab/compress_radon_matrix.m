function [] = compress_radon_matrix(Q_list, D_list, theta_list)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

mkdir 'measurement_matrix/sparse'
mkdir 'measurement_matrix_missing_angles/sparse'

for q = 1:numel(Q_list)
    for d = 1:numel(D_list)
        if numel(theta_list) == 181
            path1 = "measurement_matrix/Q" + string(Q_list(q)) + "_D" + string(D_list(d)) +".mat";
            path2 = "measurement_matrix/sparse/sparse_Q" + string(Q_list(q)) + "_D" + string(D_list(d)) +".mat";
        else
            path1 = "measurement_matrix_missing_angles/Q" + string(Q_list(q)) + "_D" + string(D_list(d)) +".mat";
            path2 = "measurement_matrix_missing_angles/sparse/sparse_Q" + string(Q_list(q)) + "_D" + string(D_list(d)) +".mat";
        end
        struct = load(path1,'A');
        A_sparse = sparse(struct.A);
        save(path2,'A_sparse','-v7.3')
    end
end

end

