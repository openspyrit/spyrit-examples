function [output] = resize(input, new_size)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
output = zeros(new_size, 1);
old_size = numel(input);
rapport = old_size / new_size;

for i = 1:new_size
   sums = 0;
   beg_index = floor((i-1)*rapport + 1);
   end_index = floor(i*rapport);
   sums = sum(input(beg_index : end_index));
   sums = sums + input(beg_index) * (beg_index - (i-1)*rapport) + input(end_index) * (i*rapport - end_index);
   output(i) = sums / rapport;
end

end

