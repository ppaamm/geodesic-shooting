import numpy as np

def ellipse (Pk, avg):
    covariance = Pk
    [eigenval, eigenvec] = np.linalg.eig(covariance)
    return r_ellipse


function [ r_ellipse ] = ellipse( Pk, avg )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    % Calculate the eigenvectors and eigenvalues
covariance = Pk;
[eigenvec, eigenval ] = eig(covariance);
% eigenvec
% Get the index of the largest eigenvector
[largest_eigenvec_ind_c, r_ell] = find(eigenval == max(max(eigenval)));
largest_eigenvec = eigenvec(:, largest_eigenvec_ind_c);

% Get the largest eigenvalue
largest_eigenval = max(max(eigenval));

% Get the smallest eigenvector and eigenvalue
if(largest_eigenvec_ind_c == 1)
    smallest_eigenval = max(eigenval(:,2));
    smallest_eigenvec = eigenvec(:,2);
else
    smallest_eigenval = max(eigenval(:,1));
    smallest_eigenvec = eigenvec(1,:);
end

% Calculate the angle between the x-axis and the largest eigenvector
angle = atan2(largest_eigenvec(2), largest_eigenvec(1));

% This angle is between -pi and pi. Let's shift it such that the angle is
% between 0 and 2pi
if(angle < 0)
    angle = angle + 2*pi;
end

% Get the coordinates of the data mean avg = [Xch(1,k),Xch(2,k)];
%avg = mu;

% Get the 99% confidence interval error ellipse
chisquare_val = 0.089;%sqrt(9.21);
theta_grid = linspace(0,2*pi);
phi = angle;
%phi

X0=avg(1);
Y0=avg(2);

a_ellipse=chisquare_val*sqrt(largest_eigenval);
b_ellipse=chisquare_val*sqrt(smallest_eigenval);
% a_ellipse
% b_ellipse

% the ellipse in x and y coordinates
ellipse_x_r  = a_ellipse*cos( theta_grid );
ellipse_y_r  = b_ellipse*sin( theta_grid );

%Define a rotation matrix
R_ellipse = [ cos(phi) sin(phi); -sin(phi) cos(phi) ];
%R_ellipse

%let's rotate the ellipse to some angle phi
r_ellipse = [ellipse_x_r;ellipse_y_r]' * R_ellipse;

%let's center the ellipse on the mean
r_ellipse = [r_ellipse(:,1)+X0 r_ellipse(:,2)+Y0];

end

