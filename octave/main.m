% =========================================================================
% Main script to run and compare linear solvers on a given matrix problem.
%
% This file allows you to:
% 1. Specify a matrix from the SuiteSparse Matrix Collection (Matrix Market).
% 2. Automatically download and load the matrix.
% 3. Choose a solver to test (CG Normal, BiCGSTAB, GMRES, or LU).
% 4. Run the solver and display performance metrics.
%
% USAGE:
%   - Make sure all solver files (cg_normal.m, bicgstab.m, gmres.m,
%     lu_pivot.m, solve_lu.m) are in the same directory as this script.
%   - Set the `matrix_url` and `solver_to_use` variables below.
%   - Run the script from the Octave command line.
% =========================================================================

clear all;
clc;
close all;

% ---- 1. Configuration ----
% Choose the matrix to test by providing its URL from the Matrix Market.
matrix_url = "https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/bcspwr/bcspwr07.mtx.gz";

% Choose the solver to use: 'cg_normal', 'bicgstab', 'gmres', 'lu'
solver_to_use = 'cg_normal';

% ---- 2. Matrix Loading ----
% This section handles downloading, unzipping, and loading the matrix.
% It requires the 'io' package for reading Matrix Market files.
% ---- 2. Environment Setup ----
% Ensure the 'io' package for reading Matrix Market files is available.
try
    pkg load io;
    % For robustness, manually add the package's path. This solves issues
    % where Octave fails to find functions in locally installed packages.
    io_pkg_info = pkg('list', 'io');
    if ~isempty(io_pkg_info)
        addpath(fullfile(io_pkg_info{1}.dir, 'inst'));
    end
catch
    error("The 'io' package is required. Please install it by running 'pkg install -forge io' in the Octave console, then 'pkg load io'.");
end

[~, matrix_name, ext] = fileparts(matrix_url);
if strcmp(ext, ".gz")
    [~, matrix_name, ~] = fileparts(matrix_name);
end
local_mtx_file = [matrix_name, ".mtx"];
local_gz_file = [local_mtx_file, ".gz"];

% Download and unzip if the matrix file doesn't exist locally
if ~exist(local_mtx_file, 'file')
    fprintf('Matrix file not found. Downloading %s...\n', matrix_name);
    try
        urlwrite(matrix_url, local_gz_file);
        gunzip(local_gz_file);
        delete(local_gz_file); % Clean up the gz file
        fprintf('Download and unzip complete.\n');
    catch
        error('Failed to download or unzip the matrix file from the URL.');
    end
end

fprintf('Loading matrix %s...\n', matrix_name);
A = mmread(local_mtx_file);
fprintf('Matrix loaded successfully.\n\n');

% ---- 3. Problem Setup ----
% Create a linear system Ax = b where the true solution is known.
n = size(A, 2);
x_true = ones(n, 1); % Define a known "true" solution
b = A * x_true;     % Calculate the corresponding right-hand side b

% Set common parameters for iterative solvers
x0 = zeros(n, 1);   % Initial guess
tol = 1e-8;         % Convergence tolerance
max_iter = 2 * n;   % Maximum number of iterations

% ---- 4. Solver Execution ----
fprintf('--- Running Solver: %s ---\n', upper(solver_to_use));
tic; % Start timer

switch solver_to_use
    case 'cg_normal'
        [x, rel_res, iters] = cg_normal(A, b, x0, max_iter, tol);

    case 'bicgstab'
        [x, rel_res, iters] = bicgstab(A, b, x0, max_iter, tol);

    case 'gmres'
        m = 100; % Restart parameter for GMRES
        % Note: GMRES function needs to be adapted to return iters and rel_res
        % For now, we call the existing one.
        x = gmres(A, b, x0, m);
        iters = m; % For this simple case, iters is the restart param
        rel_res = norm(b - A*x) / norm(b);

    case 'lu'
        [L, U, P] = lu_pivot(A);
        x = solve_lu(L, U, P, b);
        iters = 'N/A'; % Direct solver, no iterations
        rel_res = norm(b - A*x) / norm(b);

    otherwise
        error('Unknown solver specified. Please choose from "cg_normal", "bicgstab", "gmres", or "lu".');
end

elapsed_time = toc; % Stop timer

% ---- 5. Results Display ----
solution_error = norm(x - x_true);

fprintf('\n--- Results for %s ---\n', matrix_name);
fprintf('Matrix Dimensions: %d x %d\n', size(A, 1), size(A, 2));
fprintf('Solver Used:       %s\n', upper(solver_to_use));
fprintf('----------------------------------------\n');
fprintf('Execution Time:    %.4f seconds\n', elapsed_time);
if isnumeric(iters)
    fprintf('Iterations:        %d\n', iters);
end
fprintf('Final Rel. Res.:   %e\n', rel_res);
fprintf('Solution Error:    %e (||x - x_true||)\n', solution_error);
fprintf('========================================\n');

