%! Solves the linear system Ax = b using the GMRES method.
%!
%! This version is updated to include a convergence check within the loop
%! and to return the final iteration count and relative residual. It uses
%! Givens rotations to solve the least-squares subproblem efficiently.
%!
%! @param A        The coefficient matrix `A`.
%! @param b        The right-hand side vector `b`.
%! @param x0       The initial guess for the solution `x`.
%! @param m        The restart parameter (max size of the Krylov subspace).
%! @param tol      The convergence tolerance.
%!
%! @return x       The approximate solution vector `x`.
%! @return rel_res The final relative residual norm.
%! @return iters   The number of iterations performed.
function [x, rel_res, iters] = gmres(A, b, x0, m, tol)
  n = size(A, 1);
  x = x0;
  iters = 0;

  % ---- Initial Setup ----
  r = b - A * x;
  norm_b = norm(b);
  if norm_b == 0.0, norm_b = 1.0; end % Handle zero RHS

  beta = norm(r);
  rel_res = beta / norm_b;

  if rel_res < tol
    return; % Already converged
  end

  % V: Orthonormal basis for the Krylov subspace
  V = zeros(n, m + 1);
  % H: Upper-Hessenberg matrix
  H = zeros(m + 1, m);
  V(:, 1) = r / beta;

  % g: Right-hand side of the least-squares problem, updated by Givens rotations
  g = zeros(m + 1, 1);
  g(1) = beta;

  % C, S: Store cosine and sine for Givens rotations
  C = zeros(m, 1);
  S = zeros(m, 1);

  % ---- Arnoldi Iteration Loop ----
  for j = 1:m
    % w = A * V(:,j);
    w = A * V(:, j);

    % Modified Gram-Schmidt process
    for i = 1:j
      H(i, j) = w' * V(:, i);
      w = w - H(i, j) * V(:, i);
    end
    H(j + 1, j) = norm(w);

    % ---- Apply previous Givens rotations to the new column of H ----
    for i = 1:j-1
      temp = C(i) * H(i, j) + S(i) * H(i+1, j);
      H(i+1, j) = -S(i) * H(i, j) + C(i) * H(i+1, j);
      H(i, j) = temp;
    end

    % ---- Calculate and apply new Givens rotation ----
    % This rotates the vector [H(j,j), H(j+1,j)]' to [*, 0]'
    [c_j, s_j] = givens_rotation(H(j, j), H(j + 1, j));
    C(j) = c_j;
    S(j) = s_j;

    H(j, j) = C(j) * H(j, j) + S(j) * H(j + 1, j);
    H(j + 1, j) = 0.0;

    % Apply the same rotation to the RHS vector g
    g(j+1) = -S(j) * g(j);
    g(j)   =  C(j) * g(j);

    rel_res = abs(g(j + 1)) / norm_b;
    iters = j;

    if rel_res < tol
      break; % Converged
    end

    % Check for breakdown
    if abs(H(j + 1, j)) < 1e-12 && j < m
      break;
    end

    V(:, j + 1) = w / H(j + 1, j);
  end

  % ---- Solve the least-squares problem ----
  % Solve the upper triangular system Hy = g
  y = H(1:iters, 1:iters) \ g(1:iters);

  % ---- Update the solution ----
  % x = x0 + V * y
  x = x0 + V(:, 1:iters) * y;
end

% Helper function to compute Givens rotation parameters
function [c, s] = givens_rotation(a, b)
  if b == 0
    c = 1;
    s = 0;
  elseif a == 0
    c = 0;
    s = 1;
  else
    r = sqrt(a^2 + b^2);
    c = a / r;
    s = b / r;
  end
end

