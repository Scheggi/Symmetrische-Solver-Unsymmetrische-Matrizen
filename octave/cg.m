%! Solves Ax = b for a non-symmetric A using the Normal Equations approach.
%!
%! This method transforms the original linear system Ax = b into the
%! symmetric positive semi-definite system (A'A)x = A'b, and then solves
%! this new system using the standard Conjugate Gradient (CG) method.
%!
%! Note: This approach can square the condition number of the matrix, which
%! may affect convergence and numerical stability for ill-conditioned problems.
%!
%! @param A        The coefficient matrix `A` (can be non-symmetric or rectangular).
%! @param b        The right-hand side vector `b`.
%! @param x0       The initial guess for the solution `x`.
%! @param max_iter The maximum number of iterations to perform.
%! @param tol      The convergence tolerance for the relative residual of the normal equations.
%!
%! @return x       The solution vector `x`.
%! @return rel_res The final relative residual norm `||A'b - A'Ax|| / ||A'b||`.
%! @return iters   The number of iterations performed.
function [x, rel_res, iters] = cg_normal(A, b, x0, max_iter, tol)
  % ---- Step 1: Transform the system to the Normal Equations ----
  % Bx = c, where B = A'A and c = A'b
  B = A' * A;
  c = A' * b;

  % ---- Step 2: Initialize the CG solver ----
  x = x0;
  r = c - B * x; % Initial residual of the normal system
  p = r;         % Initial search direction
  rs_old = r' * r;

  norm_c = norm(c);
  if norm_c == 0.0 % Trivial case
    x = zeros(size(A, 2), 1);
    rel_res = 0.0;
    iters = 0;
    return;
  end

  % ---- Step 3: Main CG Iteration Loop ----
  for iter_count = 1:max_iter
    % Bp = A' * (A * p); This is more efficient for sparse A
    Bp = B * p;

    % Calculate step size alpha
    alpha_den = p' * Bp;
    if abs(alpha_den) < 1e-20
      warning('cgnormal: breakdown', 'Solver failed because p''*B*p is near zero.');
      break;
    end
    alpha = rs_old / alpha_den;

    % Update solution and residual
    x = x + alpha * p;
    r = r - alpha * Bp;
    rs_new = r' * r;

    % ---- Convergence Check ----
    rel_res = sqrt(rs_new) / norm_c;
    if rel_res < tol
      iters = iter_count;
      return; % Converged
    end

    % Update search direction p
    p = r + (rs_new / rs_old) * p;
    rs_old = rs_new;
  end

  % ---- Loop finished without converging ----
  rel_res = sqrt(rs_new) / norm_c;
  iters = max_iter;
  if rel_res >= tol
    warning('cgnormal: no convergence', ...
            'CGNE failed to converge within %d iterations to a tolerance of %e.', ...
            max_iter, tol);
  end
end

