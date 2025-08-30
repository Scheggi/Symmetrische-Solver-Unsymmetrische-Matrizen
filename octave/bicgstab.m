%! Solves the linear system Ax = b using the BiCGSTAB method.
%!
%! @param A        The coefficient matrix `A` (can be non-symmetric).
%! @param b        The right-hand side vector `b`.
%! @param x0       The initial guess for the solution `x`.
%! @param max_iter The maximum number of iterations to perform.
%! @param tol      The convergence tolerance for the relative residual.
%!
%! @return x       The solution vector `x`.
%! @return rel_res The final relative residual norm `||b - Ax|| / ||b||`.
%! @return iters   The number of iterations performed.
function [x, rel_res, iters] = bicgstab(A, b, x0, max_iter, tol)
  % ---- Initialization ----
  x = x0;
  r = b - A * x;
  r0 = r; % Shadow residual

  rho_old = 1.0;
  alpha = 1.0;
  omega = 1.0;

  n = length(b);
  v = zeros(n, 1);
  p = zeros(n, 1);

  norm_b = norm(b);
  if norm_b == 0.0 % Trivial case
    x = zeros(n, 1);
    rel_res = 0.0;
    iters = 0;
    return;
  end

  % ---- Main Iteration Loop ----
  for iter_count = 1:max_iter
    % rho = r0' * r;
    rho = r0' * r;
    if abs(rho) < 1e-20
      warning('bicgstab: breakdown', 'Solver failed because rho is near zero.');
      break;
    end

    % beta = (rho / rho_old) * (alpha / omega);
    beta = (rho / rho_old) * (alpha / omega);

    % p = r + beta * (p - omega * v);
    p_update = p - omega * v;
    p = r + beta * p_update;

    % v = A * p;
    v = A * p;

    % alpha = rho / (r0' * v);
    r0_dot_v = r0' * v;
    if abs(r0_dot_v) < 1e-20
      warning('bicgstab: breakdown', "Solver failed because r0' * v is near zero.");
      break;
    end
    alpha = rho / r0_dot_v;

    % s = r - alpha * v;
    s = r - alpha * v;

    % t = A * s;
    t = A * s;

    % omega = (t' * s) / (t' * t);
    t_dot_t = t' * t;
    if abs(t_dot_t) < 1e-20
      warning('bicgstab: breakdown', "Solver failed because t' * t is near zero.");
      break;
    end
    omega = (t' * s) / t_dot_t;

    % x = x + alpha * p + omega * s;
    x = x + alpha * p + omega * s;

    % r = s - omega * t;
    r = s - omega * t;

    % ---- Convergence Check ----
    rel_res = norm(r) / norm_b;
    if rel_res < tol
      iters = iter_count;
      return; % Converged
    end

    % Update for next iteration
    rho_old = rho;
  end

  % ---- Loop finished without converging ----
  rel_res = norm(r) / norm_b;
  iters = max_iter;
  if rel_res >= tol
    warning('bicgstab: no convergence', ...
            'BiCGSTAB failed to converge within %d iterations to a tolerance of %e.', ...
            max_iter, tol);
  end
end

