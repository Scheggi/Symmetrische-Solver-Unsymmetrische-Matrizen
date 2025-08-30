% OCTAVE EXAMPLES: Symmetrizing & Krylov Methods for (Non-)Symmetric Systems
% -----------------------------------------------------------------------------
% This single Octave/MATLAB-compatible file implements all algorithms and
% examples referenced in the attached write-up:
%  - CG on normal equations (both A^T A and A A^T variants)
%  - LU factorization without pivoting (educational), with solve helpers
%  - GMRES (simple restarted implementation with Givens rotations)
%  - BiCGSTAB
%  - A simple (didactic) non-symmetric CG-like variant
%
% Each function is documented, tested on a small example, and prints residuals.
% Where appropriate, comments explain numerical pros/cons.
%
% USAGE
% -----
% From Octave:
%   octave --no-gui
%   >> run("octave_unsymmetric_solvers_examples.m");
%   >> run_examples();
%
% The 'run_examples' function at the bottom executes all demos.
%
% NOTE
% ----
% This code prioritizes clarity and pedagogy over maximum efficiency. For large
% problems use Octave's built-ins (e.g., bicgstab, gmres) when available.


%% -----------------------------------------------------------------------------
%% 1) CORE BUILDING BLOCKS
%% -----------------------------------------------------------------------------

function assert_spd(M)
  % Asserts that M is (numerically) symmetric positive definite.
  if norm(M - M', 'fro') > 1e-10
    error('Matrix is not symmetric: ||M - M''||_F too large');
  end
  % Try a Cholesky as a cheap SPD test
  [~, p] = chol((M+M')/2);
  if p ~= 0
    error('Matrix is not numerically SPD (Cholesky failed).');
  end
end

  % Conjugate Gradient for SPD systems M x = f.
  % Inputs: x0 (initial guess), tol, maxit.
function [x, it, reshist] = cg_spd(M, f, x0, tol, maxit)

  if nargin < 3 || isempty(x0), x0 = zeros(size(f)); end
  if nargin < 4 || isempty(tol), tol = 1e-10; end
  if nargin < 5 || isempty(maxit), maxit = 10*size(M,1); end

  % assert_spd(M);

  x = x0;
  r = f - M*x;
  p = r;
  rr_old = r'*r;
  res0 = sqrt(rr_old);
  if res0 == 0, it=0; reshist = 0; return; end

  reshist = res0;
  for k = 1:maxit
    Mp = M*p;
    alpha = rr_old / (p' * Mp);
    x = x + alpha * p;
    r = r - alpha * Mp;
    rr = r' * r;
    res = sqrt(rr);
    reshist(end+1) = res; %#ok<AGROW>
    if res <= tol * res0
      it = k; return;
    end
    beta = rr / rr_old;
    p = r + beta * p;
    rr_old = rr;
  end
  it = maxit;
end





%% -----------------------------------------------------------------------------
%% 3) LU FACTORIZATION (educational, no pivoting) + helpers
%% -----------------------------------------------------------------------------
function [L, U] = lu_factorization_nopivot(A)
  % Doolittle-style LU without pivoting. For pedagogical use on well-behaved A.
  % For rectangular A (m x n), returns L (m x m lower unit) and U (m x n upper).
  [m,n] = size(A);
  L = eye(m);
  U = zeros(m,n);
  for k = 1:m
    % Compute U(k,k:n)
    U(k, k:n) = A(k, k:n) - L(k, 1:k-1) * U(1:k-1, k:n);
    if abs(U(k,k)) < eps
      warning('Zero/near-zero pivot at k=%d; method requires pivoting in general.', k);
    end
    % Compute L(i,k)
    for i = k+1:m
      Ukk = U(k,k);
      if abs(Ukk) < eps, error('Division by zero at step %d; pivoting required.', k); end
      L(i,k) = ( A(i,k) - L(i,1:k-1) * U(1:k-1,k) ) / Ukk;
    end
  end
end

function y = forward_substitution(L, b)
  % L lower unit-triangular (m x m), b length m
  m = length(b);
  y = zeros(m,1);
  for i = 1:m
    y(i) = b(i) - L(i,1:i-1) * y(1:i-1);
  end
end

function x = back_substitution_upper(U, y)
  % Back substitution for square U (m x m). If U is m x n (n>=m), we solve for
  % the first m variables (basic variables) and set free vars to zero.
  [m,n] = size(U);
  if n < m
    error('U must have at least as many columns as rows.');
  end
  % Use the left m x m block of U:
  Um = U(:,1:m);
  x_basic = zeros(m,1);
  for i = m:-1:1
    rhs = y(i) - Um(i, i+1:m) * x_basic(i+1:m);
    x_basic(i) = rhs / Um(i,i);
  end
  % Assemble full x with free variables = 0 (minimum-basic solution)
  x = [x_basic; zeros(n-m,1)];
end

function x = solve_lu_nopivot(A, b)
  % Educational solver using our LU without pivoting.
  [L,U] = lu_factorization_nopivot(A);
  y = forward_substitution(L, b);
  x = back_substitution_upper(U, y);
end


%% -----------------------------------------------------------------------------
%% 4) GMRES (restarted) with Givens rotations
%% -----------------------------------------------------------------------------
function [x, flag, relres, iter, reshist] = gmres_simple(A, b, restart, tol, maxit, x0)
  % A compact GMRES(m) implementation.
  %  - restart: subspace dimension m (e.g., 20-50)
  %  - tol: relative residual tolerance
  %  - maxit: max outer cycles
  %  - x0: initial guess
  if nargin < 3 || isempty(restart), restart = 30; end
  if nargin < 4 || isempty(tol), tol = 1e-8; end
  if nargin < 5 || isempty(maxit), maxit = 50; end
  if nargin < 6 || isempty(x0), x0 = zeros(size(b)); end

  x = x0;
  n = length(b);
  flag = 1;
  reshist = [];

  for outer = 1:maxit
    r = b - A*x;
    beta = norm(r);
    reshist(end+1) = beta; %#ok<AGROW>
    if beta == 0 || beta <= tol * norm(b)
      flag = 0; relres = beta / norm(b); iter = outer-1; return;
    end

    V = zeros(n, restart+1);
    H = zeros(restart+1, restart);
    cs = zeros(restart,1); sn = zeros(restart,1);

    V(:,1) = r / beta;
    g = zeros(restart+1,1); g(1) = beta;

    jconv = -1;
    for j = 1:restart
      w = A * V(:,j);
      % Arnoldi orthogonalization
      for i = 1:j
        H(i,j) = V(:,i)' * w;
        w = w - H(i,j) * V(:,i);
      end
      H(j+1,j) = norm(w);
      if H(j+1,j) ~= 0
        V(:,j+1) = w / H(j+1,j);
      end

      % Apply existing Givens rotations to new column j
      for i = 1:j-1
        temp =  cs(i)*H(i,j) + sn(i)*H(i+1,j);
        H(i+1,j) = -sn(i)*H(i,j) + cs(i)*H(i+1,j);
        H(i,j)   = temp;
      end
      % Compute & apply new Givens rotation
      [cs(j), sn(j)] = rotmat(H(j,j), H(j+1,j));
      temp   = cs(j)*g(j);
      g(j+1) = -sn(j)*g(j);
      g(j)   = temp;
      H(j+1,j) = 0;  % rotation annihilates this entry

      rel = abs(g(j)) / norm(b);
      reshist(end+1) = abs(g(j)); %#ok<AGROW>
      if rel <= tol
        jconv = j; break;
      end
    end

    if jconv < 0, jconv = restart; end
    y = H(1:jconv,1:jconv) \ g(1:jconv);
    x = x + V(:,1:jconv) * y;
  end

  r = b - A*x; relres = norm(r)/norm(b); iter = maxit; if relres <= tol, flag = 0; end
end

function [c, s] = rotmat(a, b)
  % Compute Givens rotation parameters c,s so that [c s; -s c]'*[a;b] = [r;0]
  if b == 0
    c = 1; s = 0;
  else
    if abs(b) > abs(a)
      tau = -a/b; s = 1/sqrt(1+tau^2); c = s*tau;
    else
      tau = -b/a; c = 1/sqrt(1+tau^2); s = c*tau;
    end
  end
end


%% -----------------------------------------------------------------------------
%% 5) BiCGSTAB
%% -----------------------------------------------------------------------------
function [x, flag, relres, iter, reshist] = bicgstab_simple(A, b, x0, tol, maxit)
  if nargin < 3 || isempty(x0), x0 = zeros(size(b)); end
  if nargin < 4 || isempty(tol), tol = 1e-8; end
  if nargin < 5 || isempty(maxit), maxit = 1000; end

  x = x0;
  r = b - A*x;
  r0 = r;   % shadow residual
  rho_old = 1; alpha = 1; omega = 1;
  v = zeros(size(b)); p = zeros(size(b));

  reshist = norm(r);
  for iter = 1:maxit
    rho = r0' * r;
    if abs(rho) < eps, flag = 1; relres = norm(r)/norm(b); return; end

    beta = (rho/rho_old) * (alpha/omega);
    p = r + beta*(p - omega*v);

    v = A*p;
    alpha = rho / (r0' * v);

    s = r - alpha*v;
    if norm(s) <= tol*norm(b)
      x = x + alpha*p; flag = 0; relres = norm(s)/norm(b); reshist(end+1)=relres; return;
    end

    t = A*s;
    omega = (t' * s) / (t' * t);
    x = x + alpha*p + omega*s;
    r = s - omega*t;

    relres = norm(r)/norm(b);
    reshist(end+1) = relres; %#ok<AGROW>

    if relres <= tol
      flag = 0; return;
    end
    if abs(omega) < eps
      flag = 1; return;
    end
    rho_old = rho;
  end
  flag = 1;  % not converged within maxit
end


%% -----------------------------------------------------------------------------
%% 6) Non-symmetric CG-like method (didactic only)
%% -----------------------------------------------------------------------------
function [x, reshist] = nonsym_cg_didactic(A, b, x0, maxit, tol)
  % Warning: This is a pedagogical variant (no guarantees for general A).
  if nargin < 3 || isempty(x0), x0 = zeros(size(b)); end
  if nargin < 4 || isempty(maxit), maxit = 100; end
  if nargin < 5 || isempty(tol), tol = 1e-8; end

  x = x0;
  r = b - A*x;
  p = r;
  reshist = norm(r);
  for k = 1:maxit
    Ap = A*p;
    denom = (p' * Ap);
    if abs(denom) < eps, break; end
    alpha = (r' * r) / denom;
    x = x + alpha * p;
    r_new = r - alpha * Ap;
    reshist(end+1) = norm(r_new); %#ok<AGROW>
    if reshist(end) <= tol * norm(b)
      break;
    end
    beta = (r_new' * r_new) / (r' * r);
    p = r_new + beta * p;
    r = r_new;
  end
end


%% -----------------------------------------------------------------------------
%% 7) DEMOS / EXAMPLES
%% -----------------------------------------------------------------------------
function run_examples()
  fprintf('--- RUNNING EXAMPLES ---\n');

  %% Example from the document: A in R^{2x3}, x_true = [1;0;-1], b = A*x_true
  A = [1 2 3; 4 5 6];
  x_true = [1; 0; -1];
  b = A * x_true;

  fprintf('\n[Normal Equations via CG] (underdetermined, min-norm solution)\n');
  [x_ne, it_ne, res_ne] = solve_normal_eq(A, b, 1e-12, 200);
  fprintf('  it = %d, relres = %.2e\n', it_ne, norm(A*x_ne - b)/norm(b));
  fprintf('  x_ne   = [% .6f % .6f % .6f]^T\n', x_ne);
  fprintf('  x_true = [% .6f % .6f % .6f]^T\n', x_true);

  % LU factorization demo (educational). For rectangular A, we can still factor
  % into L (2x2) and U (2x3). Solve with "basic" variables (free vars = 0):
  fprintf('\n[LU (no pivoting) educational demo]\n');
  [L,U] = lu_factorization_nopivot(A);
  disp('  L ='); disp(L); disp('  U ='); disp(U);
  y = forward_substitution(L, b);
  x_lu_basic = back_substitution_upper(U, y); % free variables set to zero
  fprintf('  x_lu (free vars = 0) = [% .6f % .6f % .6f]^T\n', x_lu_basic);
  fprintf('  Residual ||Ax-b||/||b|| = %.2e\n', norm(A*x_lu_basic - b)/norm(b));
  % If we want the *minimum-norm* solution consistent with b, we should prefer
  % the normal-equations route (or pinv), not plain rectangular LU without a
  % strategy for free variables.

  % Square nonsymmetric system for iterative solvers
  B = [3 1 0; -1 3 1; 0 -1 3]; % nonsymmetric but well-behaved
  xs_true = [1; 2; -1];
  rhs = B * xs_true;

  fprintf('\n[GMRES(m)]\n');
  [xg, flagg, rg, itg, histg] = gmres_simple(B, rhs, 20, 1e-10, 50, zeros(size(rhs)));
  fprintf('  flag=%d, relres=%.2e, it=%d\n', flagg, norm(B*xg - rhs)/norm(rhs), itg);
  fprintf('  x_gmres = [% .6f % .6f % .6f]^T\n', xg);

  fprintf('\n[BiCGSTAB]\n');
  [xb, flagb, relb, itb, histb] = bicgstab_simple(B, rhs, zeros(size(rhs)), 1e-10, 1000);
  fprintf('  flag=%d, relres=%.2e, it=%d\n', flagb, relb, itb);
  fprintf('  x_bicgstab = [% .6f % .6f % .6f]^T\n', xb);

  fprintf('\n[Non-symmetric CG (didactic)]\n');
  [xn, histn] = nonsym_cg_didactic(B, rhs, zeros(size(rhs)), 200, 1e-10);
  fprintf('  relres=%.2e\n', norm(B*xn - rhs)/norm(rhs));
  fprintf('  x_nscg = [% .6f % .6f % .6f]^T\n', xn);

  fprintf('\nHint: try "pinv(A)*b" vs. normal equations to see min-norm behavior.\n');
  fprintf('--- DONE ---\n');
end

