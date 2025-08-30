%! Performs an LU decomposition with partial pivoting (PA = LU).
%!
%! This function decomposes a given matrix A into a lower triangular matrix L,
%! an upper triangular matrix U, and a permutation matrix P, such that PA = LU.
%! Partial pivoting (row swapping) is used to ensure numerical stability.
%!
%! @param A The coefficient matrix to be decomposed.
%!
%! @return L The lower triangular matrix with ones on the diagonal.
%! @return U The upper triangular matrix.
%! @return P The permutation matrix that stores the row interchanges.
function [L, U, P] = lu_pivot(A)
  % ---- Initialization ----
  n = size(A, 1);
  U = A;
  L = eye(n); % Identity matrix for L
  P = eye(n); % Identity matrix for P

  % ---- Main loop of Gaussian elimination ----
  for k = 1:n-1
    % ---- Pivot search (find the maximum element) ----
    % Find the index of the element with the largest absolute value in the
    % current column k, starting from the diagonal element.
    [~, max_idx] = max(abs(U(k:n, k)));
    pivot_row = max_idx + k - 1; % Correct the index to refer to the full matrix

    % ---- Row interchange in U, P, and L ----
    if pivot_row ~= k
      % Swap rows in U
      temp_row_U = U(k, :);
      U(k, :) = U(pivot_row, :);
      U(pivot_row, :) = temp_row_U;

      % Swap rows in P
      temp_row_P = P(k, :);
      P(k, :) = P(pivot_row, :);
      P(pivot_row, :) = temp_row_P;

      % Swap the already computed parts of L to maintain the correctness
      % of the multipliers.
      if k > 1
        temp_row_L = L(k, 1:k-1);
        L(k, 1:k-1) = L(pivot_row, 1:k-1);
        L(pivot_row, 1:k-1) = temp_row_L;
      end
    end

    % ---- Elimination step ----
    % Calculate the multipliers and perform the row operations.
    for i = k+1:n
      % Multiplier for the current row
      L(i, k) = U(i, k) / U(k, k);
      % Subtract the multiple of the pivot row from the current row
      U(i, k:n) = U(i, k:n) - L(i, k) * U(k, k:n);
    end
  end
end


%! Solves the linear system Ax = b using a pre-computed PA=LU decomposition.
%!
%! @param L The lower triangular matrix from the lu_pivot function.
%! @param U The upper triangular matrix from the lu_pivot function.
%! @param P The permutation matrix from the lu_pivot function.
%! @param b The right-hand side vector.
%!
%! @return x The solution vector.
function x = solve_lu(L, U, P, b)
  n = length(b);

  % ---- Step 1: Apply permutation ----
  % Solve PAx = Pb -> LUx = Pb.
  % Define c = Pb.
  c = P * b;

  % ---- Step 2: Forward substitution (Solve Ly = c) ----
  y = zeros(n, 1);
  y(1) = c(1); % L has ones on the diagonal
  for i = 2:n
    y(i) = c(i) - L(i, 1:i-1) * y(1:i-1);
  end

  % ---- Step 3: Backward substitution (Solve Ux = y) ----
  x = zeros(n, 1);
  x(n) = y(n) / U(n, n);
  for i = n-1:-1:1
    x(i) = (y(i) - U(i, i+1:n) * x(i+1:n)) / U(i, i);
  end
end

