function [x, iterationCount, residualHistory] = solve_normal_eq(A, b, tolerance, maxIterations)
  %SOLVE_NORMAL_EQ  CG auf Normalgleichungen:
  %  - m >= n (überbestimmt):  (A' A) x = A' b
  %  - m <  n (unterbestimmt, Min-Norm): (A A') y = b,  x = A' y
  %
  % Inputs
  %   A            : m x n Matrix
  %   b            : m x 1 Vektor
  %   tolerance    : Relativtoleranz (optional, default 1e-10)
  %   maxIterations: Max. CG-Iterationen (optional, default 200)
  %
  % Outputs
  %   x                 : Lösung
  %   iterationCount    : Anzahl Iterationen des inneren CG
  %   residualHistory   : Verlauf der Residuen-Norm im inneren CG

  [m, n] = size(A);

  if m >= n
    normalMatrix = A' * A;
    normalRhs    = A' * b;
    initialGuess = zeros(n, 1);
    [x, iterationCount, residualHistory] = cg_spd(normalMatrix, normalRhs, initialGuess, tolerance, maxIterations);
  else
    normalMatrix = A * A';
    normalRhs    = b;
    initialGuess = zeros(m, 1);
    [y, iterationCount, residualHistory] = cg_spd(normalMatrix, normalRhs, initialGuess, tolerance, maxIterations);
    x = A' * y;
  end
end



function [x, iterationCount, residualHistory] = cg_spd(A, b, initialGuess, tolerance, maxIterations)
  %CG_SPD  Klassischer Conjugate-Gradient für SPD-Systeme A*x = b
  %
  % Inputs
  %   A             : Symmetrisch positiv definite Matrix
  %   b             : Result
  %   initialGuess  : Startwert x0 (optional, default zeros)
  %   tolerance     : Relativtoleranz bzgl. ||r||/||r0|| (optional, default 1e-10)
  %   maxIterations : Max. Iterationen (optional, default 1000)
  %
  % Outputs
  %   x                 : Näherungslösung
  %   iterationCount    : Anzahl Iterationen
  %   residualHistory   : Verlauf der Residuen-Norm

  x = initialGuess;
  residual = b - A * x;
  searchDirection = residual;

  residualNormSq = residual' * residual;
  initialResidualNorm = sqrt(residualNormSq);
  residualHistory = initialResidualNorm;

  if initialResidualNorm == 0
    iterationCount = 0;
    return;
  end

  for k = 1:maxIterations
    Asearch = A * searchDirection;
    alpha   = residualNormSq / (searchDirection' * Asearch);

    x        = x + alpha * searchDirection;
    residual = residual - alpha * Asearch;

    newResidualNormSq = residual' * residual;
    residualHistory(end+1) = sqrt(newResidualNormSq);

    if sqrt(newResidualNormSq) <= tolerance * initialResidualNorm
      iterationCount = k;
      return;
    end

    beta = newResidualNormSq / residualNormSq;
    searchDirection = residual + beta * searchDirection;
    residualNormSq  = newResidualNormSq;
  end

  iterationCount = maxIterations; % Konvergenz nicht erreicht
end


function [x, iterationCount, residualHistory] = cg_classic(A, b, x0, tolerance, maxIterations)
  % Conjugate Gradient method for solving SPD system A*x = b
  %
  % Inputs:
  %   A             - Symmetric positive definite matrix
  %   b             - Right-hand side vector
  %   x0            - Initial guess (default: zeros)
  %   tolerance     - Relative residual tolerance (default: 1e-10)
  %   maxIterations - Maximum number of iterations (default: 1000)
  %
  % Outputs:
  %   x                 - Approximate solution
  %   iterationCount     - Iterations performed
  %   residualHistory    - History of residual norms

  x = x0;
  residual = b - A*x;
  searchDir = residual;
  residualNormSquared = residual' * residual;
  initialResidualNorm = sqrt(residualNormSquared);

  residualHistory = initialResidualNorm;
  if initialResidualNorm == 0
    iterationCount = 0;
    return;
  end

  for k = 1:maxIterations
    Ap = A * searchDir;
    alpha = residualNormSquared / (searchDir' * Ap);

    x = x + alpha * searchDir;
    residual = residual - alpha * Ap;

    newResidualNormSquared = residual' * residual;
    residualHistory(end+1) = sqrt(newResidualNormSquared); %#ok<AGROW>

    if sqrt(newResidualNormSquared) <= tolerance * initialResidualNorm
      iterationCount = k;
      return;
    end

    beta = newResidualNormSquared / residualNormSquared;
    searchDir = residual + beta * searchDir;

    residualNormSquared = newResidualNormSquared;
  end

  iterationCount = maxIterations;
end


