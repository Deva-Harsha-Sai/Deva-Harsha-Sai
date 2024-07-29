
    % arithmetic.pl
    % Define some arithmetic operations
    add(X, Y, Z) :- Z is X + Y.
    subtract(X, Y, Z) :- Z is X - Y.
    multiply(X, Y, Z) :- Z is X * Y.
    divide(X, Y, Z) :- Y \= 0, Z is X / Y.

    % Queries you can run:
    % ?- add(3, 5, Result).
    % ?- subtract(10, 4, Result).
    % ?- multiply(4, 5, Result).
    % ?- divide(20, 4, Result).
    