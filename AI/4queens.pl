
    % 4queens.pl
    % Define the solution
    valid([]).
    valid([X/Y | Others]) :-
        valid(Others),
        member(Y, [1, 2, 3, 4]),
        noattack(X/Y, Others).

    noattack(_, []).
    noattack(X/Y, [X1/Y1 | Others]) :-
        Y =\= Y1,
        abs(Y1 - Y) =\= abs(X1 - X),
        noattack(X/Y, Others).

    % Queries you can run:
    % ?- valid([1/Y1, 2/Y2, 3/Y3, 4/Y4]).
    