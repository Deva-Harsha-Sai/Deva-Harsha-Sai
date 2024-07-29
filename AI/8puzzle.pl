
    % 8puzzle.pl
    % Define the initial state and the goal state
    initial([1, 2, 3, 4, 5, 6, 7, 8, 0]).
    goal([1, 2, 3, 4, 5, 6, 7, 8, 0]).

    % Define the moves (left, right, up, down)
    move([0, B, C, D, E, F, G, H, I], [B, 0, C, D, E, F, G, H, I]).
    move([A, 0, C, D, E, F, G, H, I], [A, C, 0, D, E, F, G, H, I]).
    move([A, B, 0, D, E, F, G, H, I], [A, B, F, D, E, 0, G, H, I]).
    move([A, B, C, 0, E, F, G, H, I], [A, B, C, E, 0, F, G, H, I]).
    move([A, B, C, D, 0, F, G, H, I], [A, B, C, D, F, 0, G, H, I]).
    move([A, B, C, D, E, 0, G, H, I], [A, B, C, D, E, I, G, H, 0]).
    move([A, B, C, D, E, F, 0, H, I], [A, B, C, D, E, F, H, 0, I]).
    move([A, B, C, D, E, F, G, 0, I], [A, B, C, D, E, F, G, I, 0]).

    % Define the solution
    solve(State) :- goal(State), write('Solved!'), nl.
    solve(State) :- move(State, Move), write(Move), nl, solve(Move).

    % Queries you can run:
    % ?- solve([1, 2, 3, 4, 5, 6, 7, 8, 0]).
    