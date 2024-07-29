
    % waterjug.pl
    % Define the initial state and the goal state
    state(0, 0).
    goal(state(2, _)).

    % Define the actions
    move(state(X, Y), fillX, state(3, Y)).
    move(state(X, Y), fillY, state(X, 4)).
    move(state(3, Y), emptyX, state(0, Y)).
    move(state(X, 4), emptyY, state(X, 0)).
    move(state(3, Y), pourXY, state(Z, 4)) :- Z is X - (4 - Y), Z >= 0.
    move(state(X, 4), pourYX, state(3, Z)) :- Z is Y - (3 - X), Z >= 0.

    % Define the solution
    solve(State) :- goal(State), write('Solved!'), nl.
    solve(State) :- move(State, Move, State1), write(Move), nl, solve(State1).

    % Queries you can run:
    % ?- solve(state(0, 0)).
    