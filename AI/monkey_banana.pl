
    % monkey_banana.pl
    % Define the initial state and the goal state
    state(atdoor, onfloor, atwindow, hasnot).
    goal(state(_, _, _, has)).

    % Define the actions
    move(state(middle, onbox, middle, hasnot), grasp, state(middle, onbox, middle, has)).
    move(state(atdoor, onfloor, middle, H), walk(middle), state(middle, onfloor, middle, H)).
    move(state(atwindow, onfloor, middle, H), walk(middle), state(middle, onfloor, middle, H)).
    move(state(middle, onfloor, middle, H), climb, state(middle, onbox, middle, H)).
    move(state(atdoor, onfloor, atwindow, H), push(atdoor, middle), state(middle, onfloor, middle, H)).
    move(state(atwindow, onfloor, atwindow, H), push(atwindow, middle), state(middle, onfloor, middle, H)).

    % Define the solution
    solve(State) :- goal(State), write('Solved!'), nl.
    solve(State) :- move(State, Move, State1), write(Move), nl, solve(State1).

    % Queries you can run:
    % ?- solve(state(atdoor, onfloor, atwindow, hasnot)).
    