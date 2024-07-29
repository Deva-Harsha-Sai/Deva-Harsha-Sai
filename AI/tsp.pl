
    % tsp.pl
    % Example code snippet for TSP
    % Define cities and distances
    distance(a, b, 1).
    distance(a, c, 4).
    distance(a, d, 3).
    distance(b, c, 2).
    distance(b, d, 5).
    distance(c, d, 1).

    % Define the path cost calculation
    path_cost([A, B], Cost) :- distance(A, B, Cost).
    path_cost([A, B | Rest], Cost) :-
        distance(A, B, Cost1),
        path_cost([B | Rest], Cost2),
        Cost is Cost1 + Cost2.

    % Find the shortest path
    tsp(Start, Path, Cost) :-
        findall(P, permutation([a, b, c, d], P), Perms),
        member([Start | Rest], Perms),
        append([Start | Rest], [Start], Path),
        path_cost(Path, Cost),
        write(Path), nl, write('Cost: '), write(Cost), nl.

    % Queries you can run:
    % ?- tsp(a, Path, Cost).
    