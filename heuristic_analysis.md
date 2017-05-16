

# Heuristic Analysis

## Three Evaluation Functions

The best performing heuristic (AB_Custom) is one that incorporates adding taken-nodes count within a 7 x 7 range from the player (incorporated when the blank space to total space ratio is less than 0.7), distance from other player (ratio with max distance possible), distance from center (ratio over maximum distance), and the open move difference between mine vs. the opponent.

The second best performing heuristic (AB_Custom-2) incorporates distance from other player (ratio with max distance possible), distance from center (using absolute values), and the open move difference between mine vs. the opponent.

The third best performing heuristic (AB_Custom-3) only incorporates the open move difference between mine vs. the opponent and the center score ratio.

The heuristic function needs to be fast in order to search deeper to get more depth thus, more information. 

With the second heuristic function, maintaining a larger distance between the two players seemed to help our performance. I assume that this helped us focus on surviving without competing with the other player.

With the first heuristic function, we found that when I inadvertently tried to pick nodes that is centered in the middle of the board, vs. edges of the board (when the user has lower number of taken nodes in the 7 x 7 range), that the player performed better. My theory is that it incentivized the player to focus on being in the middle of the board. Maybe that helped it create an isolation scenario.

Lastly, the third heuristic seemed to prove that a combination of heuristic functions performs better when compared to evaluation functions that incorporated only one of each.

## Report list

When analyzing the table below with the match outcomes, all the heuristic functions performed well against all the minimax-based heuristic functions. It seems like pruning and iterative deepening is helping with maximizing the best decision within a given time frame.

The AB_Custom and AB_Custom-2 score heuristics both did well. However, they differed in performance when competing against Alpha Beta heuristics vs Minimax heuristics. My theory is as follows: AB_Custom, which has a longer running time due to the extra 7x7 taken-node analysis takes a longer time to run, thus, limiting the depth searched when compared to AB_Custom-2, which is a heuristic function that's equivalent to AB_Custom except for the 7x7 taken-node analysis.

Interestingly, AB_Custom performed better than AB_Custom-2 when playing against Minimax heuristics. My theory behind this is that having that extra 7x7 analysis does not take a big enough hit against the less optimal search algorithm, whilst the benefit of that heuristic out-performs AB_Custom-2

                              *************************
                              	   Playing Matches
                              *************************
    Match #   Opponent    AB_Improved   AB_Custom   AB_Custom_2  AB_Custom_3
                          Won | Lost   Won | Lost   Won | Lost   Won | Lost
      1       Random      185 |  15    185 |  15    185 |  15    183 |  17
      2       MM_Open     146 |  54    158 |  42    161 |  39    153 |  47
      3      MM_Center    174 |  26    188 |  12    175 |  25    177 |  23
      4     MM_Improved   135 |  65    149 |  51    138 |  62    152 |  48
      5       AB_Open     111 |  89    100 |  100   109 |  91    100 |  100
      6      AB_Center    120 |  80    109 |  91    111 |  89    117 |  83
      7     AB_Improved   88  |  112   102 |  98    112 |  88    103 |  97

--------------------------------------------------------------------------
           Win Rate:      68.5%        70.8%        70.8%        70.4%

## Recommendation

I recommend AB_Custom-2. With the quickness of adding together the manhattan distance from the opponent player, manhattan distance from the center, and lastly, the open move difference, AB_Custom-2 has a win rate of 70.8% when compared to all other heuristic functions. Lastly, let us focus the results to only alpha beta search algorithms and AB_Custom and AB_Custom-2 heuristic functions. It is apparent in the table above that AB_Custom-2 performed better. Thus, concluding that the quickness of the AB_Custom-2 heuristic function relative to AB_Custom heuristic function is important.