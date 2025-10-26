# Robot Navigation
### ME 469: HW1
### Author: Jared Berry
### Due: 10/27/2025

Hello!

To run for submission B, run the following command from directory HW1/:

```

python run.py

```

All plots will populate in figures/, and data (ds1) is located in data/. JSON files with metrics will populate in
metrics/

In each A* path plot, the blue cell is the start and the green cell is the goal. The plots with robot trajectories
will represent the full trajectory with a black line, and will also plot the robot position and heading as a blue arrow
at a certain interval.

#### Code Structure

- run.py: Contains main function for running each question.

- questions.py: (From top to bottom) First are helper functions for plotting and obstacle generation. Next are separate functions for each question associated with a plot or data.

- search.py: Contains all search algorithms implemented.

- data/ : Contains ds1

- figures/ : Contains all figures used in report. 