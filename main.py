# Dynamically download required modules
import subprocess
import sys
import pkg_resources

required = {"numpy", "matplotlib"}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    python_path = sys.executable
    subprocess.check_call([python_path, "-m", "pip", "install", *missing])


import itertools
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import AutoLocator

import numpy as np

# PARAMETERS FOR THE SIMULATION

N = 1350  # Cells count
D = 0.02  # Initial sick cells ratio
R = 0.2  # Swift cells ratio
X = 28  # Generations count until recovery
P_HIGH = 0.28  # High infection chance
P_LOW = 0.20  # Low infection chance
T = 0.05  # Infection chance change threshold (From high to low)

# COLOR CONSTANTS
COLOR_EMPTY_CELL = [0x00, 0x00, 0x00]  # Black
COLOR_HEALTHY_CELL = [0, 0x3e, 0xff]  # Blue
COLOR_SICK_CELL = [0xff, 0, 0]  # Red
COLOR_RECOVERED_CELL = [0, 0xf1, 0x33]  # Green
GRID_COLORS = np.array([COLOR_EMPTY_CELL,
                        COLOR_HEALTHY_CELL,
                        COLOR_SICK_CELL,
                        COLOR_RECOVERED_CELL])


class CellState:
    """
    Enum for saving cell state. The state values correspond to the color indexes for the grid
    """
    EMPTY = 0
    HEALTHY = 1
    SICK = 2
    RECOVERED = 3


class Cell:
    def __init__(self, x, y, speed=1):
        """
        Cell init
        :param x: X position
        :param y: Y position
        :param speed: Max steps per generation
        """
        self.x = x
        self.y = y
        self.state = CellState.HEALTHY
        self._speed = speed

        self.sickness_generation = None

    @property
    def speed(self):
        return self._speed

    def __repr__(self):
        return f"X: {self.x}    Y: {self.y}    speed: {self.speed}     state: {self.state}"


class CellularAutomaton:
    """
    Class that represent the cellular automaton described in the exercise
    """
    def __init__(self, size_x, size_y, cells_count, initial_sick_chance, swift_cells_chance,
                 generations_until_recovery, low_infection_chance, high_infection_chance,
                 low_infection_threshold):
        self._size_x = size_x
        self._size_y = size_y
        self._cells_count = cells_count
        self._initial_sick_chance = initial_sick_chance
        self._swift_cells_chance = swift_cells_chance
        self._generations_until_recovery = generations_until_recovery
        self._low_infection_chance = low_infection_chance
        self._high_infection_chance = high_infection_chance
        self._low_infection_threshold = low_infection_threshold
        self._grid = np.zeros((size_x, size_y), dtype=int)
        self._generations_passed = 0

        self._cells_states = {
            CellState.HEALTHY: [],
            CellState.SICK: [],
            CellState.RECOVERED: []
        }

        self.__init_cells()

    @property
    def generations_passed(self):
        return self._generations_passed

    @property
    def sick_ratio(self):
        return self.__get_cell_ratio(CellState.SICK)

    @property
    def healthy_ratio(self):
        return self.__get_cell_ratio(CellState.HEALTHY)

    @property
    def recovered_ratio(self):
        return self.__get_cell_ratio(CellState.RECOVERED)

    def __get_cell_ratio(self, state):
        return len(self._cells_states[state]) / float(len(self._cells))

    @property
    def _cells(self):
        result = []
        for _, cells in self._cells_states.items():
            result += cells
        return result

    @property
    def _infection_chance(self):
        return self._high_infection_chance if self.sick_ratio < self._low_infection_threshold else self._low_infection_chance

    def __init_cells(self):
        available_positions = list(range(self._size_x * self._size_y))  # Board is empty.
        for i in range(self._cells_count):
            flat_index = random.choice(available_positions)  # Choose position.
            x, y = np.unravel_index(flat_index, (self._size_x, self._size_y))
            speed = 10 if random.random() < self._swift_cells_chance else 1  # Choose cell speed.
            cell = Cell(x, y, speed)
            self._cells_states[CellState.HEALTHY].append(cell)
            self._grid[x % self._size_x, y % self._size_y] = cell.state
            available_positions.remove(flat_index)

    def start_disease(self):
        # Sicken cells.
        for cell in self._cells:
            if random.random() < self._initial_sick_chance:
                self.__update_cell_state(cell, CellState.SICK)

    def __update_cell_state(self, cell, new_state):
        # Remove the cell from its current list
        self._cells_states[cell.state].remove(cell)

        # Update cell state
        cell.state = new_state
        self._grid[cell.x % self._size_x, cell.y % self._size_y] = new_state

        # Add the cell to the new state list
        self._cells_states[cell.state].append(cell)

        if new_state == CellState.SICK:
            cell.sickness_generation = self._generations_passed

        if new_state == CellState.RECOVERED:
            cell.sickness_generation = None

    def __is_valid_move(self, cell, target_x, target_y):
        if cell.x == target_x and cell.y == target_y:
            return True

        return self._grid[target_x % self._size_x, target_y % self._size_y] == CellState.EMPTY

    def __move_cell(self, cell: Cell):
        """
        Moves cell around the grid.
        1. The function creates a list of all movement offset options.
        2. Then it randomly chooses one option, and checks if it is valid (if a cell is in the target position, using __is_valid_move)
        While a valid option wasn't found:
            Remove the sampled invalid option from the list
            Go to step 2

        Eventually a valid option must be found, because a cell can always stay in place.
        :param cell: The cell object to move
        """
        # Create a list of all the move options for current cell.
        options = list(itertools.product(range(-cell.speed, cell.speed + 1), repeat=2))

        while True:
            x_offset, y_offset = random.choice(options)  # Sample a move.
            target_x, target_y = cell.x + x_offset, cell.y + y_offset

            if not self.__is_valid_move(cell, target_x, target_y):
                options.remove((x_offset, y_offset))  # Option is not relevant for next sample.
            else:
                break
        # Chose a valid move.
        self._grid[cell.x % self._size_x, cell.y % self._size_y] = CellState.EMPTY
        cell.x = target_x % self._size_x
        cell.y = target_y % self._size_y
        self._grid[cell.x, cell.y] = cell.state

    def step_generation(self):
        for cell in self._cells:
            self.__move_cell(cell)

        for healthy in self._cells_states[CellState.HEALTHY]:
            self._handle_healthy_cell(healthy)

        for sick in self._cells_states[CellState.SICK]:
            self._handle_sick_cell(sick)

        self._generations_passed += 1

    def is_at_risk(self, cell):
        offsets = [-1, 0, 1]
        for x in offsets:
            for y in offsets:
                if self._grid[(cell.x + x) % self._size_x, (cell.y + y) % self._size_y] == CellState.SICK:
                    return True

        return False

    def _handle_healthy_cell(self, cell):
        if not self.is_at_risk(cell):
            return

        infected = random.random() < self._infection_chance
        if not infected:
            return

        self.__update_cell_state(cell, CellState.SICK)

    def _handle_sick_cell(self, cell):
        if self._generations_passed - cell.sickness_generation >= self._generations_until_recovery:
            self.__update_cell_state(cell, CellState.RECOVERED)

    def __str__(self):
        return self._grid.__str__()

    def __repr__(self):
        return self._grid.__repr__()


def run_simulation(automaton):
    """
    Runs the whole simulation using matplotlib animation
    """
    sickness_ratio_values = []

    fig, (ax_grid, ax_sickness) = plt.subplots(2, figsize=(10, 8))
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])

    state_text = ax_grid.text(205, 110, "")
    grid_plot = ax_grid.imshow(GRID_COLORS[automaton._grid], interpolation='nearest')

    sickness_line, = ax_sickness.plot([], [])
    ax_sickness.set_xlabel("Generations")
    ax_sickness.set_ylabel("Sick %")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    legend_text = ax_sickness.text(0.77, 0.95, "", transform=ax_sickness.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    def update(frame):
        plt.suptitle(f"Generation {automaton.generations_passed}")
        state_text.set_text(f"Sick: {automaton.sick_ratio * 100 : .2f}%\n"
                            f"Healthy: {automaton.healthy_ratio * 100 : .2f}%\n"
                            f"Recovered: {automaton.recovered_ratio * 100 : .2f}%")

        # place a text box in upper left in axes coords
        legend_text.set_text(f"N = {N}\n"
                             f"D = {D}\n"
                             f"R = {R}\n"
                             f"X = {X}\n"
                             f"P_HIGH = {P_HIGH}\n"
                             f"P_LOW = {P_LOW}\n"
                             f"T = {T}\n"
                             f"Healthy: {automaton.healthy_ratio * 100 : .2f}%")

        if sickness_ratio_values and sickness_ratio_values[-1] == 0 and len(sickness_ratio_values) > 1:
            # No more sick cells. Stop simulation
            ani.pause()
            return

        sickness_ratio_values.append(automaton.sick_ratio)

        automaton.step_generation()
        grid_plot.set_data(GRID_COLORS[automaton._grid])

        sickness_line.set_data(np.arange(automaton.generations_passed), np.array(sickness_ratio_values))

        ax_sickness.set_xlim(0, automaton.generations_passed)
        ax_sickness.set_ylim(0, max(sickness_ratio_values) + 0.001)
        ax_sickness.xaxis.set_major_locator(AutoLocator())
        ax_sickness.yaxis.set_major_locator(AutoLocator())
        ax_sickness.axhline(y=automaton._low_infection_threshold, color='r', linestyle='-', linewidth=0.5)

        if automaton.generations_passed == 1:
            automaton.start_disease()

        return grid_plot, sickness_line

    ani = animation.FuncAnimation(fig, update, interval=1)
    plt.show()


if __name__ == '__main__':
    automaton = CellularAutomaton(200, 200, N, D, R, X, P_LOW, P_HIGH, T)
    run_simulation(automaton)
