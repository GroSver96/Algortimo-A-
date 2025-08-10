import tkinter as tk
import math
from queue import PriorityQueue
import time
import random

# Configuraciones
ROWS, COLS = 20, 20
WIDTH = 600
CELL_SIZE = WIDTH // COLS

# Colores
COLOR_BG = "white"
COLOR_GRID = "gray90"
COLOR_START = "green"
COLOR_END = "red"
COLOR_WALL = "black"
COLOR_OPEN = "lightblue"
COLOR_CLOSED = "lightgray"
COLOR_PATH = "yellow"

root = tk.Tk()
root.title("Algoritmo A* (A Estrella) - Laberinto")
canvas = tk.Canvas(root, width=WIDTH, height=WIDTH, bg=COLOR_BG)
canvas.pack()

instructions = tk.Label(root, text="Click Izq: Inicio, Click Der: Final, Shift+Click Izq (o arrastrar): Obstáculo, Barra Espaciadora: Iniciar, C: Reiniciar, R: Mapa Aleatorio",
                        font=("Arial", 10), fg="black")
instructions.pack()

grid = []
start = None
end = None
painting = False  # Variable para saber si estamos pintando con shift+click arrastrando

# Mapa por defecto tipo laberinto más complejo parecido a la imagen
default_map = [
    "####################",
    "#S#......#.........#",
    "#.#.####.#.######..#",
    "#.#....#.#......#..#",
    "#.####.#.######.#..#",
    "#......#........#..#",
    "######.########.#..#",
    "#......#......#.#..#",
    "#.######.####.#.#..#",
    "#........#..#.#.#..#",
    "#.########.##.#.#..#",
    "#........#....#.#..#",
    "######.######.#.#..#",
    "#......#......#.#..#",
    "#.######.######.#..#",
    "#........#........#",
    "########.#.########",
    "#........#.......E#",
    "####################",
    "####################"
]

class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * CELL_SIZE
        self.y = row * CELL_SIZE
        self.color = COLOR_BG
        self.neighbors = []

    def draw(self):
        canvas.create_rectangle(self.x, self.y, self.x + CELL_SIZE, self.y + CELL_SIZE,
                                fill=self.color, outline=COLOR_GRID)

    def set_color(self, color):
        self.color = color
        self.draw()
        root.update()

    def is_wall(self):
        return self.color == COLOR_WALL

    def update_neighbors(self, grid):
        self.neighbors = []
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # ortogonales
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonales
        ]
        for dr, dc in directions:
            r, c = self.row + dr, self.col + dc
            if 0 <= r < ROWS and 0 <= c < COLS and not grid[r][c].is_wall():
                self.neighbors.append(grid[r][c])

def heuristic(a, b):
    return math.sqrt((a.row - b.row)**2 + (a.col - b.col)**2)

def distance(a, b):
    return math.sqrt((a.row - b.row)**2 + (a.col - b.col)**2)

def reconstruct_path(came_from, current):
    while current in came_from:
        current = came_from[current]
        if current.color not in [COLOR_START, COLOR_END]:
            current.set_color(COLOR_PATH)
            time.sleep(0.02)

def a_star(start, end):
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {node: float("inf") for row in grid for node in row}
    g_score[start] = 0
    f_score = {node: float("inf") for row in grid for node in row}
    f_score[start] = heuristic(start, end)

    open_set_hash = {start}

    while not open_set.empty():
        _, _, current = open_set.get()
        open_set_hash.remove(current)

        if current == end:
            reconstruct_path(came_from, end)
            return True

        for neighbor in current.neighbors:
            temp_g = g_score[current] + distance(current, neighbor)
            if temp_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g
                f_score[neighbor] = temp_g + heuristic(neighbor, end)
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    if neighbor.color not in [COLOR_START, COLOR_END]:
                        neighbor.set_color(COLOR_OPEN)
                        time.sleep(0.01)

        if current != start and current != end:
            current.set_color(COLOR_CLOSED)
            time.sleep(0.01)

    return False

def make_grid():
    return [[Node(r, c) for c in range(COLS)] for r in range(ROWS)]

def draw_grid():
    canvas.delete("all")
    for row in grid:
        for node in row:
            node.draw()

def get_clicked_pos(event):
    row = event.y // CELL_SIZE
    col = event.x // CELL_SIZE
    return row, col

def load_default_map(grid, map_data):
    global start, end
    start = None
    end = None
    for r, row in enumerate(map_data):
        for c, val in enumerate(row):
            node = grid[r][c]
            if val == 'S':
                start = node
                start.set_color(COLOR_START)
            elif val == 'E':
                end = node
                end.set_color(COLOR_END)
            elif val == '#':
                node.set_color(COLOR_WALL)
            else:
                node.set_color(COLOR_BG)

def generate_random_map(grid, obstacle_prob=0.2):
    global start, end
    start = None
    end = None

    for row in grid:
        for node in row:
            node.set_color(COLOR_BG)

    start_row, start_col = random.randint(0, ROWS-1), random.randint(0, COLS-1)
    end_row, end_col = random.randint(0, ROWS-1), random.randint(0, COLS-1)
    while (end_row, end_col) == (start_row, start_col):
        end_row, end_col = random.randint(0, ROWS-1), random.randint(0, COLS-1)

    start = grid[start_row][start_col]
    start.set_color(COLOR_START)

    end = grid[end_row][end_col]
    end.set_color(COLOR_END)

    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) != (start_row, start_col) and (r, c) != (end_row, end_col):
                if random.random() < obstacle_prob:
                    grid[r][c].set_color(COLOR_WALL)

def left_click(event):
    global start
    row, col = get_clicked_pos(event)
    node = grid[row][col]
    if node == end:
        return
    if start and node != start:
        start.set_color(COLOR_BG)
    start = node
    start.set_color(COLOR_START)

def right_click(event):
    global end
    row, col = get_clicked_pos(event)
    node = grid[row][col]
    if node == start:
        return
    if end and node != end:
        end.set_color(COLOR_BG)
    end = node
    end.set_color(COLOR_END)

def shift_left_click(event):
    global painting
    painting = True
    paint_wall(event)

def paint_wall(event):
    if not painting:
        return
    row, col = get_clicked_pos(event)
    node = grid[row][col]
    if node != start and node != end and node.color != COLOR_WALL:
        node.set_color(COLOR_WALL)

def stop_paint(event):
    global painting
    painting = False

def shift_drag(event):
    paint_wall(event)

def start_pathfinding(event):
    for row in grid:
        for node in row:
            node.update_neighbors(grid)
    if start and end:
        a_star(start, end)

def reset_grid():
    global grid, start, end
    start = None
    end = None
    grid = make_grid()
    load_default_map(grid, default_map)
    draw_grid()
    root.update()

def load_random_map(event=None):
    generate_random_map(grid, obstacle_prob=0.2)
    draw_grid()
    root.update()

# Bindings eventos
canvas.bind("<Button-1>", left_click)           # Click izquierdo para inicio
canvas.bind("<Button-3>", right_click)          # Click derecho para fin
canvas.bind("<Shift-Button-1>", shift_left_click)  # Shift + click izquierdo para empezar a pintar muro
canvas.bind("<B1-Motion>", shift_drag)          # Arrastrar con click izquierdo y shift para pintar
canvas.bind("<ButtonRelease-1>", stop_paint)    # Soltar click izquierdo para parar de pintar

root.bind("<space>", start_pathfinding)         # Barra espacio inicia A*
root.bind("<Key>", lambda e: reset_grid() if e.char.lower() == "c" else None)  # 'c' para reiniciar
root.bind("r", load_random_map)                  # 'r' para mapa aleatorio

# Inicializar grid con mapa por defecto
grid = make_grid()
load_default_map(grid, default_map)
draw_grid()

root.mainloop()
