import tkinter as tk
import math
from queue import PriorityQueue
import time
import numpy as np

# ==============================
# Config
# ==============================
ROWS, COLS = 30, 30
WIDTH = 600
CELL_SIZE = WIDTH // COLS

COLOR_GRID = "gray80"
COLOR_START = "lime"
COLOR_END = "red"
COLOR_OPEN = "cyan"
COLOR_CLOSED = "orange"
COLOR_PATH = "yellow"
COLOR_WALL = "gray30"

root = tk.Tk()
root.title("A* en calles del mapa (auto + calibración + obstáculos manuales)")
canvas = tk.Canvas(root, width=WIDTH, height=WIDTH, bg="white")
canvas.pack()

instructions = tk.Label(
    root,
    text=("Click Izq: Inicio | Click Der: Final | ESPACIO: A* | C: Reiniciar | "
          "I: Cargar imagen | K: Calibrar | M: Máscara | Shift+Click: alternar obstáculo"),
    font=("Arial", 10), fg="black")
instructions.pack()

grid = []
start = None
end = None

# máscara detectada originalmente y la máscara activa (editable)
original_mask = np.zeros((ROWS, COLS), dtype=bool)
walkable_mask = np.zeros((ROWS, COLS), dtype=bool)
overlay_on = False

_background_photo = None
_last_img_rgb = None  # PIL Image RGB (WIDTHxWIDTH)

# ==============================
# Nodos
# ==============================
class Node:
    def __init__(self, r, c):
        self.row, self.col = r, c
        self.x, self.y = c * CELL_SIZE, r * CELL_SIZE
        self.color = ""
        self.canvas_id = None
        self.neighbors = []

    def draw(self):
        if self.canvas_id:
            canvas.delete(self.canvas_id)
            self.canvas_id = None
        if self.color:
            outline = "white" if self.color in [COLOR_START, COLOR_END, COLOR_PATH] else COLOR_GRID
            width = 2 if self.color in [COLOR_START, COLOR_END, COLOR_PATH] else 1
            self.canvas_id = canvas.create_rectangle(
                self.x, self.y, self.x+CELL_SIZE, self.y+CELL_SIZE,
                fill=self.color, outline=outline, width=width
            )

    def set_color(self, color):
        self.color = color
        self.draw()
        if color in [COLOR_OPEN, COLOR_CLOSED]:
            time.sleep(0.01)
        elif color == COLOR_PATH:
            time.sleep(0.02)
        root.update()

    def reset_to_background(self):
        if self.canvas_id:
            canvas.delete(self.canvas_id)
            self.canvas_id = None
        self.color = ""

    def is_wall(self):
        return not walkable_mask[self.row, self.col]

    def update_neighbors(self, grid):
        self.neighbors = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            r, c = self.row+dr, self.col+dc
            if 0 <= r < ROWS and 0 <= c < COLS and not grid[r][c].is_wall():
                self.neighbors.append(grid[r][c])

# ==============================
# A*
# ==============================
def heuristic(a,b):
    return abs(a.row-b.row)+abs(a.col-b.col)

def distance(a,b):
    return 1

def reconstruct_path(came_from, current):
    path=[]
    while current in came_from:
        current = came_from[current]
        if current.color not in [COLOR_START, COLOR_END]:
            path.append(current)
    for n in reversed(path):
        n.set_color(COLOR_PATH)

def a_star(start, end):
    count=0
    pq=PriorityQueue()
    pq.put((0,count,start))
    came_from={}
    g={node:float("inf") for row in grid for node in row}
    f={node:float("inf") for row in grid for node in row}
    g[start]=0; f[start]=heuristic(start,end)
    open_hash={start}

    while not pq.empty():
        _,_,cur=pq.get(); open_hash.remove(cur)
        if cur==end:
            reconstruct_path(came_from,end); return True
        for nb in cur.neighbors:
            tg=g[cur]+distance(cur,nb)
            if tg<g[nb]:
                came_from[nb]=cur; g[nb]=tg; f[nb]=tg+heuristic(nb,end)
                if nb not in open_hash:
                    count+=1; pq.put((f[nb],count,nb)); open_hash.add(nb)
                    if nb.color not in [COLOR_START, COLOR_END]:
                        nb.set_color(COLOR_OPEN)
        if cur not in [start,end]:
            cur.set_color(COLOR_CLOSED)
    return False

# ==============================
# Utilidades de dibujo/máscara
# ==============================
def make_grid():
    return [[Node(r,c) for c in range(COLS)] for r in range(ROWS)]

def draw_grid_lines():
    for i in range(ROWS+1):
        y=i*CELL_SIZE; canvas.create_line(0,y,WIDTH,y,fill=COLOR_GRID,width=1)
    for i in range(COLS+1):
        x=i*CELL_SIZE; canvas.create_line(x,0,x,WIDTH,fill=COLOR_GRID,width=1)

def draw_nodes_from_mask():
    for r in range(ROWS):
        for c in range(COLS):
            update_cell_draw(r, c)

def update_cell_draw(r, c):
    node = grid[r][c]
    node.reset_to_background()
    if not walkable_mask[r, c]:
        node.color = COLOR_WALL
    node.draw()

def toggle_mask_overlay():
    global overlay_on
    overlay_on = not overlay_on
    if overlay_on:
        for r in range(ROWS):
            for c in range(COLS):
                if walkable_mask[r,c]:
                    canvas.create_rectangle(
                        c*CELL_SIZE, r*CELL_SIZE, (c+1)*CELL_SIZE, (r+1)*CELL_SIZE,
                        fill="#00ff0040", outline="")
    else:
        reset_grid()

# ==============================
# Detección (HSV)
# ==============================
def rgb_to_hsv_np(arr):  # arr (H,W,3) 0..255
    arr = arr.astype(np.float32)/255.0
    import colorsys
    hsv = np.zeros_like(arr)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            hsv[y,x,:] = colorsys.rgb_to_hsv(arr[y,x,0], arr[y,x,1], arr[y,x,2])
    return hsv

def auto_mask_from_image(img_rgb_600):
    """Reglas HSV para gris + violeta asfáltico, reducción y limpieza ligera."""
    from PIL import Image, ImageFilter
    arr = np.array(img_rgb_600, dtype=np.uint8)
    hsv = rgb_to_hsv_np(arr)
    H, S, V = hsv[:,:,0]*360.0, hsv[:,:,1], hsv[:,:,2]

    m1 = (S < 0.22) & (V > 0.28) & (V < 0.85)
    m2 = (H > 260) & (H < 300) & (S > 0.18) & (S < 0.55) & (V > 0.28) & (V < 0.70)
    mask = (m1 | m2)

    small = Image.fromarray((mask.astype(np.uint8)*255)).resize((COLS, ROWS), Image.Resampling.BOX)
    small = small.filter(ImageFilter.MaxFilter(3))
    small = small.filter(ImageFilter.MinFilter(3))
    return np.array(small) > 127

calib_samples = []

def start_calibration():
    print("🧪 Calibración: haz 3–6 clics en la CARRETERA y pulsa ENTER.")
    root.bind("<Return>", apply_calibration)
    canvas.bind("<Button-1>", sample_click)

def sample_click(event):
    global calib_samples
    if _last_img_rgb is None:
        return
    x = min(max(event.x,0), WIDTH-1)
    y = min(max(event.y,0), WIDTH-1)
    arr = np.array(_last_img_rgb)
    r,g,b = arr[y,x]
    import colorsys
    h,s,v = colorsys.rgb_to_hsv(r/255.0,g/255.0,b/255.0)
    calib_samples.append((h*360.0,s,v))
    print(f"muestra {len(calib_samples)}: H={h*360:.1f} S={s:.2f} V={v:.2f}")

def apply_calibration(event=None):
    global walkable_mask, original_mask, calib_samples
    if len(calib_samples)==0:
        print("No hay muestras. Cancelo calibración."); return
    hs = np.array(calib_samples)
    Hm, Sm, Vm = hs.mean(axis=0)
    dH = max(12, min(35, hs[:,0].ptp()/2 + 15))
    dS = max(0.10, min(0.28, hs[:,1].ptp()/2 + 0.12))
    dV = max(0.10, min(0.28, hs[:,2].ptp()/2 + 0.12))

    arr = np.array(_last_img_rgb, dtype=np.uint8)
    hsv = rgb_to_hsv_np(arr); H,S,V = hsv[:,:,0]*360.0, hsv[:,:,1], hsv[:,:,2]
    m = (np.abs((H - Hm + 180) % 360 - 180) < dH) & (np.abs(S-Sm) < dS) & (np.abs(V-Vm) < dV)

    from PIL import Image, ImageFilter, Image as PILImage
    small = PILImage.fromarray((m.astype(np.uint8)*255)).resize((COLS, ROWS), PILImage.Resampling.BOX)
    small = small.filter(ImageFilter.MaxFilter(3))
    small = small.filter(ImageFilter.MinFilter(3))
    original_mask = np.array(small) > 127
    walkable_mask = original_mask.copy()

    calib_samples = []
    redraw_after_mask()
    print("✅ Calibración aplicada. (Pulsa M para ver overlay)")

# ==============================
# Cargar imagen
# ==============================
def load_map_from_image(file_path):
    global _background_photo, _last_img_rgb, original_mask, walkable_mask
    from PIL import Image, ImageTk

    img = Image.open(file_path).convert("RGB")
    img_resized = img.resize((WIDTH, WIDTH), Image.Resampling.LANCZOS)
    _last_img_rgb = img_resized.copy()
    _background_photo = ImageTk.PhotoImage(img_resized)

    canvas.delete("all")
    canvas.create_image(WIDTH//2, WIDTH//2, image=_background_photo)

    original_mask = auto_mask_from_image(img_resized)
    walkable_mask = original_mask.copy()

    draw_nodes_from_mask()
    draw_grid_lines()

def load_background_image_dialog():
    from tkinter import filedialog
    file_path = filedialog.askopenfilename(
        title="Seleccionar imagen de mapa",
        filetypes=[("Imágenes","*.png *.jpg *.jpeg *.bmp *.gif")]
    )
    if file_path:
        load_map_from_image(file_path)
        print("✅ Imagen cargada. Si algo falla, usa K para calibrar.")

# ==============================
# Interacción (inicio/fin, obstáculos, etc.)
# ==============================
def get_clicked_cell(event):
    return event.y // CELL_SIZE, event.x // CELL_SIZE

def left_click(event):
    # INICIO
    global start
    r,c = get_clicked_cell(event)
    if not (0 <= r < ROWS and 0 <= c < COLS): return
    if not walkable_mask[r,c]: return
    node = grid[r][c]
    if node == end: return
    if start: start.reset_to_background(); update_cell_draw(start.row, start.col)
    start = node; start.set_color(COLOR_START)

def right_click(event):
    # FIN
    global end
    r,c = get_clicked_cell(event)
    if not (0 <= r < ROWS and 0 <= c < COLS): return
    if not walkable_mask[r,c]: return
    node = grid[r][c]
    if node == start: return
    if end: end.reset_to_background(); update_cell_draw(end.row, end.col)
    end = node; end.set_color(COLOR_END)

def start_pathfinding(event):
    if not start or not end:
        print("❌ Coloca INICIO y FIN sobre la carretera"); return
    for row in grid:
        for node in row:
            if node.color in [COLOR_OPEN, COLOR_CLOSED, COLOR_PATH]:
                node.reset_to_background()
                update_cell_draw(node.row, node.col)
            node.update_neighbors(grid)
    print("🔍 Ejecutando A*...")
    ok = a_star(start, end)
    print("✅ Camino encontrado" if ok else "❌ No hay camino")

def reset_grid():
    global start, end
    start = None; end = None
    redraw_after_mask()

def redraw_after_mask():
    canvas.delete("all")
    if _background_photo is not None:
        canvas.create_image(WIDTH//2, WIDTH//2, image=_background_photo)
    draw_nodes_from_mask()
    draw_grid_lines()

def key_handler(event):
    ch = event.char.lower()
    if ch == "c": reset_grid()
    elif ch == "i": load_background_image_dialog()
    elif ch == "k": start_calibration()
    elif ch == "m": toggle_mask_overlay()

# ---- Obstáculos manuales: Shift + click ----
def shift_left_click(event):
    # alterna entre transitable/muro
    r,c = get_clicked_cell(event)
    if not (0 <= r < ROWS and 0 <= c < COLS): return

    # no permitir modificar donde están start/end
    if start and (r,c)==(start.row,start.col): return
    if end and (r,c)==(end.row,end.col): return

    walkable_mask[r,c] = not walkable_mask[r,c]
    update_cell_draw(r, c)

def shift_drag(event):
    # pintar mientras arrastras con Shift
    shift_left_click(event)

# ==============================
# Inicializar
# ==============================
canvas.bind("<Button-1>", left_click)
canvas.bind("<Button-3>", right_click)
canvas.bind("<Shift-Button-1>", shift_left_click)
canvas.bind("<Shift-B1-Motion>", shift_drag)

root.bind("<space>", start_pathfinding)
root.bind("<Key>", key_handler)
canvas.focus_set()

grid = make_grid()

# Carga automática (ajusta la ruta si hace falta)
try:
    DEFAULT_IMG = r"/mnt/data/242bc1d8-997b-4846-88b3-f68fbafc102c.jpg"
    load_map_from_image(DEFAULT_IMG)
    print("🖼️ Mapa cargado (AUTO). Shift+Click para poner/quitar obstáculos.")
except Exception as e:
    print("Pulsa I para cargar tu imagen. Detalle:", e)

print("""
Extras:
- Shift + click (o arrastrar): alterna obstáculo en la celda.
- K: Calibración por clics (ENTER para aplicar).
- M: Ver/ocultar overlay verde de celdas transitables.
""")

root.mainloop()
