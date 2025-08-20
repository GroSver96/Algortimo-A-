import pygame
import math
from queue import PriorityQueue
from PIL import Image
import os
import numpy as np

# ------------------------------
# CONFIGURACIÓN
# ------------------------------
GRID_SIZE = 60           # cantidad de celdas por lado (sube/baja para más/menos detalle)
WINDOW = 800             # tamaño del lado del área del mapa (no incluye la barra inferior)
FOOTER_H = 60            # alto de la barra inferior reducido
IMG_FILENAME = "ciudad.jpg"

# Colores UI
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID_COLOR = (200, 200, 200)
ORANGE = (255, 165, 0)      # inicio
CYAN = (0, 180, 200)        # fin
GREEN = (0, 200, 0)         # open set
RED = (220, 0, 0)           # closed set
PURPLE = (150, 0, 150)      # path
CLOSED_ROAD = (244, 50, 11) # Color para rutas cerradas: #F4320B
FOOTER_BG = (245, 245, 245)
FOOTER_TEXT = (20, 20, 20)

# Configuración de reconocimiento de carreteras MEJORADO
ROAD_COLORS = [
    (106, 95, 107),   # Color original #6a5f6b
    (104, 97, 105),   # Color #686169
    (128, 128, 128),  # Gris medio
    (96, 96, 96),     # Gris oscuro
    (160, 160, 160),  # Gris claro
    (80, 80, 80),     # Gris muy oscuro
    (200, 200, 200),  # Gris muy claro
    (120, 120, 120),  # Gris intermedio
    (140, 140, 140),  # Gris intermedio claro
    (110, 110, 110),  # Gris intermedio oscuro
]

# Tolerancia más agresiva para reconocer más carreteras
ROAD_TOL = 50  # Aumentado significativamente

# Configuración para análisis de luminosidad - EXPANDIDO
USE_BRIGHTNESS_DETECTION = True
MIN_BRIGHTNESS = 50      # Reducido para incluir carreteras más oscuras
MAX_BRIGHTNESS = 200     # Aumentado para incluir carreteras más claras

# Configuración para análisis de saturación - MÁS PERMISIVO
USE_SATURATION_FILTER = True
MAX_SATURATION = 70      # Aumentado para ser más permisivo

# Variable global para el estado del camino
path_status = "Sin intentos de búsqueda"

# ------------------------------
# UTILIDADES MEJORADAS
# ------------------------------
def rgb_to_hsv(r, g, b):
    """Convierte RGB a HSV"""
    r, g, b = r/255.0, g/255.0, b/255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    # Value (brillo)
    v = max_val
    
    # Saturation
    s = 0 if max_val == 0 else diff / max_val
    
    # Hue
    if diff == 0:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    return h, s * 100, v * 100

def get_brightness(r, g, b):
    """Calcula el brillo percibido de un color RGB"""
    return 0.299 * r + 0.587 * g + 0.114 * b

def rgb_distance(a, b):
    """Distancia Euclídea en RGB"""
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def is_road_color_improved(rgb):
    """
    Función mejorada para detectar si un color corresponde a una carretera.
    Usa criterios más amplios para capturar más variaciones.
    """
    r, g, b = rgb
    
    # Método 1: Comparación con colores conocidos de carreteras (más tolerante)
    for road_color in ROAD_COLORS:
        if rgb_distance(rgb, road_color) <= ROAD_TOL:
            return True
    
    # Método 2: Análisis de brillo expandido
    brightness = get_brightness(r, g, b)
    if not (MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS):
        return False
    
    # Método 3: Análisis de saturación más permisivo
    h, s, v = rgb_to_hsv(r, g, b)
    if s > MAX_SATURATION:
        return False
    
    # Método 4: Detección amplia de tonos grises/neutrales
    rgb_range = max(r, g, b) - min(r, g, b)
    if rgb_range <= 60 and MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS:
        return True
    
    # Método 5: Detección por rangos de colores comunes en carreteras
    # Rango de grises oscuros a medios
    if (75 <= r <= 170 and 70 <= g <= 170 and 75 <= b <= 170 and 
        abs(r - g) <= 25 and abs(g - b) <= 25 and abs(r - b) <= 25):
        return True
    
    # Método 6: Detección específica para tonos violeta-gris (como #6a5f6b)
    if (90 <= r <= 130 and 80 <= g <= 120 and 90 <= b <= 130):
        return True
    
    return False

def analyze_image_colors(image_path, rows):
    """Analiza los colores de la imagen para ayudar con la calibración"""
    try:
        pil_img = Image.open(image_path).convert("RGB")
        pil_img = pil_img.resize((rows, rows), Image.NEAREST)
        pixels = pil_img.load()
        
        color_histogram = {}
        road_candidates = []
        
        for i in range(rows):
            for j in range(rows):
                rgb = pixels[j, i]
                color_key = f"{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                color_histogram[color_key] = color_histogram.get(color_key, 0) + 1
                
                if is_road_color_improved(rgb):
                    road_candidates.append(rgb)
        
        # Mostrar estadísticas básicas
        road_count = len(road_candidates)
        total_pixels = rows * rows
        road_percentage = (road_count / total_pixels) * 100
        
        print(f"\n=== ANÁLISIS RÁPIDO ===")
        print(f"Píxeles detectados como carretera: {road_count}/{total_pixels} ({road_percentage:.1f}%)")
        
        # Mostrar solo los 5 colores más frecuentes
        sorted_colors = sorted(color_histogram.items(), key=lambda x: x[1], reverse=True)
        print("Top 5 colores más frecuentes:")
        for i, (color_hex, count) in enumerate(sorted_colors[:5]):
            r = int(color_hex[0:2], 16)
            g = int(color_hex[2:4], 16) 
            b = int(color_hex[4:6], 16)
            percentage = (count / total_pixels) * 100
            is_road = "SÍ" if is_road_color_improved((r, g, b)) else "NO"
            print(f"  #{color_hex} RGB({r:3d},{g:3d},{b:3d}) - {percentage:4.1f}% - Carretera: {is_road}")
        
    except Exception as e:
        print(f"No se pudo analizar la imagen: {e}")

def draw_rect_alpha(surface, color, rect, alpha):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    r, g, b = color
    s.fill((r, g, b, alpha))
    surface.blit(s, (rect[0], rect[1]))

# ------------------------------
# CELDA
# ------------------------------
class Cell:
    def __init__(self, row, col, cell_px):
        self.row = row
        self.col = col
        self.x = col * cell_px     # IMPORTANTE: x usa columna
        self.y = row * cell_px     # IMPORTANTE: y usa fila
        self.w = cell_px

        # Mapa base (derivado de la imagen)
        self.walkable = True  # True si es carretera
        self.original_color = None  # Color original del píxel
        self.closed_by_user = False  # Nueva propiedad para marcar celdas cerradas por el usuario
        # Overlays del algoritmo (se dibujan semitransparentes encima)
        self.overlay = None   # None o color RGB
        self.alpha = 0        # 0..255

    def set_walkable(self, ok: bool, original_color=None):
        self.walkable = ok
        self.original_color = original_color
        # Si se marca como no transitable por el usuario, establecer la bandera
        if not ok and original_color is None:
            self.closed_by_user = True

    # Métodos de overlay (no cambian walkable)
    def clear_overlay(self):
        self.overlay = None
        self.alpha = 0

    def make_start(self):
        self.overlay = ORANGE
        self.alpha = 255

    def make_end(self):
        self.overlay = CYAN
        self.alpha = 255

    def make_open(self):
        self.overlay = GREEN
        self.alpha = 110

    def make_closed(self):
        self.overlay = RED
        self.alpha = 90

    def make_path(self):
        self.overlay = PURPLE
        self.alpha = 200

    def draw(self, win):
        # Dibujar fondo para celdas cerradas por el usuario
        if self.closed_by_user:
            draw_rect_alpha(win, CLOSED_ROAD, (self.x, self.y, self.w, self.w), 200)
        
        # Dibujar overlays
        if self.overlay is not None and self.alpha > 0:
            draw_rect_alpha(win, self.overlay, (self.x, self.y, self.w, self.w), self.alpha)

    def __lt__(self, other):
        return False

# ------------------------------
# GRID / IMAGEN
# ------------------------------
def load_background_image(path, target_px):
    try:
        img = pygame.image.load(path)
        img = pygame.transform.smoothscale(img, (target_px, target_px))
        return img
    except Exception as e:
        print(f"[WARN] No se pudo cargar la imagen '{path}': {e}")
        return None

def build_grid_from_image(rows, width_px, image_path):
    # Creamos la grilla
    cell_px = width_px // rows
    grid = [[Cell(r, c, cell_px) for c in range(rows)] for r in range(rows)]

    # Analizamos la imagen antes de procesarla
    analyze_image_colors(image_path, rows)

    # Leemos imagen con PIL y reducimos al tamaño del grid usando NEAREST
    try:
        pil_img = Image.open(image_path).convert("RGB")
        pil_img = pil_img.resize((rows, rows), Image.NEAREST)
        pixels = pil_img.load()
    except Exception as e:
        print(f"[WARN] No se pudo abrir '{image_path}'. Se asumirá todo NO caminable. Error: {e}")
        for r in range(rows):
            for c in range(rows):
                grid[r][c].set_walkable(False)
        return grid

    # Estadísticas de detección
    road_count = 0
    total_count = rows * rows

    # Mapeo: (x=j, y=i) => (col, fila)
    for i in range(rows):
        for j in range(rows):
            rgb = pixels[j, i]
            is_road = is_road_color_improved(rgb)
            grid[i][j].set_walkable(is_road, rgb)
            if is_road:
                road_count += 1

    road_percentage = (road_count / total_count) * 100
    print(f"Detección final: {road_count}/{total_count} celdas transitables ({road_percentage:.1f}%)\n")

    return grid

def draw_grid_lines(win, rows, width_px):
    gap = width_px // rows
    # líneas horizontales
    for i in range(rows + 1):
        pygame.draw.line(win, GRID_COLOR, (0, i * gap), (width_px, i * gap), 1)
    # líneas verticales
    for j in range(rows + 1):
        pygame.draw.line(win, GRID_COLOR, (j * gap, 0), (j * gap, width_px), 1)

def draw_footer(win, width_px, footer_h, font):
    global path_status
    # Fondo
    pygame.draw.rect(win, FOOTER_BG, (0, width_px, width_px, footer_h))
    
    # Línea de controles
    controls = "Izq: Inicio | Der: Fin | Shift+Izq: Cerrar ruta | ESPACIO: A* | R: Recargar | D: Debug | ESC: Salir"
    
    # Línea de estado del camino
    status_line = f"Estado: {path_status}"
    
    # Dibujar texto
    y_offset = width_px + 8
    
    # Controles
    surf = font.render(controls, True, FOOTER_TEXT)
    win.blit(surf, (8, y_offset))
    y_offset += surf.get_height() + 4
    
    # Estado
    surf = font.render(status_line, True, FOOTER_TEXT)
    win.blit(surf, (8, y_offset))

def draw_everything(win, background, grid, rows, width_px, footer_h, font, debug_mode=False):
    # fondo (imagen)
    if background:
        win.blit(background, (0, 0))
    else:
        win.fill(WHITE)

    # Modo debug: mostrar celdas no transitables originales
    if debug_mode:
        for row in grid:
            for cell in row:
                if not cell.walkable and not cell.closed_by_user:
                    draw_rect_alpha(win, RED, (cell.x, cell.y, cell.w, cell.w), 60)

    # Dibujar celdas (incluyendo las cerradas por el usuario)
    for row in grid:
        for cell in row:
            cell.draw(win)

    # cuadrícula
    draw_grid_lines(win, rows, width_px)

    # barra inferior
    draw_footer(win, width_px, footer_h, font)

    pygame.display.update()

# ------------------------------
# A* (8 direcciones, coste real)
# ------------------------------
def heuristic(p1, p2):
    # Euclídea para combinar con diagonales
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x1 - x2, y1 - y2)

def neighbors_8(grid, node):
    rows = len(grid)
    r, c = node.row, node.col
    res = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < rows and grid[nr][nc].walkable:
                res.append(grid[nr][nc])
    return res

def run_astar(draw_fn, grid, start, end):
    global path_status
    path_status = "Buscando camino..."
    
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}

    g_score = {cell: float("inf") for row in grid for cell in row}
    f_score = {cell: float("inf") for row in grid for cell in row}
    g_score[start] = 0.0
    f_score[start] = heuristic((start.row, start.col), (end.row, end.col))

    open_hash = {start}

    while not open_set.empty():
        # permitimos cerrar ventana durante cálculo
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        current = open_set.get()[2]
        open_hash.remove(current)

        if current == end:
            # reconstrucción del camino
            cur = end
            path_length = 0
            while cur in came_from:
                prev = came_from[cur]
                # Calcular distancia real
                path_length += math.hypot(cur.row - prev.row, cur.col - prev.col)
                cur = prev
                if cur not in (start, end):
                    cur.make_path()
                    draw_fn()
            start.make_start()
            end.make_end()
            path_status = f"¡CAMINO ENCONTRADO! Longitud: {path_length:.1f} unidades"
            print(f"¡Camino encontrado! Longitud: {path_length:.2f} unidades")
            return True

        # expandir
        for nb in neighbors_8(grid, current):
            # coste real (diagonal = sqrt(2), ortogonal = 1)
            step = math.hypot(nb.row - current.row, nb.col - current.col)
            tentative_g = g_score[current] + step

            if tentative_g < g_score[nb]:
                came_from[nb] = current
                g_score[nb] = tentative_g
                f_score[nb] = tentative_g + heuristic((nb.row, nb.col), (end.row, end.col))
                if nb not in open_hash:
                    count += 1
                    open_set.put((f_score[nb], count, nb))
                    open_hash.add(nb)
                    if nb not in (start, end):
                        nb.make_open()

        if current not in (start, end):
            current.make_closed()

        draw_fn()

    path_status = "NO SE ENCONTRÓ CAMINO - Verifica que los puntos estén conectados"
    print("No se encontró camino posible")
    return False

# ------------------------------
# INPUT / POS
# ------------------------------
def get_cell_from_mouse(pos, rows, width_px):
    x, y = pos
    if y >= width_px or x >= width_px or x < 0 or y < 0:
        return None  # clic en la barra inferior o fuera
    gap = width_px // rows
    col = x // gap
    row = y // gap
    return (row, col)

# ------------------------------
# MAIN
# ------------------------------
def main():
    global path_status
    pygame.init()
    pygame.display.set_caption("A* Mejorado - Detección Avanzada de Carreteras")
    win = pygame.display.set_mode((WINDOW, WINDOW + FOOTER_H))
    font = pygame.font.SysFont("Arial", 12)

    # Imagen de fondo
    img_path = os.path.join(os.getcwd(), IMG_FILENAME)
    background = load_background_image(img_path, WINDOW)

    # Grid derivado de la imagen (walkable si es carretera detectada)
    grid = build_grid_from_image(GRID_SIZE, WINDOW, img_path)

    start = None
    end = None
    running = True
    solving = False
    debug_mode = False

    def redraw():
        draw_everything(win, background, grid, GRID_SIZE, WINDOW, FOOTER_H, font, debug_mode)

    print("=== CONTROLES ===")
    print("Clic izquierdo: Establecer punto de inicio")
    print("Clic derecho: Establecer punto de destino")
    print("Shift + Clic izquierdo: Cerrar ruta")
    print("ESPACIO: Ejecutar A* | R: Recargar | D: Debug | ESC: Salir")

    while running:
        redraw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Ejecutar A*
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not solving:
                if start and end:
                    if not start.walkable or not end.walkable:
                        path_status = "ERROR: Inicio o destino en zona no transitable"
                        if start and not start.walkable:
                            start.make_closed()
                        if end and not end.walkable:
                            end.make_closed()
                        redraw()
                    else:
                        solving = True
                        # limpiar overlays previos excepto inicio/fin
                        for row in grid:
                            for cell in row:
                                if cell not in (start, end):
                                    cell.clear_overlay()
                        start.make_start()
                        end.make_end()
                        run_astar(redraw, grid, start, end)
                        solving = False
                else:
                    path_status = "Establece punto de inicio y destino"

            # Recargar imagen
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r and not solving:
                start = None
                end = None
                path_status = "Imagen recargada"
                background = load_background_image(img_path, WINDOW)
                grid = build_grid_from_image(GRID_SIZE, WINDOW, img_path)

            # Toggle debug mode
            if event.type == pygame.KEYDOWN and event.key == pygame.K_d and not solving:
                debug_mode = not debug_mode

            # Salir con ESC
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            # Clic izquierdo
            if pygame.mouse.get_pressed()[0] and not solving:
                cell_rc = get_cell_from_mouse(pygame.mouse.get_pos(), GRID_SIZE, WINDOW)
                if cell_rc:
                    r, c = cell_rc
                    cell = grid[r][c]
                    
                    # Verificar si Shift está presionado (cerrar ruta)
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                        cell.set_walkable(False)
                        cell.clear_overlay()
                        path_status = f"Ruta cerrada en ({r}, {c})"
                    else:
                        # Establecer punto de inicio
                        if start and start is not cell:
                            start.clear_overlay()
                        start = cell
                        start.make_start()
                        path_status = f"Inicio en ({r}, {c})"

            # Clic derecho -> FIN
            if pygame.mouse.get_pressed()[2] and not solving:
                cell_rc = get_cell_from_mouse(pygame.mouse.get_pos(), GRID_SIZE, WINDOW)
                if cell_rc:
                    r, c = cell_rc
                    cell = grid[r][c]
                    if end and end is not cell:
                        end.clear_overlay()
                    end = cell
                    end.make_end()
                    path_status = f"Destino en ({r}, {c})"

    pygame.quit()

if __name__ == "__main__":
    main()