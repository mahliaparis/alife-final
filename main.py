import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame

# ================================================================
# Evolving Pheromone Foraging Robots
# ------------------------------------------------
# Standalone Artificial Life final project
# - Ant differential-drive robots (alife-sim style bodies)
# - Pheromone heatmap
# - GA evolves behavior from random search to
#   efficient pheromone-guided foraging
# - Live visualization in pygame
# ================================================================

# -----------------------------
# tunable project settings
# -----------------------------
WINDOW_W = 1100
WINDOW_H = 760
WORLD_W = 820
WORLD_H = 720
PANEL_X = WORLD_W
FPS = 45

CELL = 8
GRID_W = WORLD_W // CELL
GRID_H = WORLD_H // CELL

N_ANTS = 12
N_FOOD_PATCHES = 3
FOOD_PER_PATCH = 20
NEST_POS = np.array([WORLD_W * 0.50, WORLD_H * 0.50], dtype=float)
NEST_RADIUS = 24

POP_SIZE = 6
ELITES = 2
MUTATION_RATE = 0.16
MUTATION_SCALE = 0.12
GENERATIONS = 32
STEPS_PER_EVAL = 700
BEST_REPLAY_STEPS = 700

# Multiple trials make the GA more stable.
EVAL_TRIALS = 2

# Rendering / simulation switches
DRAW_SENSOR_LINES = False
DRAW_TEXT = True
SAVE_PLOT_AT_END = True
FAST_RENDER_SKIP = 2
N_FIXED_LAYOUTS = 2

# Reproducibility
RANDOM_SEED = 11
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------------
# Colors
# -----------------------------
BG = (10, 12, 18)
WORLD_BG = (20, 22, 30)
GRID_COLOR = (28, 31, 42)
WHITE = (240, 240, 240)
LIGHT = (190, 200, 220)
YELLOW = (255, 215, 0)
RED = (230, 85, 85)
GREEN = (90, 220, 120)
CYAN = (50, 210, 235)
MAGENTA = (240, 120, 255)
NEST_COLOR = (100, 140, 255)
FOOD_COLOR = (70, 240, 130)
ROBOT_BODY = (240, 180, 70)
ROBOT_CARRY = (255, 90, 90)
PANEL_BG = (16, 18, 25)

# -----------------------------
# Helper math
# -----------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def vec_from_angle(theta):
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)


def to_grid(pos: np.ndarray) -> Tuple[int, int]:
    gx = int(clamp(pos[0] / CELL, 0, GRID_W - 1))
    gy = int(clamp(pos[1] / CELL, 0, GRID_H - 1))
    return gx, gy


@dataclass
class Genome:
    # Genome parameters intentionally map to visible behavior.
    speed: float                # forward movement speed
    turn_gain: float            # how strongly sensor imbalance causes turning
    sensor_distance: float      # how far ahead the left/right sensors are
    sensor_angle: float         # how wide the sensors are spread from heading
    random_turn: float          # movement randomness
    pheromone_follow: float     # attraction to pheromone while searching
    pheromone_deposit: float    # amount dropped while carrying food home
    evaporation_resist: float   # makes stronger trails persist longer in effect
    food_bias: float            # attraction to nearby food
    nest_bias: float            # attraction to nest when carrying

    def clipped(self):
        self.speed = clamp(self.speed, 1.0, 3.8)
        self.turn_gain = clamp(self.turn_gain, 0.02, 0.45)
        self.sensor_distance = clamp(self.sensor_distance, 8.0, 28.0)
        self.sensor_angle = clamp(self.sensor_angle, 0.15, 1.20)
        self.random_turn = clamp(self.random_turn, 0.0, 0.35)
        self.pheromone_follow = clamp(self.pheromone_follow, 0.0, 3.0)
        self.pheromone_deposit = clamp(self.pheromone_deposit, 0.0, 3.8)
        self.evaporation_resist = clamp(self.evaporation_resist, 0.55, 1.65)
        self.food_bias = clamp(self.food_bias, 0.0, 3.0)
        self.nest_bias = clamp(self.nest_bias, 0.0, 4.0)
        return self

    def copy(self):
        return Genome(**self.__dict__)

    def values(self):
        return list(self.__dict__.values())


def random_genome() -> Genome:
    return Genome(
        speed=random.uniform(1.2, 3.2),
        turn_gain=random.uniform(0.05, 0.30),
        sensor_distance=random.uniform(10.0, 24.0),
        sensor_angle=random.uniform(0.2, 1.0),
        random_turn=random.uniform(0.05, 0.28),
        pheromone_follow=random.uniform(0.0, 2.0),
        pheromone_deposit=random.uniform(0.4, 2.7),
        evaporation_resist=random.uniform(0.7, 1.45),
        food_bias=random.uniform(0.2, 2.4),
        nest_bias=random.uniform(0.6, 3.4),
    ).clipped()


def crossover(a: Genome, b: Genome) -> Genome:
    vals = {}
    for k in a.__dict__.keys():
        vals[k] = getattr(a, k) if random.random() < 0.5 else getattr(b, k)
    return Genome(**vals).clipped()


def mutate(g: Genome) -> Genome:
    h = g.copy()
    for k in h.__dict__.keys():
        if random.random() < MUTATION_RATE:
            value = getattr(h, k)
            if k in ("sensor_distance",):
                scale = 2.8
            elif k in ("sensor_angle", "turn_gain", "random_turn"):
                scale = 0.08
            else:
                scale = MUTATION_SCALE * max(abs(value), 0.6)
            setattr(h, k, value + random.gauss(0, scale))
    return h.clipped()


class FoodPatch:
    def __init__(self, x: float, y: float, amount: int):
        self.pos = np.array([x, y], dtype=float)
        self.amount = amount
        self.radius = 18

    def has_food(self):
        return self.amount > 0


class AntRobot:
    def __init__(self, genome: Genome):
        angle = random.uniform(-math.pi, math.pi)
        r = random.uniform(0, NEST_RADIUS * 0.7)
        t = random.uniform(-math.pi, math.pi)
        offset = np.array([math.cos(t), math.sin(t)], dtype=float) * r
        self.pos = NEST_POS.copy() + offset
        self.angle = angle
        self.genome = genome
        self.carrying = False
        self.radius = 7
        self.left_wheel = 0.0
        self.right_wheel = 0.0

    def sensor_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        forward = vec_from_angle(self.angle)
        left_dir = vec_from_angle(self.angle - self.genome.sensor_angle)
        right_dir = vec_from_angle(self.angle + self.genome.sensor_angle)
        left_sensor = self.pos + left_dir * self.genome.sensor_distance + forward * 2.5
        right_sensor = self.pos + right_dir * self.genome.sensor_distance + forward * 2.5
        return left_sensor, right_sensor


class World:
    def __init__(self, genome: Genome, food_layout=None, visualize=False):
        self.genome = genome
        self.visualize = visualize
        self.pheromone = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.ants = [AntRobot(genome) for _ in range(N_ANTS)]
        self.food = self._make_food(food_layout)
        self.food_returned = 0
        self.steps = 0

    def _make_food(self, food_layout=None):
        if food_layout is not None:
            return [FoodPatch(x, y, amount) for x, y, amount in food_layout]
        food = []
        margin = 75
        min_from_nest = 170
        while len(food) < N_FOOD_PATCHES:
            x = random.uniform(margin, WORLD_W - margin)
            y = random.uniform(margin, WORLD_H - margin)
            if np.linalg.norm(np.array([x, y]) - NEST_POS) > min_from_nest:
                food.append(FoodPatch(x, y, FOOD_PER_PATCH))
        return food

    def export_food_layout(self):
        return [(fp.pos[0], fp.pos[1], fp.amount) for fp in self.food]

    def sample_pheromone(self, pos: np.ndarray) -> float:
        gx, gy = to_grid(pos)
        total = 0.0
        count = 0
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                nx = clamp(gx + ox, 0, GRID_W - 1)
                ny = clamp(gy + oy, 0, GRID_H - 1)
                total += self.pheromone[int(ny), int(nx)]
                count += 1
        return total / max(count, 1)

    def nearest_food_vector(self, pos: np.ndarray, max_dist=70.0):
        best_vec = None
        best_d = 1e9
        for fp in self.food:
            if not fp.has_food():
                continue
            vec = fp.pos - pos
            d = float(np.linalg.norm(vec))
            if d < best_d and d < max_dist:
                best_d = d
                best_vec = vec / (d + 1e-6)
        return best_vec

    def step(self):
        self.steps += 1
        g = self.genome

        # Global pheromone evaporation.
        evap = 0.992 + (g.evaporation_resist - 1.0) * 0.003
        evap = clamp(evap, 0.986, 0.996)
        self.pheromone *= evap

        for ant in self.ants:
            left_sensor, right_sensor = ant.sensor_positions()
            left_ph = self.sample_pheromone(left_sensor)
            right_ph = self.sample_pheromone(right_sensor)
            ph_bias = (right_ph - left_ph) * g.turn_gain * g.pheromone_follow

            # Food attraction only while searching.
            food_turn = 0.0
            if not ant.carrying:
                food_vec = self.nearest_food_vector(ant.pos)
                if food_vec is not None:
                    desired = math.atan2(food_vec[1], food_vec[0])
                    delta = wrap_angle(desired - ant.angle)
                    food_turn = clamp(delta * 0.14 * g.food_bias, -0.35, 0.35)

            # Nest homing while carrying food.
            nest_turn = 0.0
            if ant.carrying:
                home_vec = NEST_POS - ant.pos
                desired = math.atan2(home_vec[1], home_vec[0])
                delta = wrap_angle(desired - ant.angle)
                nest_turn = clamp(delta * 0.13 * g.nest_bias, -0.42, 0.42)

            jitter = random.uniform(-g.random_turn, g.random_turn)
            turn = ph_bias + food_turn + nest_turn + jitter
            turn = clamp(turn, -0.50, 0.50)

            # Differential-drive style movement.
            base = g.speed
            ant.left_wheel = clamp(base - turn * 3.2, 0.3, 4.5)
            ant.right_wheel = clamp(base + turn * 3.2, 0.3, 4.5)
            forward_speed = (ant.left_wheel + ant.right_wheel) * 0.5
            angular_speed = (ant.right_wheel - ant.left_wheel) * 0.09
            ant.angle = wrap_angle(ant.angle + angular_speed)
            ant.pos += vec_from_angle(ant.angle) * forward_speed

            # Bounce off world borders.
            bounced = False
            if ant.pos[0] < ant.radius:
                ant.pos[0] = ant.radius
                bounced = True
            elif ant.pos[0] > WORLD_W - ant.radius:
                ant.pos[0] = WORLD_W - ant.radius
                bounced = True
            if ant.pos[1] < ant.radius:
                ant.pos[1] = ant.radius
                bounced = True
            elif ant.pos[1] > WORLD_H - ant.radius:
                ant.pos[1] = WORLD_H - ant.radius
                bounced = True
            if bounced:
                ant.angle = wrap_angle(ant.angle + random.uniform(1.8, 2.6))

            # Pick up food.
            if not ant.carrying:
                for fp in self.food:
                    if fp.has_food() and np.linalg.norm(ant.pos - fp.pos) <= fp.radius:
                        fp.amount -= 1
                        ant.carrying = True
                        break

            # Deposit pheromone while carrying food home.
            if ant.carrying:
                gx, gy = to_grid(ant.pos)
                self.pheromone[gy, gx] += 0.85 * g.pheromone_deposit
                if np.linalg.norm(ant.pos - NEST_POS) <= NEST_RADIUS:
                    ant.carrying = False
                    self.food_returned += 1

        np.clip(self.pheromone, 0.0, 255.0, out=self.pheromone)

    def fitness(self):
        # Strongly reward collected food, lightly reward trail richness.
        trail_bonus = float(np.mean(self.pheromone)) * 0.02
        return self.food_returned + trail_bonus


class GeneticTrainer:
    def __init__(self):
        self.population = [random_genome() for _ in range(POP_SIZE)]
        self.best_history = []
        self.avg_history = []
        self.global_best = None
        self.global_best_score = -1e9
        self.best_food_layout = None
        self.fixed_layouts = [World(random_genome()).export_food_layout() for _ in range(N_FIXED_LAYOUTS)]

    def evaluate_genome(self, genome: Genome) -> float:
        scores = []
        for trial in range(EVAL_TRIALS):
            layout = self.fixed_layouts[trial % len(self.fixed_layouts)]
            world = World(genome, food_layout=layout)
            for _ in range(STEPS_PER_EVAL):
                world.step()
            scores.append(world.fitness())
        return float(sum(scores) / len(scores))

    def next_generation(self):
        scored = []
        for genome in self.population:
            score = self.evaluate_genome(genome)
            scored.append((score, genome))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_genome = scored[0]
        avg_score = sum(s for s, _ in scored) / len(scored)
        self.best_history.append(best_score)
        self.avg_history.append(avg_score)

        if best_score > self.global_best_score:
            self.global_best_score = best_score
            self.global_best = best_genome.copy()

            # Freeze one world layout for replay with best genome.
            replay_world = World(self.global_best)
            self.best_food_layout = replay_world.export_food_layout()

        elites = [g.copy() for _, g in scored[:ELITES]]
        parents = [g for _, g in scored[: max(ELITES + 4, POP_SIZE // 2)]]

        new_pop = elites[:]
        while len(new_pop) < POP_SIZE:
            a, b = random.sample(parents, 2)
            child = crossover(a, b)
            child = mutate(child)
            new_pop.append(child)
        self.population = new_pop
        return best_score, avg_score, best_genome.copy()


def draw_world(screen, world: World, font, generation, phase, best_score, avg_score, step_i, total_steps):
    screen.fill(BG)
    pygame.draw.rect(screen, WORLD_BG, (0, 0, WORLD_W, WORLD_H))
    pygame.draw.rect(screen, PANEL_BG, (PANEL_X, 0, WINDOW_W - PANEL_X, WINDOW_H))

    # Pheromone heatmap
    ph = world.pheromone
    if np.max(ph) > 0:
        surf = pygame.Surface((GRID_W, GRID_H))
        arr = np.zeros((GRID_W, GRID_H, 3), dtype=np.uint8)
        norm = np.clip((ph / (np.max(ph) + 1e-6)) ** 0.7, 0, 1)
        arr[:, :, 0] = (norm.T * 70).astype(np.uint8)
        arr[:, :, 1] = (norm.T * 210).astype(np.uint8)
        arr[:, :, 2] = (norm.T * 255).astype(np.uint8)
        pygame.surfarray.blit_array(surf, arr)
        surf = pygame.transform.scale(surf, (WORLD_W, WORLD_H))
        surf.set_alpha(140)
        screen.blit(surf, (0, 0))

    # Nest
    pygame.draw.circle(screen, NEST_COLOR, NEST_POS.astype(int), NEST_RADIUS)
    pygame.draw.circle(screen, (160, 190, 255), NEST_POS.astype(int), NEST_RADIUS, 2)

    # Food
    for fp in world.food:
        if fp.amount <= 0:
            continue
        intensity = clamp(fp.amount / FOOD_PER_PATCH, 0.15, 1.0)
        color = (int(50 + 60 * intensity), int(120 + 120 * intensity), int(60 + 90 * intensity))
        pygame.draw.circle(screen, color, fp.pos.astype(int), fp.radius)
        txt = font.render(str(fp.amount), True, (10, 20, 15))
        screen.blit(txt, (fp.pos[0] - 8, fp.pos[1] - 8))

    # Ant robots
    for ant in world.ants:
        body_color = ROBOT_CARRY if ant.carrying else ROBOT_BODY
        pos = ant.pos.astype(int)
        pygame.draw.circle(screen, body_color, pos, ant.radius)
        nose = ant.pos + vec_from_angle(ant.angle) * (ant.radius + 5)
        pygame.draw.line(screen, WHITE, pos, nose.astype(int), 2)

        # tiny wheel marks so they feel robotic
        left_dir = vec_from_angle(ant.angle - math.pi / 2)
        w1 = ant.pos + left_dir * 4
        w2 = ant.pos - left_dir * 4
        pygame.draw.circle(screen, (55, 55, 60), w1.astype(int), 2)
        pygame.draw.circle(screen, (55, 55, 60), w2.astype(int), 2)

        if DRAW_SENSOR_LINES:
            ls, rs = ant.sensor_positions()
            pygame.draw.line(screen, CYAN, pos, ls.astype(int), 1)
            pygame.draw.line(screen, MAGENTA, pos, rs.astype(int), 1)

    # Side panel
    y = 20
    def write(line, color=WHITE, big=False):
        nonlocal y
        f = pygame.font.SysFont("consolas", 24 if big else 18)
        s = f.render(line, True, color)
        screen.blit(s, (PANEL_X + 20, y))
        y += 28 if big else 24

    write("Evolving Pheromone Robots", YELLOW, big=True)
    write(f"Phase: {phase}")
    write(f"Generation: {generation + 1}/{GENERATIONS}")
    write(f"Step: {step_i}/{total_steps}")
    write(f"Food returned: {world.food_returned}", GREEN)
    write(f"Best fitness: {best_score:.2f}")
    write(f"Avg fitness:  {avg_score:.2f}")
    y += 8
    write("Best genome", YELLOW)
    g = world.genome
    for key, val in g.__dict__.items():
        write(f"{key[:14]:14s}: {val:>5.2f}", LIGHT)

    y += 12
    write("How to read it", YELLOW)
    write("Blue circle = nest", LIGHT)
    write("Green circles = food", LIGHT)
    write("Bright cyan = pheromone", LIGHT)
    write("Orange robots = searching", LIGHT)
    write("Red robots = carrying food", LIGHT)
    write("Later generations build", LIGHT)
    write("clear highways to food.", LIGHT)

    pygame.display.flip()


def save_plot(trainer: GeneticTrainer, out_path="fitness_curve.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(trainer.best_history, label="Best fitness")
    plt.plot(trainer.avg_history, label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolution of pheromone-guided foraging")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    pygame.init()
    pygame.display.set_caption("Artificial Life Final Project - Evolving Pheromone Robots")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    trainer = GeneticTrainer()
    running = True

    # Training loop: for each generation, evaluate all genomes without rendering,
    # then render the best genome on a frozen world layout so the audience can
    # visibly see progression generation by generation.
    last_best = None
    last_best_score = 0.0
    last_avg = 0.0

    for gen in range(GENERATIONS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not running:
            break

        best_score, avg_score, best_genome = trainer.next_generation()
        last_best = best_genome
        last_best_score = best_score
        last_avg = avg_score

        # Replay the best genome this generation on a fixed layout.
        world = World(best_genome, food_layout=trainer.best_food_layout, visualize=True)
        for step_i in range(1, BEST_REPLAY_STEPS + 1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not running:
                break
            world.step()
            if step_i % FAST_RENDER_SKIP == 0 or step_i == 1 or step_i == BEST_REPLAY_STEPS:
                draw_world(
                    screen,
                    world,
                    font,
                    gen,
                    phase="training replay",
                    best_score=best_score,
                    avg_score=avg_score,
                    step_i=step_i,
                    total_steps=BEST_REPLAY_STEPS,
                )
                clock.tick(FPS)
        if not running:
            break

    # Final infinite replay of best genome discovered.
    if running and trainer.global_best is not None:
        final_world = World(trainer.global_best, food_layout=trainer.best_food_layout, visualize=True)
        if SAVE_PLOT_AT_END:
            save_plot(trainer, os.path.join(os.path.dirname(__file__), "fitness_curve.png"))

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            final_world.step()
            if final_world.steps % FAST_RENDER_SKIP == 0 or final_world.steps == 1:
                draw_world(
                    screen,
                    final_world,
                    font,
                    GENERATIONS - 1,
                    phase="final best replay",
                    best_score=trainer.global_best_score,
                    avg_score=trainer.avg_history[-1] if trainer.avg_history else 0.0,
                    step_i=final_world.steps,
                    total_steps=999999,
                )
                clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
