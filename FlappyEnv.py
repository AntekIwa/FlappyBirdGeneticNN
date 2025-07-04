import pygame
import random
import numpy as np

# --- Game settings ---
WIDTH, HEIGHT = 400, 600
PIPE_WIDTH = 60
GRAVITY = 0.5
JUMP_STRENGTH = -8
FPS = 60

class Bird:
    def __init__(self, net):
        self.x = 100
        self.y = HEIGHT // 2
        self.radius = 12
        self.velocity = 0
        self.network = net
        self.alive = True
        self.score = 0
        self.frames_alive = 0
        self.color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255)
        )

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        self.frames_alive += 1

    def jump(self):
        self.velocity = JUMP_STRENGTH

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius,
                           self.radius * 2, self.radius * 2)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randint(100, HEIGHT - 200)
        self.pipe_gap = random.randint(100, 300)

    def update(self):
        self.x -= 3

    def get_top_bottom(self):
        return self.height, self.height + self.pipe_gap

    def collides_with(self, bird: Bird):
        rect = bird.get_rect()
        top, bottom = self.get_top_bottom()
        top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, top)
        bottom_rect = pygame.Rect(self.x, bottom, PIPE_WIDTH, HEIGHT)
        return rect.colliderect(top_rect) or rect.colliderect(bottom_rect)

class FlappyEnvPopulation:
    def __init__(self, networks, render=False, gen=0):
        self.networks = networks
        self.birds = [Bird(net) for net in networks]
        self.pipes = []
        x = 400
        for _ in range(3):
            self.pipes.append(Pipe(x))
            x += random.randint(180, 260)
        self.frame = 0
        self.render = render
        self.gen = gen

        if render:
            pygame.init()
            self.win = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24)

    def get_next_pipe(self, bird):
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH > bird.x:
                return pipe
        return self.pipes[0]

    def step(self):
        if self.render:
            self.clock.tick(FPS)
            self.win.fill((0, 0, 0))

        for pipe in self.pipes:
            pipe.update()

        if self.pipes[0].x < -PIPE_WIDTH: #old pipe go out of screen, lets generate new one
            self.pipes.pop(0)
            next_distance = random.randint(150, 300)
            self.pipes.append(Pipe(self.pipes[-1].x + next_distance))

        all_dead = True

        for bird in self.birds:
            if not bird.alive:
                continue

            all_dead = False
            bird.update()

            pipe = self.get_next_pipe(bird)
            top, bottom = pipe.get_top_bottom()
            #INPUTS for our NN are: birs.y posision, birds velocity, distance on x-axis to pipe, pitagoras distance to top of gap and to bottom of gap,
            #distance to bottom and top of gapp (in y-axis), top and bottom of gap
            inputs = np.array([
                bird.y / HEIGHT,
                bird.velocity / 10.0,
                (pipe.x - bird.x) / WIDTH,
                ((pipe.x - bird.x)/WIDTH)**2 + ((top - bird.y)/WIDTH)**2,
                ((pipe.x - bird.x)/WIDTH)**2 + ((bottom - bird.y)/WIDTH)**2,
                bird.y - bottom,
                top - bird.y,
                top / HEIGHT,
                bottom / HEIGHT
            ])

            output = bird.network.propagate(inputs)
            if output[0] > 0.5: #checking if our NN decide to jump
                bird.jump()

            if pipe.collides_with(bird) or bird.y < 0 or bird.y > HEIGHT: #checking if bird went into pipe or exit board
                bird.alive = False

            if pipe.x + PIPE_WIDTH < bird.x and not hasattr(pipe, 'scored'): #checking if bird pass the obstacle
                pipe.scored = True
                bird.score += 1

            if self.render:
                pygame.draw.circle(self.win, bird.color, (int(bird.x), int(bird.y)), bird.radius)

        if self.render:
            for pipe in self.pipes:
                #showing pipes
                pygame.draw.rect(self.win, (0, 255, 0), (pipe.x, 0, PIPE_WIDTH, pipe.height))
                pygame.draw.rect(self.win, (0, 255, 0),
                                 (pipe.x, pipe.height + pipe.pipe_gap, PIPE_WIDTH, HEIGHT))

                # showing acutal gap size
                gap_text = self.font.render(str(pipe.pipe_gap), True, (255, 255, 0))
                self.win.blit(gap_text, (pipe.x + 5, pipe.height + pipe.pipe_gap // 2 - 10))
                
            alive_count = sum(1 for b in self.birds if b.alive)
            text = self.font.render(f"Alive: {alive_count}", True, (255,255,255))
            self.win.blit(text, (WIDTH - 130, 10))
            gen_text = self.font.render(f"Gen: {self.gen}", True, (255, 255, 255))
            self.win.blit(gen_text, (10, 10))
            pygame.display.update()

        self.frame += 1
        return all_dead

    def run(self, max_frames=1000): #to not play for infinity, lets set max score
        while self.frame < max_frames:
            if self.step():
                break

        return [b.score * 1000 + b.frames_alive for b in self.birds] #score for bird(NN)
