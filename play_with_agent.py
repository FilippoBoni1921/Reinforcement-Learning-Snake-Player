import pygame
import random
import numpy as np
import tensorflow as tf
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

import environment_fully_observable 
import environment_partially_observable
from environment_definition_for_training import get_env
from ppo_agent import Agent_PPO

N = 1
ITERATIONS = 1000

env_ = get_env(N)
agent = Agent_PPO(len_actions = 4)
agent.actor.load_weights("ppo_actor_weights.h5")


pygame.init()

# Impostazioni del gioco Snake
GRID_SIZE = 7
GRID_WIDTH = GRID_SIZE
GRID_HEIGHT = GRID_SIZE
CELL_SIZE = 100

# Impostazioni finestra di gioco
window_size = (GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Snake Visualization")

# Impostazioni finestra di score
score_window_size = (200, 100)
score_screen = pygame.Surface(score_window_size)
score_screen.fill((255, 255, 255))

# Colori
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)

# Funzione per disegnare la griglia e i wall
def draw_grid_and_walls():
    for x in range(0, window_size[0], CELL_SIZE):
        pygame.draw.line(screen, WHITE, (x, 0), (x, window_size[1]))
    for y in range(0, window_size[1], CELL_SIZE):
        pygame.draw.line(screen, WHITE, (0, y), (window_size[0], y))

    # Disegna i wall lungo i quattro lati
    pygame.draw.rect(screen, GRAY, (0, 0, window_size[0], CELL_SIZE))  # Prima riga
    pygame.draw.rect(screen, GRAY, (0, 0, CELL_SIZE, window_size[1]))  # Prima colonna
    pygame.draw.rect(screen, GRAY, (0, window_size[1] - CELL_SIZE, window_size[0], CELL_SIZE))  # Ultima riga
    pygame.draw.rect(screen, GRAY, (window_size[0] - CELL_SIZE, 0, CELL_SIZE, window_size[1]))  # Ultima colonna


def draw_score_window(score):
    font = pygame.font.Font(None, 36)
    text = font.render(f"Score: {score}", True, BLACK)
    score_screen.fill((255, 255, 255))
    score_screen.blit(text, (50, 25))
    screen.blit(score_screen, (window_size[0] // 2 - score_window_size[0] // 2, window_size[1] // 2 - score_window_size[1] // 2))


def draw_score_area(rew,score,iteration):
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, WHITE)
    rew_text = font.render(f"Reward: {rew}", True, WHITE)
    iteration_text = font.render(f"Iteration: {iteration}", True, WHITE)
    screen.blit(score_text, (10, 10))
    screen.blit(rew_text, (200, 10))
    screen.blit(iteration_text, (400, 10))

def draw_game_over_message():
    font = pygame.font.Font(None, 100)
    text = font.render("You Win!!!", True, WHITE)
    screen.blit(text, (window_size[0] // 2 - text.get_width() // 2, window_size[1] // 2 - text.get_height() // 2))


# Funzione principale di visualizzazione
def visualize_agent():
    running = True
    clock = pygame.time.Clock()

  
    for iteration in range(ITERATIONS):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Ottenere la board dall'environment
        board = env_.boards[0]

        states = env_.to_state()
        
        probs = agent.actor(states)
        
        actions = tf.random.categorical(tf.math.log(probs), 1, dtype=tf.int32)

        rewards,_,_,_,_ = env_.move(actions)
        length = np.mean([len(sublist)+1 for sublist in env_.bodies])

        screen.fill(BLACK)
        draw_grid_and_walls()

        # Ottenere la posizione della testa, corpo e frutto
        head_position = np.argwhere(board==4)[0]
        body_positions = np.argwhere(board==3)
        fruit_position = np.argwhere(board==2)[0]  

        # Disegna il frutto
        pygame.draw.rect(screen, GREEN, (fruit_position[1] * CELL_SIZE, fruit_position[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Disegna la testa del serpente in rosso
        pygame.draw.rect(screen, RED, (head_position[1] * CELL_SIZE, head_position[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Disegna il corpo del serpente in arancione
        for body_position in body_positions:
            pygame.draw.rect(screen, ORANGE, (body_position[1] * CELL_SIZE, body_position[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        rew = np.round(rewards.numpy()[0][0],1)
        score = length
        draw_score_area(rew,score,iteration)

        game_over = False
        if game_over:
            screen.fill(BLACK)
            draw_game_over_message()
        
        pygame.display.flip()
        clock.tick(5)  # Regola la velocità di visualizzazione

    #pygame.quit()

if __name__ == '__main__':
    visualize_agent()

