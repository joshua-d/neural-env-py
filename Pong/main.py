import pygame
import neural_temp
from pong import *


fps = 60


def create_games(nt, game_list):

    indices = []
    scrambled_networks = []

    for i in range(0, nt.pop_size):
        indices.append(i)

    for i in range(0, nt.pop_size):
        index = random.randint(0, len(indices) - 1)
        scrambled_networks.append(nt.networks[indices[index]])
        indices.pop(index)

    i = 0
    while True:
        if i > nt.pop_size - 2:
            break
        game_list.append(Game(nt, [scrambled_networks[i], scrambled_networks[i + 1]]))
        i += 2


def main():
    pygame.init()
    pygame.display.set_caption("Pong - Neural Temp")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    nt = neural_temp.NeuralTemp(200, 5, 3, 1, 1)

    games = []
    create_games(nt, games)

    current_frame = 0
    running = True

    while running:
        start_tick = pygame.time.get_ticks()
        pygame.event.get()
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), (SCREEN_OFFSET, SCREEN_OFFSET, X_MAX, Y_MAX), 2)

        if current_frame == 1200:
            nt.reproduce()
            print(nt.networks[0].fitness)
            games = []
            create_games(nt, games)
            current_frame = 0

        for game in games:
            game.update(current_frame, 6)
            if game.networks[0] == nt.networks[0] or game.networks[1] == nt.networks[0]:
                game.draw(pygame, screen)

        current_frame += 1
        pygame.display.flip()
        pygame.time.wait(math.floor(1000 / fps) - (pygame.time.get_ticks() - start_tick))


if __name__ == "__main__":
    main()
