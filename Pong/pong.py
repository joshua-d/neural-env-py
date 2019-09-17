import math
import random


SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1200
SCREEN_OFFSET = 50

X_MAX = SCREEN_WIDTH - 2 * SCREEN_OFFSET
Y_MAX = SCREEN_HEIGHT - 2 * SCREEN_OFFSET


def cos(angle):
    return math.cos(angle * math.pi / 180)


def sin(angle):
    return math.sin(angle * math.pi / 180)


class Paddle:

    def __init__(self, game, network, player):

        self.game = game
        self.player = player

        self.network = network

        self.width = 20
        self.height = 100

        self.speed = 10

        self.direction = 0

        if player == 0:
            self.position = [50, Y_MAX / 2 - self.height / 2]
            self.face = self.position[0] + self.width
        else:
            self.position = [X_MAX - 50 - self.width, Y_MAX / 2 - self.height / 2]
            self.face = self.position[0]

    def update(self):
        if self.direction == 1:
            self.position[1] -= self.speed
        elif self.direction == 2:
            self.position[1] += self.speed

        if self.position[1] < 0:
            self.position[1] = 0
        elif self.position[1] + self.height > Y_MAX:
            self.position[1] = Y_MAX - self.height

    def draw(self, pygame, screen):
        pygame.draw.rect(screen, (255, 255, 255), (SCREEN_OFFSET + self.position[0], SCREEN_OFFSET + self.position[1], self.width, self.height))


class Ball:

    def __init__(self, game):

        self.game = game
        self.paddles = game.paddles

        self.size = 20
        self.speed = 10

        self.position = [X_MAX / 2 - self.size / 2, Y_MAX / 2 - self.size / 2]
        self.velocity = [0, 0]
        self.velocity[1] = random.randint(-self.speed / 2, self.speed / 2)
        self.velocity[0] = math.sqrt(self.speed ** 2 - self.velocity[1] ** 2)
        self.direction = False

        self.change_bound = 0.5

    def check_collision(self):

        if self.position[0] <= self.paddles[0].face and self.position[0] > self.paddles[0].position[0] and self.position[1] + self.size >= self.paddles[0].position[1] and self.position[1] <= self.paddles[0].position[1] + self.paddles[0].height:
            self.position[0] = self.paddles[0].face
            self.velocity[0] *= -1

            if self.velocity[0] >= 0:
                sign = 1
            else:
                sign = -1
            # change_bound = abs(self.velocity[1]) / self.volatility_divisor
            old_vel = self.velocity[1]
            self.velocity[1] += random.uniform(-self.change_bound, self.change_bound)
            self.velocity[0] = math.sqrt(abs(self.velocity[0]**2 + old_vel**2 - self.velocity[1]**2)) * sign
            self.game.networks[0].fitness -= 1

        elif self.position[0] + self.size >= self.paddles[1].face and self.position[0] < self.paddles[1].position[0] + self.paddles[1].width and self.position[1] + self.size >= self.paddles[1].position[1] and self.position[1] <= self.paddles[1].position[1] + self.paddles[1].height:
            self.position[0] = self.paddles[1].face - self.size
            self.velocity[0] *= -1
            if self.velocity[0] >= 0:
                sign = 1
            else:
                sign = -1
            # change_bound = abs(self.velocity[1]) / self.volatility_divisor
            old_vel = self.velocity[1]
            self.velocity[1] += random.uniform(-self.change_bound, self.change_bound)
            self.velocity[0] = math.sqrt(abs(self.velocity[0] ** 2 + old_vel ** 2 - self.velocity[1] ** 2)) * sign
            self.game.networks[1].fitness -= 1

        if self.position[1] <= 0:
            self.position[1] = 0
            self.velocity[1] *= -1
            if self.velocity[1] >= 0:
                sign = 1
            else:
                sign = -1
            old_vel = self.velocity[0]
            self.velocity[0] += random.uniform(-self.change_bound, self.change_bound)
            self.velocity[1] = math.sqrt(abs(self.velocity[1] ** 2 + old_vel ** 2 - self.velocity[0] ** 2)) * sign

        elif self.position[1] + self.size >= Y_MAX:
            self.position[1] = Y_MAX - self.size
            self.velocity[1] *= -1
            if self.velocity[1] >= 0:
                sign = 1
            else:
                sign = -1
            old_vel = self.velocity[0]
            self.velocity[0] += random.uniform(-self.change_bound, self.change_bound)
            self.velocity[1] = math.sqrt(abs(self.velocity[1] ** 2 + old_vel ** 2 - self.velocity[0] ** 2)) * sign

        elif self.position[0] <= 0 or self.position[0] + self.size >= X_MAX:
            self.position = [X_MAX / 2 - self.size / 2, Y_MAX / 2 - self.size / 2]
            self.velocity[1] = random.randint(-self.speed / 2, self.speed / 2)
            self.velocity[0] = math.sqrt(self.speed**2 - self.velocity[1] ** 2)
            if not self.direction:
                self.velocity[0] *= -1
                self.direction = True
            else:
                self.direction = False

    def update(self):
        self.update_position()
        self.check_collision()

    def update_position(self):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

    def draw(self, pygame, screen):
        pygame.draw.rect(screen, (255, 255, 255), (SCREEN_OFFSET + self.position[0], SCREEN_OFFSET + self.position[1], self.size, self.size))


class Game:

    def __init__(self, nt, networks):

        self.nt = nt
        self.networks = networks

        self.paddles = [Paddle(self, networks[0], 0), Paddle(self, networks[1], 1)]
        self.ball = Ball(self)

        for network in networks:
            network.fitness = 0

    def get_input_data(self, player):
        if player == 0:
            return [self.ball.position[0] / (X_MAX - self.ball.size), self.ball.position[1] / (Y_MAX - self.ball.size), 0.5 + self.ball.velocity[0] / (self.ball.speed * 2), 0.5 + self.ball.velocity[1] / (self.ball.speed * 2), self.paddles[0].position[1] / (Y_MAX - self.paddles[0].height)]
        else:
            return [1 - self.ball.position[0] / (X_MAX - self.ball.size), self.ball.position[1] / (Y_MAX - self.ball.size), 0.5 + -self.ball.velocity[0] / (self.ball.speed * 2), 0.5 + self.ball.velocity[1] / (self.ball.speed * 2), self.paddles[1].position[1] / (Y_MAX - self.paddles[1].height)]

    def get_decision(self, player):
        self.networks[player].input_data(self.get_input_data(player))

        if self.networks[player].neuron[self.nt.output_layer][0].value >= 0.55:
            return 1
        elif self.networks[player].neuron[self.nt.output_layer][0].value <= 0.45:
            return 2
        else:
            return 0

    def update(self, frame, interval):
        if frame % interval == 0:
            self.paddles[0].direction = self.get_decision(0)
            self.paddles[1].direction = self.get_decision(1)

        for paddle in self.paddles:
            paddle.update()
        self.ball.update()

    def draw(self, pygame, screen):
        for paddle in self.paddles:
            paddle.draw(pygame, screen)
        self.ball.draw(pygame, screen)