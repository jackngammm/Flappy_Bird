import pygame
import numpy as np
import random
import copy
import webbrowser

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Load images (ensure these files are in the same directory as the script)
bird_img = pygame.image.load("C:\VSCODE\Flappy_Bird\Flappy_Bird\images\pngegg.png")
bg_img = pygame.image.load("C:\VSCODE\Flappy_Bird\Flappy_Bird\images\pngegg (2).png")
pipe_top_img = pygame.image.load("C:\VSCODE\Flappy_Bird\Flappy_Bird\images\poletop.png")
pipe_bottom_img = pygame.image.load("C:\VSCODE\Flappy_Bird\Flappy_Bird\images\pngegg (1).png")

class Bird:
    def __init__(self):
        self.x = 80
        self.y = 250
        self.width = 40
        self.height = 30
        self.alive = True
        self.gravity = 0
        self.velocity = 0.3
        self.jump_strength = -6

    def flap(self):
        self.gravity = self.jump_strength

    def update(self):
        self.gravity += self.velocity
        self.y += self.gravity

    def is_dead(self, pipes):
        if self.y >= HEIGHT or self.y + self.height <= 0:
            return True
        for pipe in pipes:
            if (self.x < pipe.x + pipe.width and
                self.x + self.width > pipe.x and
                self.y < pipe.y + pipe.height and
                self.y + self.height > pipe.y):
                return True
        return False

class Pipe:
    def __init__(self, x, y, height, is_bottom):
        self.x = x
        self.y = y
        self.width = 50
        self.height = height
        self.speed = 3
        self.is_bottom = is_bottom

    def update(self):
        self.x -= self.speed

    def is_out(self):
        return self.x + self.width < 0

class Game:
    def __init__(self):
        self.birds = []
        self.pipes = []
        self.score = 0
        self.spawn_interval = 90
        self.interval = 0
        self.background_x = 0

    def start(self, networks):
        self.birds = [Bird() for _ in networks]
        self.pipes = []
        self.score = 0

    def update(self, networks):
        nearest_pipe = None
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.birds[0].x:
                nearest_pipe = pipe
                break

        for i, bird in enumerate(self.birds):
            if bird.alive:
                if nearest_pipe is not None:
                    # Use actual pipe values
                    inputs = [
                        bird.y / HEIGHT,
                        (bird.y - (nearest_pipe.y + nearest_pipe.height)) / HEIGHT,
                        bird.gravity
                    ]
                else:
                    # Use default values if no pipe is available
                    inputs = [bird.y / HEIGHT, 1, bird.gravity]

                output = networks[i].compute(inputs)
                if output[0] > 0.5:
                    bird.flap()

                bird.update()
                if bird.is_dead(self.pipes):
                    bird.alive = False

        self.pipes = [pipe for pipe in self.pipes if not pipe.is_out()]
        if self.interval == 0:
            gap = 150
            pipe_y = random.randint(100, HEIGHT - 100 - gap)
            self.pipes.append(Pipe(WIDTH, 0, pipe_y, False))
            self.pipes.append(Pipe(WIDTH, pipe_y + gap, HEIGHT - pipe_y - gap, True))

        self.interval += 1
        if self.interval == self.spawn_interval:
            self.interval = 0

        self.score += 1


    def display(self):
        screen.blit(bg_img, (0, 0))
        for pipe in self.pipes:
            img = pipe_bottom_img if pipe.is_bottom else pipe_top_img
            screen.blit(img, (pipe.x, pipe.y))

        for bird in self.birds:
            if bird.alive:
                screen.blit(bird_img, (bird.x, bird.y))

        font = pygame.font.SysFont("Arial", 20)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        pygame.display.flip()

class Neuroevolution:
    def __init__(self, options=None):
        self.options = {
            'activation': lambda x: 1 / (1 + np.exp(-x)),
            'randomClamped': lambda: random.uniform(-1, 1),
            'network': [3, [5], 1],
            'population': 50,
            'elitism': 0.2,
            'randomBehaviour': 0.2,
            'mutationRate': 0.1,
            'mutationRange': 0.5,
            'historic': 0,
            'lowHistoric': False,
            'scoreSort': -1,
            'nbChild': 1,
        }
        if options:
            self.options.update(options)

        self.generations = Generations(self)

    def nextGeneration(self):
        if len(self.generations.generations) == 0:
            return self.generations.firstGeneration()
        else:
            return self.generations.nextGeneration()

    def networkScore(self, network, score):
        self.generations.addGenome(Genome(score, network))

class Neuron:
    def __init__(self):
        self.value = 0.0
        self.weights = []

    def populate(self, nb):
        self.weights = [random.uniform(-1, 1) for _ in range(nb)]

class Layer:
    def __init__(self):
        self.neurons = []

    def populate(self, nbNeurons, nbInputs):
        self.neurons = [Neuron() for _ in range(nbNeurons)]
        for neuron in self.neurons:
            neuron.populate(nbInputs)

class Network:
    def __init__(self, options):
        self.options = options
        self.layers = []

    def perceptronGeneration(self, inputSize, hiddenLayers, outputSize):
        layer = Layer()
        layer.populate(inputSize, 0)
        self.layers.append(layer)

        previousNeurons = inputSize
        for hiddenSize in hiddenLayers:
            layer = Layer()
            layer.populate(hiddenSize, previousNeurons)
            self.layers.append(layer)
            previousNeurons = hiddenSize

        layer = Layer()
        layer.populate(outputSize, previousNeurons)
        self.layers.append(layer)

    def compute(self, inputs):
        for i, inputValue in enumerate(inputs):
            self.layers[0].neurons[i].value = inputValue

        for i in range(1, len(self.layers)):
            for neuron in self.layers[i].neurons:
                sum_value = sum(prev_neuron.value * neuron.weights[j] for j, prev_neuron in enumerate(self.layers[i - 1].neurons))
                neuron.value = self.options['activation'](sum_value)

        return [neuron.value for neuron in self.layers[-1].neurons]

class Genome:
    def __init__(self, score, network):
        self.score = score
        self.network = network

class Generation:
    def __init__(self, neuroevolution):
        self.neuroevolution = neuroevolution
        self.genomes = []

    def addGenome(self, genome):
        self.genomes.append(genome)
        self.genomes.sort(key=lambda x: x.score, reverse=self.neuroevolution.options['scoreSort'] == -1)

    def breed(self, g1, g2, nbChilds):
        children = []
        for _ in range(nbChilds):
            child = copy.deepcopy(g1)
            for i in range(len(g2.network.weights)):
                if random.random() > 0.5:
                    child.network.weights[i] = g2.network.weights[i]
                if random.random() < self.neuroevolution.options['mutationRate']:
                    child.network.weights[i] += random.uniform(-1, 1) * self.neuroevolution.options['mutationRange']
            children.append(child)
        return children

    def generateNextGeneration(self):
        next_gen = []
        elitism_count = int(self.neuroevolution.options['elitism'] * self.neuroevolution.options['population'])
        next_gen.extend(copy.deepcopy(genome.network) for genome in self.genomes[:elitism_count])

        random_behaviour_count = int(self.neuroevolution.options['randomBehaviour'] * self.neuroevolution.options['population'])
        for _ in range(random_behaviour_count):
            nn = Network(self.neuroevolution.options)
            nn.perceptronGeneration(self.neuroevolution.options['network'][0], self.neuroevolution.options['network'][1], self.neuroevolution.options['network'][2])
            next_gen.append(nn)

        max_breed = len(self.genomes)
        while len(next_gen) < self.neuroevolution.options['population']:
            for i in range(max_breed):
                children = self.breed(self.genomes[i], self.genomes[(i+1) % max_breed], self.neuroevolution.options['nbChild'])
                next_gen.extend(children[:self.neuroevolution.options['population'] - len(next_gen)])
                if len(next_gen) >= self.neuroevolution.options['population']:
                    break
        return next_gen

class Generations:
    def __init__(self, neuroevolution):
        self.neuroevolution = neuroevolution
        self.generations = []

    def firstGeneration(self):
        gen = Generation(self.neuroevolution)
        networks = []
        for _ in range(self.neuroevolution.options['population']):
            network = Network(self.neuroevolution.options)
            network.perceptronGeneration(self.neuroevolution.options['network'][0], self.neuroevolution.options['network'][1], self.neuroevolution.options['network'][2])
            networks.append(network)
        self.generations.append(gen)
        return networks

    def nextGeneration(self):
        if len(self.generations) == 0:
            return self.firstGeneration()
        gen = self.generations[-1]
        new_gen = gen.generateNextGeneration()
        self.generations.append(Generation(self.neuroevolution))
        return new_gen

    def addGenome(self, genome):
        self.generations[-1].addGenome(genome)

# Initialize the Neuroevolution and Game
neuroevo = Neuroevolution()
game = Game()

# Main loop for evolving generations
while True:
    networks = neuroevo.nextGeneration()
    game.start(networks)

    # Run the game with the current generation of networks
    while any(bird.alive for bird in game.birds):
        screen.fill((0, 0, 0))
        game.update(networks)
        game.display()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                # Adjust game speed
                if event.key == pygame.K_1:
                    FPS = 60  # x1 speed
                elif event.key == pygame.K_2:
                    FPS = 120  # x2 speed
                elif event.key == pygame.K_3:
                    FPS = 180  # x3 speed
                elif event.key == pygame.K_4:
                    FPS = 300  # x5 speed
                elif event.key == pygame.K_5:
                    FPS = 0  # Max speed
                elif event.key == pygame.K_g:  # Open GitHub link
                    webbrowser.open("https://github.com/jackngammm/Flappy_Bird")

        if FPS > 0:
            pygame.time.Clock().tick(FPS)
        else:
            # For max speed, run without delay
            pygame.event.pump()