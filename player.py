import pygame


# Object class
class Player(pygame.sprite.Sprite):
	def __init__(self, color, height, width):
		super().__init__()

		self.image = pygame.Surface([width, height])
		self.image.fill((167, 255, 100))
		self.image.set_colorkey((255, 100, 98))

		pygame.draw.rect(self.image,(255, 100, 98),pygame.Rect(0, 0, width, height))

		self.rect = self.image.get_rect()
