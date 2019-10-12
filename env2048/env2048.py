import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

import gym


UP = u'\ue013'
DOWN = u'\ue015'
LEFT = u'\ue012'
RIGHT = u'\ue014'


class Env2048(gym.Env):
  class WrongInputError(Exception):
    pass
  class TimeoverError(Exception):
    pass

  def __init__(self):
    self.driver = webdriver.Chrome('./chromedriver')
    self.driver.get("https://1024game.org")
    self.SLEEP_PERIOD = 0.4

  def step(self, action):
    target_key = None
    if action == 0: target_key = UP
    elif action == 1: target_key = DOWN
    elif action == 2: target_key = LEFT
    elif action == 3: target_key = RIGHT
    else:
      raise self.WrongInputError()

    self.driver.find_element_by_tag_name('body').send_keys(target_key)
    time.sleep(self.SLEEP_PERIOD)
    
    tiles = [
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0]
    ]
    max_value = 0

    for e in self.driver.find_elements_by_class_name('tile'):
      (x, y) = map(lambda p: int(p) - 1,
        re.findall(r"\d-\d", e.get_attribute('className'))[0].split('-'))
      tiles[y][x] = int(e.text)
      if int(e.text) > max_value:
        max_value = int(e.text)
    done = False

    game_message = self.driver.find_element_by_class_name('game-message')
    if game_message.get_attribute('className') == 'game-message game-over':
      done = True
  
    return (tiles, max_value, done, {})

  def reset(self):
    restart_button = self.driver.find_element_by_class_name('restart-button')
    restart_button.click()
    time.sleep(self.SLEEP_PERIOD)
    tiles = [
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0],
      [0,0,0,0]
    ]

    for e in self.driver.find_elements_by_class_name('tile'):
      (x, y) = map(lambda p: int(p) - 1,
        re.findall(r"\d-\d", e.get_attribute('className'))[0].split('-'))
      tiles[y][x] = int(e.text)
    return tiles

  def render(self, mode='human'):
    pass

  def close(self):
    self.driver.quit()

  def seed(self, seed=None):
    return None
