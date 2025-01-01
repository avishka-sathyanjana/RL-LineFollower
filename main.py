#!/usr/bin/env pybricks-micropython

from pybricks.ev3devices import ColorSensor, InfraredSensor, Motor
from pybricks.hubs import EV3Brick
from pybricks.parameters import Port
from pybricks.robotics import DriveBase
from pybricks.tools import wait
import random
import pickle

Q_TABLE_FILE = 'q_table.pkl'
#Q(s,a)=Q(s,a) + α×(R+γ × maxQ(s',a′)−Q(s,a))

WHITE_THRESHOLD = 25
BLACK_THRESHOLD = 8

# alpha - how quickly the robot updates the Q-values.
LEARNING_RATE = 0.1

# gamma - how much the robot values future rewards compared to immediate rewards.
DISCOUNT_FACTOR = 0.9

# Euler's number - exploration decay 
E=2.7321

# how fast the robot reduces its exploration.
TEMP=1000

TRAINING = False

class MODE:
    INNER_LINE = 'INNER_LINE'
    OUTER_LINE = 'OUTER_LINE'

class DIRECTION:
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"

class LIGHT_STATE:
    WHITE = "WHITE"
    MIDDLE = "MIDDLE"
    BLACK = "BLACK"

class STRATEGY:
    RANDOM = 'Random'
    GREEDY = 'Greedy'

#to identify the mode
STATE_TRANSITIONS = {
    'INNER' : [
        (LIGHT_STATE.MIDDLE, 'turn_right', LIGHT_STATE.WHITE),
        (LIGHT_STATE.BLACK, 'turn_right', LIGHT_STATE.MIDDLE),
        (LIGHT_STATE.WHITE, 'turn_left', LIGHT_STATE.MIDDLE),
        (LIGHT_STATE.MIDDLE, 'turn_left', LIGHT_STATE.BLACK)
    ],
    'OUTER' : [
        (LIGHT_STATE.MIDDLE, 'turn_right', LIGHT_STATE.BLACK),
        (LIGHT_STATE.WHITE, 'turn_right', LIGHT_STATE.MIDDLE),
        (LIGHT_STATE.BLACK, 'turn_left', LIGHT_STATE.MIDDLE),
        (LIGHT_STATE.MIDDLE, 'turn_left', LIGHT_STATE.WHITE)
    ]
}

ev3 = EV3Brick()
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
light_sensor = ColorSensor(Port.S3)
ir_sensor = InfraredSensor(Port.S4)
robot = DriveBase(left_motor, right_motor, wheel_diameter=40, axle_track=50)

def save_qtable(q_dict):
    temp = {}
    for key, value in q_dict.items():
        temp[(key[0],key[1],key[2].__name__)] = value
    with open(Q_TABLE_FILE, 'wb') as file:
        pickle.dump(temp, file)
       
def load_qtable():
    with open(Q_TABLE_FILE, 'rb') as file:
        temp = pickle.load(file)
    table = {}
    for key,value in temp.items():
        table[(key[0],key[1],globals()[key[2]])] = value
    return table

def forward(robot):
    robot.drive(100,0)
    wait(250) 

def backward(robot):
    robot.drive(-100, 0)
    wait(250)

def turn_left(robot, previous_light_state):
    while previous_light_state == get_light_state():
        robot.drive(10,-110)
        wait(100) 

def turn_right(robot,previous_light_state):
    while previous_light_state == get_light_state():
        robot.drive(10,110)
        wait(100) 

def obstacle_aviodance():
    ev3.speaker.say("An obstacle detected.")
    ev3.speaker.say("Turning back.")
    robot.drive_time(-80, 0, 1000)
    robot.drive_time(0, 100, 8000)
  
actions = [forward, backward, turn_left, turn_right]

Q_table = {}
for mode in [MODE.INNER_LINE, MODE.OUTER_LINE]:
    for act in actions:
        for light in [LIGHT_STATE.BLACK, LIGHT_STATE.BLACK]:
            Q_table[(mode,light, act)] = 0

def get_light_state():
    light = light_sensor.reflection()
    if(light >= WHITE_THRESHOLD):
        return LIGHT_STATE.WHITE
    if(light <= BLACK_THRESHOLD):
        return LIGHT_STATE.BLACK
    return LIGHT_STATE.MIDDLE



def get_best_action(Q_table, mode, light_state):
    # any q value is better than -inf
    max_q = -float("inf")
    best_action = None
    
    #all posible actions that robot can take
    for act in actions:
        # Get the Q-value for this combination from Q-table
        q = Q_table[(mode, light_state, act)]
        # largets q value is the best action
        if q > max_q:
            best_action = act
            max_q = q
    return best_action, max_q

def get_mode(light_state, new_light_state, action, mode):
    if((light_state, action.__name__, new_light_state) in STATE_TRANSITIONS['INNER']):
        return MODE.INNER_LINE
    elif ((light_state, action.__name__, new_light_state) in STATE_TRANSITIONS['OUTER']):
        return MODE.OUTER_LINE
    else:
        return mode

def learn():
    light_state = get_light_state()
    previous_light_state = light_state 
    mode = MODE.INNER_LINE
    iterations = 0
    action = None
    direction = DIRECTION.FORWARD
    exploration_strategy = STRATEGY.RANDOM
    
    while True:
        # Stop training when exploration probability gets very low
        if (E**(iterations/-TEMP) < 0.01):
            TRAINING = False
            save_qtable(Q_table)
            break
        
        # Exploration: Try random actions sometimes
        # Probability of exploration decreases over time (iterations)
        if random.uniform(0, 1) <  E**(iterations/-TEMP):
            # random action is getting a low probability when thee itearation is high
            action = random.choice(actions)
            exploration_strategy = STRATEGY.RANDOM
        else:
            action = get_best_action(Q_table, mode, light_state)[0]
            exploration_strategy = STRATEGY.GREEDY
            
        print(exploration_strategy, action.__name__ , mode)

        action(robot, light_state)
        if action == forward:
            direction = DIRECTION.FORWARD
        else:
            direction = DIRECTION.BACKWARD

        new_light_state = get_light_state()
        new_mode = get_mode(light_state, new_light_state, action, mode)

        max_q_next = get_best_action(Q_table, new_mode, new_light_state)[1]
        reward_next = get_reward(new_light_state, direction)
        
        Q_table[(mode, light_state, action)] += LEARNING_RATE * (
            reward_next + DISCOUNT_FACTOR * max_q_next - Q_table[( mode, light_state, action)])

        ev3.screen.clear()
        ev3.screen.draw_text(20,20, iterations)
        ev3.screen.draw_text(20,40, E**(iterations/-TEMP))
        ev3.screen.draw_text(20,60, exploration_strategy)
        ev3.screen.draw_text(20,80, action.__name__)
        ev3.screen.draw_text(20,100, reward_next)
        
        light_state = new_light_state
        mode = new_mode
        iterations += 1

        save_qtable(Q_table)

def line_following(Q_table, mode, light_state):
    action = get_best_action(Q_table, mode,light_state)[0]
    print("line following",mode, action.__name__, light_state)

    ev3.screen.clear()
    ev3.screen.draw_text(20,20,"Following the line..!")

    action(robot, light_state)
    
    new_light_state = get_light_state()
    mode = get_mode(light_state, new_light_state, action, mode)   

    return mode, new_light_state

def run():
    light_state = get_light_state()
    mode = MODE.INNER_LINE

    Q_table = load_qtable()
    print(Q_table)

    action = turn_right
    action(robot, light_state)
    
    ls = get_light_state()
    mode = get_mode(light_state, ls, action, mode)    
    light_state = ls

    while True:
        if (ir_sensor.distance() < 20):
            robot.stop()
            print("obstacle")
            obstacle_aviodance()
            
        else:
            mode, light_state = line_following(Q_table, mode, light_state)

if(TRAINING):
    learn()

run()#!/usr/bin/env pybricks-micropython

from pybricks.ev3devices import ColorSensor, InfraredSensor, Motor
from pybricks.hubs import EV3Brick
from pybricks.parameters import Port
from pybricks.robotics import DriveBase
from pybricks.tools import wait
import random
import pickle

Q_TABLE_FILE = 'q_table.pkl'
#Q(s,a)=Q(s,a) + α×(R+γ × maxQ(s',a′)−Q(s,a))

WHITE_THRESHOLD = 25
BLACK_THRESHOLD = 8

# alpha - how quickly the robot updates the Q-values.
LEARNING_RATE = 0.1

# gamma - how much the robot values future rewards compared to immediate rewards.
DISCOUNT_FACTOR = 0.9

# Euler's number - exploration decay 
E=2.7321

# how fast the robot reduces its exploration.
TEMP=1000

TRAINING = False

class MODE:
    INNER_LINE = 'INNER_LINE'
    OUTER_LINE = 'OUTER_LINE'

class DIRECTION:
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"

class LIGHT_STATE:
    WHITE = "WHITE"
    MIDDLE = "MIDDLE"
    BLACK = "BLACK"

class STRATEGY:
    RANDOM = 'Random'
    GREEDY = 'Greedy'

#to identify the mode
STATE_TRANSITIONS = {
    'INNER' : [
        (LIGHT_STATE.MIDDLE, 'turn_right', LIGHT_STATE.WHITE),
        (LIGHT_STATE.BLACK, 'turn_right', LIGHT_STATE.MIDDLE),
        (LIGHT_STATE.WHITE, 'turn_left', LIGHT_STATE.MIDDLE),
        (LIGHT_STATE.MIDDLE, 'turn_left', LIGHT_STATE.BLACK)
    ],
    'OUTER' : [
        (LIGHT_STATE.MIDDLE, 'turn_right', LIGHT_STATE.BLACK),
        (LIGHT_STATE.WHITE, 'turn_right', LIGHT_STATE.MIDDLE),
        (LIGHT_STATE.BLACK, 'turn_left', LIGHT_STATE.MIDDLE),
        (LIGHT_STATE.MIDDLE, 'turn_left', LIGHT_STATE.WHITE)
    ]
}

ev3 = EV3Brick()
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
light_sensor = ColorSensor(Port.S3)
ir_sensor = InfraredSensor(Port.S4)
robot = DriveBase(left_motor, right_motor, wheel_diameter=40, axle_track=50)

def save_qtable(q_dict):
    temp = {}
    for key, value in q_dict.items():
        temp[(key[0],key[1],key[2].__name__)] = value
    with open(Q_TABLE_FILE, 'wb') as file:
        pickle.dump(temp, file)
       
def load_qtable():
    with open(Q_TABLE_FILE, 'rb') as file:
        temp = pickle.load(file)
    table = {}
    for key,value in temp.items():
        table[(key[0],key[1],globals()[key[2]])] = value
    return table

def forward(robot):
    robot.drive(100,0)
    wait(250) 

def backward(robot):
    robot.drive(-100, 0)
    wait(250)

def turn_left(robot, previous_light_state):
    while previous_light_state == get_light_state():
        robot.drive(10,-110)
        wait(100) 

def turn_right(robot,previous_light_state):
    while previous_light_state == get_light_state():
        robot.drive(10,110)
        wait(100) 

def obstacle_aviodance():
    ev3.speaker.say("An obstacle detected.")
    ev3.speaker.say("Turning back.")
    robot.drive_time(-80, 0, 1000)
    robot.drive_time(0, 100, 8000)
  
actions = [forward, backward, turn_left, turn_right]

Q_table = {}
for mode in [MODE.INNER_LINE, MODE.OUTER_LINE]:
    for act in actions:
        for light in [LIGHT_STATE.BLACK, LIGHT_STATE.BLACK]:
            Q_table[(mode,light, act)] = 0

def get_light_state():
    light = light_sensor.reflection()
    if(light >= WHITE_THRESHOLD):
        return LIGHT_STATE.WHITE
    if(light <= BLACK_THRESHOLD):
        return LIGHT_STATE.BLACK
    return LIGHT_STATE.MIDDLE

def get_reward(new_light_state, direction):
    if new_light_state in [LIGHT_STATE.BLACK, LIGHT_STATE.BLACK]:
        return -10 
    elif direction == DIRECTION.FORWARD:
        return 15
    elif direction == DIRECTION.BACKWARD:
        return 5
    else:
        return 10

def get_best_action(Q_table, mode, light_state):
    # any q value is better than -inf
    max_q = -float("inf")
    best_action = None
    
    #all posible actions that robot can take
    for act in actions:
        # Get the Q-value for this combination from Q-table
        q = Q_table[(mode, light_state, act)]
        # largets q value is the best action
        if q > max_q:
            best_action = act
            max_q = q
    return best_action, max_q

def get_mode(light_state, new_light_state, action, mode):
    if((light_state, action.__name__, new_light_state) in STATE_TRANSITIONS['INNER']):
        return MODE.INNER_LINE
    elif ((light_state, action.__name__, new_light_state) in STATE_TRANSITIONS['OUTER']):
        return MODE.OUTER_LINE
    else:
        return mode

def learn():
    light_state = get_light_state()
    previous_light_state = light_state 
    mode = MODE.INNER_LINE
    iterations = 0
    action = None
    direction = DIRECTION.FORWARD
    exploration_strategy = STRATEGY.RANDOM
    
    while True:
        # Stop training when exploration probability gets very low
        if (E**(iterations/-TEMP) < 0.01):
            TRAINING = False
            save_qtable(Q_table)
            break
        
        # Exploration: Try random actions sometimes
        # Probability of exploration decreases over time (iterations)
        if random.uniform(0, 1) <  E**(iterations/-TEMP):
            # random action is getting a low probability when thee itearation is high
            action = random.choice(actions)
            exploration_strategy = STRATEGY.RANDOM
        else:
            action = get_best_action(Q_table, mode, light_state)[0]
            exploration_strategy = STRATEGY.GREEDY
            
        print(exploration_strategy, action.__name__ , mode)

        action(robot, light_state)
        if action == forward:
            direction = DIRECTION.FORWARD
        else:
            direction = DIRECTION.BACKWARD

        new_light_state = get_light_state()
        new_mode = get_mode(light_state, new_light_state, action, mode)

        max_q_next = get_best_action(Q_table, new_mode, new_light_state)[1]
        reward_next = get_reward(new_light_state, direction)
        
        Q_table[(mode, light_state, action)] += LEARNING_RATE * (
            reward_next + DISCOUNT_FACTOR * max_q_next - Q_table[( mode, light_state, action)])

        ev3.screen.clear()
        ev3.screen.draw_text(20,20, iterations)
        ev3.screen.draw_text(20,40, E**(iterations/-TEMP))
        ev3.screen.draw_text(20,60, exploration_strategy)
        ev3.screen.draw_text(20,80, action.__name__)
        ev3.screen.draw_text(20,100, reward_next)
        
        light_state = new_light_state
        mode = new_mode
        iterations += 1

        save_qtable(Q_table)

def line_following(Q_table, mode, light_state):
    action = get_best_action(Q_table, mode,light_state)[0]
    print("line following",mode, action.__name__, light_state)

    ev3.screen.clear()
    ev3.screen.draw_text(20,20,"Following the line..!")

    action(robot, light_state)
    
    new_light_state = get_light_state()
    mode = get_mode(light_state, new_light_state, action, mode)   

    return mode, new_light_state

def run():
    light_state = get_light_state()
    mode = MODE.INNER_LINE

    Q_table = load_qtable()
    print(Q_table)

    action = turn_right
    action(robot, light_state)
    
    ls = get_light_state()
    mode = get_mode(light_state, ls, action, mode)    
    light_state = ls

    while True:
        if (ir_sensor.distance() < 20):
            robot.stop()
            print("obstacle")
            obstacle_aviodance()
            
        else:
            mode, light_state = line_following(Q_table, mode, light_state)

if(TRAINING):
    learn()

run()