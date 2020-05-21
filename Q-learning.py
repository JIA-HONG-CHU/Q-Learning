
import numpy as np
import pandas as pd
import random
import time



N_STATES = 20   # the length of the 1 dimensional world
ACTIONS = ['right','shoot','bombard']     # available actions
EPSILON = 0.9   # greedy police
GAMMA = 0.9    # discount factor
MAX_EPISODES = 10  # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move


#create q-table
def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,    # actions's name
    )
    # print(table)    # show table
    return table

#choose action
def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()): # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A, env_list):
    # This is how agent will interact with the environment
    #print(env_list)
    if A == 'right':    # move right
        if S == N_STATES - 2:   # terminate
            S_ = 'terminal'
            R = 5
        elif env_list[S+1]=='웃': #encounter the enemy or wall
            S_ = S 
            R = -1
        elif env_list[S+1]=='!':
            S_=S
            R=-1

        #elif (S+1)=='$': #get the money
            #S_=S+1
            #R=10
        else:  #nothing happen or encounter
            S_=S+1
            R=1
    elif A=='shoot':
        if env_list[S+1]=='웃': #encounter the enemy
            env_list[S+1]='-'
            S_=S+1
            R=2
        else:
            S_=S
            R=0 
    else:
        if env_list[S+1]=='!': #encounter the wall
            env_list[S+1]='-'
            S_=S+1
            R=3
        else:
            S_=S
            R=0
            
    #else:   # move left
        #R = 0
        #if S == 0:
            #S_ = S  # reach the left side
        #else:
            #S_ = S - 1
    return S_, R 


def update_env(S, episode, step_counter,env_list):
    # This is how environment be updated
    update_env_list=env_list.copy()
    
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        update_env_list[S] = '➤'
        interaction= ''.join(update_env_list)
        #print(interaction)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
        

def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    
    init_env_list=['-']*(N_STATES-7)+['웃']*(N_STATES-16)+['!']*(N_STATES-17)
    random.shuffle(init_env_list)
    init_env_list[0]='-'
    init_env_list[-1]='-'
    
    for episode in range(MAX_EPISODES):
        env_list=init_env_list.copy()
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter,env_list)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R  = get_env_feedback(S, A, env_list)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] =  q_target   # update
            S = S_  # move to next state

            update_env(S, episode, step_counter+1,env_list)
            step_counter += 1
            
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)






