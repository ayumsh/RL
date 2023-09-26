import numpy as np
import matplotlib.pyplot as plt

    ###################################################################################################################################################################

class Bandit:                                                                                                       # Creating Bandit Object
    def __init__(self, num_arms, num_steps, num_runs, parameters, initial_value, method, type):
        self.num_arms = num_arms                                                                                    # num arms 10
        self.num_steps = num_steps                                                                                  # num steps 1000
        self.num_runs = num_runs                                                                                    # num runs 
        self.initial_value = initial_value                                                                          # used for Optemestic Initial value
        self.parameters = parameters                                                                                # Epsilon,alpha,Q initial, C array
        self.method=method                                                                                          # which method out of the 3 string
        self.type=type                                                                                              # used for plotting purpose of optemestic initial
        self.Q = np.zeros(num_arms)                                                                                 # Q
        self.N = np.zeros(self.num_arms)                                                                            # N

    ###################################################################################################################################################################

    def actionselecteg(self, epsilon):                                                                              #Action selection epsilon greedy
        if np.random.rand() < epsilon:                       
            action = np.random.choice(self.num_arms)    
        else:                                                          
            action = np.argmax(self.Q) 
        return action                                        
    
    def updateeg(self, action, reward):
        self.Q[action] += (reward - self.Q[action]) / self.N[action]                                                #Q-update epsilon-greedy

    ###################################################################################################################################################################

    def actionselectoi(self):                                                                                       #Action selection Optimistic initial value
        action = np.argmax(self.Q) 
        return action                                        
    
    def updateoi(self, action, reward, para):
        if self.type==0: self.Q[action] += para * (reward - self.Q[action])                                         #Q-update Optimistic initial value
        else : self.Q[action] += self.initial_value * (reward - self.Q[action])
    ###################################################################################################################################################################

    def actionselectucb(self, c, step):                                                                             #Action selection Upper-Confidence-Bound
        ucb_values = self.Q + c * np.sqrt(np.log(step + 1) / (self.N + 0.001))  
        action = np.argmax(ucb_values) 
        return action                                        
    
    def updateucb(self, action, reward):
        self.Q[action] += (reward - self.Q[action]) / self.N[action]                                                #Q-update Upper-Confidence-Bound

    ###################################################################################################################################################################

    def run(self):                                                                                                  # Method for running all three methods
        average_rewards = []
        optimal_arm_counts = []
        for param in self.parameters:                                                                               # different parameters epsilon,alpha,c,Q init
            rewards_per_step = []
            optimal_counts = np.zeros(self.num_steps)

            for _ in range(self.num_runs):                                                                          # over 2000 runs
                true_action_values = np.random.normal(0, 1, self.num_arms)
                if self.type==0 :                                                                                  
                    if self.method=='Optimistic-initial-value': self.Q=np.full(self.num_arms, self.initial_value)   #setting initial value
                    else :self.Q = np.zeros(self.num_arms)                                                          # of Q  for different
                elif self.type==1 :                                                                                 # methods
                    if self.method=='Optimistic-initial-value': self.Q=np.full(self.num_arms, param)
                    else :self.Q = np.zeros(self.num_arms)
                self.N = np.zeros(self.num_arms)
                rewards = []

                for step in range(self.num_steps):                                                                  # 1000 time steps

                    if(self.method=='epsilon-greedy') : action = self.actionselecteg(param)                         #
                    elif(self.method=='Optimistic-initial-value') : action = self.actionselectoi()                  # Action Selection
                    else : action = self.actionselectucb(param,step)                                                # 

                    reward = np.random.normal(true_action_values[action], 1)                                        # picking reward from normal distribution
                    self.N[action] += 1                                                                             # updating N

                    if(self.method=='epsilon-greedy') : self.updateeg(action, reward)                               #           
                    elif(self.method=='Optimistic-initial-value') : self.updateoi(action, reward, param)            # Q update 
                    elif(self.method=='Upper-Confidence-Bound') : self.updateucb(action, reward)                    #

                    rewards.append(reward)

                    is_optimal = (action == np.argmax(true_action_values))                                          # checking if action is optimal
                    if is_optimal:                                                                                  #
                        optimal_counts[step] += 1                                                                   #

                rewards_per_step.append(rewards)

            average_rewards.append(np.mean(rewards_per_step, axis=0))
            average_optimal_counts = optimal_counts / self.num_runs
            optimal_arm_counts.append(average_optimal_counts)

        return average_rewards, optimal_arm_counts
    
    ###################################################################################################################################################################

    def plotgraph(self, average_rewards, optimal_arm_counts):                                                       # Ploting for different HyperParameters
        plt.figure(figsize=(12, 6))

        if(self.method=='epsilon-greedy') : hparam='epsilon'                                                        # Naming HyperParameter for lable
        elif(self.method=='Optimistic-initial-value') :                                                             #
            if self.type==0 : hparam = 'alpha'                                                                      #
            else : hparam='Q'                                                                                       #
        elif(self.method=='Upper-Confidence-Bound') : hparam = 'c'      
                                                    
        plt.subplot(1, 2, 1)
        for i, param in enumerate(self.parameters):
            plt.plot(average_rewards[i], label=f'{hparam}={param}')                                                 # plotting all lines

        plt.xlabel('Steps')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.title('Average Rewards')
        
        plt.subplot(1, 2, 2)
        for i, param in enumerate(self.parameters):
            plt.plot(optimal_arm_counts[i], label=f'{hparam}={param}')

        plt.xlabel('Steps')
        plt.ylabel('Optimal Arm Selection Probability')
        plt.legend()
        plt.title('Optimal Arm Selection Probability')
        
        plt.suptitle(self.method, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    ###################################################################################################################################################################

    def compare(self, average_rewards, optimal_arm_counts, methods):                                                # Ploting for Comparing Different Methods
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        for i, method in enumerate(methods):
            if method=='epsilon-greedy': str= ', epsilon = 0.1'                                                     #
            elif method=='Optimistic-initial-value' : str=', Q = 5.0'                                               # Setting Lables
            else : str= ', c=2.0'                                                                                   #
            plt.plot(average_rewards[i][0], label=method+str)

        plt.xlabel('Steps')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.title('Average Rewards')

        plt.subplot(1, 2, 2)
        for i, method in enumerate(methods):
            if method=='epsilon-greedy': str= ', epsilon = 0.1'                                                     #                          
            elif method=='Optimistic-initial-value' : str=', Q = 5.0'                                               #   Setting Lables
            else : str= ', c=2.0'                                                                                   #
            plt.plot(optimal_arm_counts[i][0], label=method+str)

        plt.xlabel('Steps')
        plt.ylabel('Optimal Arm Selection Probability')
        plt.legend()
        plt.title('Optimal Arm Selection Probability')

        plt.suptitle("Comparison of different methods", fontsize=16)
        plt.tight_layout()
        plt.show()

    ###################################################################################################################################################################

class MRP:                                                                                                          #Creating Object
    def __init__(self, num_states, alphas ,actual_reward, gamma, num_runs, num_Episodes, actions):
        self.num_states=num_states                                                                                  # num states 7
        self.alphas=alphas                                                                                          # different values of alpha
        self.actual_reward=actual_reward                                                                            # actual reward array
        self.gamma=gamma                                                                                            # gamma =1 
        self.num_runs=num_runs                                                                                      # num runs for removing noise
        self.num_Episodes=num_Episodes                                                                              # num episodes 100
        self.actions=actions                                                                                        # action array -1,1
    
    ###################################################################################################################################################################
    
    def random_walk(self):                                                                                          # running MRP
        rmse_values = np.zeros((len(self.alphas), self.num_Episodes))
        Vret=np.zeros((5,5))
        for alpha_index, alpha in enumerate(self.alphas):
            for run in range(self.num_runs):                                                                        # For smoother curve less noise
                V = np.full(self.num_states, 0.5)
                V[0] = 0
                V[self.num_states - 1] = 0
                N = np.zeros((self.num_states, 2))
                for episode in range(self.num_Episodes):
                    state = 3
                    x1=0
                    while True:                                                                                     # Until terminal state
                        x1+=1
                        action = np.random.choice(self.actions)
                        next_state = state + action
                        if next_state == 0: reward = 0                                                              # action selection left, right
                        elif next_state == 6: reward = 1                                                            #
                        else: reward = 0
                        if action == -1 : action = 0
                        N[next_state][action] +=1
                        if(alpha_index < 3): V[state] = V[state] + alpha * (reward + self.gamma * V[next_state] - V[state])             # update constant alpha
                        else : V[state] = V[state] + (1/(N[next_state][action])) * (reward + self.gamma * V[next_state] - V[state])     # alpha 1/n  
                        state = next_state
                        if state in [0, 6]:                                                                         # terminal state
                            break
                    if episode == 1 and alpha_index == 0: Vret[2]+=V[1:6]/self.num_runs                             # used for plotting 
                    if episode == 30 and alpha_index == 0: Vret[3]+=V[1:6]/self.num_runs                            # 
                    if episode == 90 and alpha_index == 0: Vret[4]+=V[1:6]/self.num_runs                            #
                    rmse = np.sqrt(np.mean((V[1:6] - self.actual_reward) ** 2))                                     # Calculating RMSE
                    rmse_values[alpha_index][episode] += rmse/self.num_runs
        return rmse_values, Vret
    
    ###################################################################################################################################################################
    
    def plotgraph(self, rmse_values):                                                                               # Plotting RMSE Vs Episode
        plt.figure(figsize=(10, 6))
        for alpha in self.alphas:
            alpha_index = self.alphas.index(alpha)
            rmse_values_alpha = rmse_values[alpha_index]
            if alpha!=-1 :plt.plot(range(self.num_Episodes), rmse_values_alpha, label=f'Alpha = {alpha}')           # constant alpha
            else : plt.plot(range(self.num_Episodes), rmse_values_alpha, label=f'Alpha = 1/n')                      # alpha =1/n
        plt.xlabel('episode')
        plt.ylabel('RMSE')
        plt.title('RMSE vs. episode for Different Alpha Values', fontsize=16)
        plt.legend()
        plt.show()

    ###################################################################################################################################################################

    def plot(self,Vret):                                                                                            # plotting Estimated value at
        x_labels = ['A', 'B', 'C', 'D', 'E']                                                                        # different episodes
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, y_values in enumerate(Vret):
            ax.plot(x_labels, y_values, label=f'Episode {i + 1}', marker='o')
        ax.set_xlabel('State')
        ax.set_ylabel('Estimated Value')
        ax.legend()
        plt.title("Values learned after episodes", fontsize=16)
        plt.show()

    ###################################################################################################################################################################
     
def q1():                                                                                                           # function to run question 2

    print("Q1 is Running")
    num_arms = 10                                                                                                   #                                                                                   
    num_steps = 1000                                                                                                #
    num_runs = 2000                                                                                                 # Setting Initial value
    epsilon = [0.1,0.01,0,0.4]                                                                                      # For Diffferent Parameters
    alpha = [0.1,0.4,0.05]                                                                                          #    
    Q = [1.0,0.25,5.0]                                                                                              #    
    c=[2.0,1.0,0.5]                                                                                                 #

    bandit1 = Bandit(num_arms, num_steps, num_runs, epsilon, 0, 'epsilon-greedy', 0)                                # Creating Objects for Each Method
    bandit2 = Bandit(num_arms, num_steps, num_runs, alpha, 5.0, 'Optimistic-initial-value', 0)                      # for alpha for Q initial
    bandit3 = Bandit(num_arms, num_steps, num_runs, c, 0, 'Upper-Confidence-Bound', 0)                              # Optimistic Initial value have 2 objects 
    bandit4 = Bandit(num_arms, num_steps, num_runs, Q, 0.1, 'Optimistic-initial-value', 1)                          # because it have 2 hyper parameters

    average_rewards1, optimal_arm_counts1 = bandit1.run()                                                           # Running Each Method  EG
    average_rewards2, optimal_arm_counts2 = bandit2.run()                                                           #                      OI
    average_rewards3, optimal_arm_counts3 = bandit3.run()                                                           #                      UCB
    average_rewards4, optimal_arm_counts4 = bandit4.run()                                                           #                      OI

    bandit1.plotgraph(average_rewards1, optimal_arm_counts1)                                                        # Plotting for different
    bandit2.plotgraph(average_rewards2, optimal_arm_counts2)                                                        # Hyper Parameters
    bandit4.plotgraph(average_rewards4, optimal_arm_counts4)                                                        # 
    bandit3.plotgraph(average_rewards3, optimal_arm_counts3)                                                        #

    average_rewardsc1=[]
    optimal_arm_countsc1=[]
    method1=['epsilon-greedy','Optimistic-initial-value']
    average_rewardsc1.append(average_rewards1)
    average_rewardsc1.append(average_rewards2)                                                                      # epsilon greedy vs optimistic initial value graph
    optimal_arm_countsc1.append(optimal_arm_counts1)
    optimal_arm_countsc1.append(optimal_arm_counts2)
    bandit2.compare(average_rewardsc1, optimal_arm_countsc1, method1)

    average_rewardsc2=[]
    optimal_arm_countsc2=[]
    method2=['epsilon-greedy','Upper-Confidence-Bound']
    average_rewardsc2.append(average_rewards1)                                                                      # epsilon greedy vs Upper-Confidence-Bound graph
    average_rewardsc2.append(average_rewards3)
    optimal_arm_countsc2.append(optimal_arm_counts1)
    optimal_arm_countsc2.append(optimal_arm_counts3)
    bandit3.compare(average_rewardsc2, optimal_arm_countsc2, method2)

    average_rewards=[] 
    optimal_arm_counts=[]
    method=['epsilon-greedy','Optimistic-initial-value','Upper-Confidence-Bound']
    average_rewards.append(average_rewards1)
    average_rewards.append(average_rewards2)
    average_rewards.append(average_rewards3)                                                                        # epsilon  vs Upper-Confidence graph vs Optimistic
    optimal_arm_counts.append(optimal_arm_counts1)
    optimal_arm_counts.append(optimal_arm_counts2)
    optimal_arm_counts.append(optimal_arm_counts3)
    bandit3.compare(average_rewards, optimal_arm_counts, method)

    ###################################################################################################################################################################

def q2():                                                                                                           # function to run question 2

    print("Q2 is Running")
    num_states = 7                                                                                                  #
    alphas = [0.1, 0.25, 0.02, -1]                                                                                  # alpha =-1 is used for alpha = 1/n
    actual_reward = [1/6, 2/6, 3/6, 4/6, 5/6]                                                                       #
    gamma = 1.0                                                                                                     # Setting Initial value
    num_runs = 200                                                                                                  # For Diffferent Parameters
    num_Episodes = 100                                                                                              #
    actions = [-1, 1]                                                                                               #
    actual=np.array(actual_reward)                                                                                  #

    walk= MRP(num_states, alphas ,actual_reward, gamma, num_runs, num_Episodes, actions)            
    rmse_values, Vret=walk.random_walk()                                                                            # running the method
    Vret[0]=actual
    Vret[1]=np.full(num_states-2, 0.5)
    walk.plotgraph(rmse_values)                                                                                     # rmse plot
    walk.plot(Vret)                                                                                                 # plot estimated value after some episodes

    ###################################################################################################################################################################

def main():
    q1()                                                                                                            # run for q1
    q2()                                                                                                            # run for q2
    
if __name__ == '__main__':
    main()
