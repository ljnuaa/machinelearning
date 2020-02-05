#######################################################################
# Copyright (C)                                                       #
# 2016 - 2019 Pinard Liu(liujianping-ok@163.com)                      #
# https://www.cnblogs.com/pinard                                      #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
##https://www.cnblogs.com/pinard/p/9714655.html ##
## 强化学习（八）价值函数的近似表示与Deep Q-Learning ##

import gym  #不需要引入
import tensorflow as tf #需要引入
import numpy as np #需要引入
import random #需要引入
from collections import deque #需要引入

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q ，Q函数中的折扣因子
INITIAL_EPSILON = 0.5 # starting value of epsilon，explore概率
FINAL_EPSILON = 0.01 # final value of epsilon，explore 概率逐渐减小，由explore转为exploit
REPLAY_SIZE = 10000 # experience replay buffer size，memorysize？待定
BATCH_SIZE = 32 # size of minibatch，为了加速计算选取将replay分割成多个batch

class DQN():#定义agent
  # DQN Agent
  def __init__(self, env): #agent叫self，他需要在env的环境下做出决策
    # init experience replay，以下定义经验回放
    self.replay_buffer = deque() #将所有经验放到一个双向队列中
    # init some parameters
    self.time_step = 0 # 初始化agent的step号
    self.epsilon = INITIAL_EPSILON #初始化agent的explore概率
    self.state_dim = env.observation_space.shape[0] #定义agent面对的state有几个，这里我们有三组states, 每一组都有对应的action
    self.action_dim = env.action_space.n #定义agent 面对的action有几个，这里我们有4个，2个，2个action.

    self.create_Q_network() #创建Q_network
    self.create_training_method()  #创建traning

    # Init session
    self.session = tf.InteractiveSession() #创建一个会话
    self.session.run(tf.global_variables_initializer())  #初始化广域变量，不显示结果

  def create_Q_network(self): #定义Qnetwork搭建一个神经网络
    # network weights
    W1 = self.weight_variable([self.state_dim,20])  #4个state元素对应20个节点
    b1 = self.bias_variable([20]) #每个节点有一个常数项
    W2 = self.weight_variable([20,self.action_dim]) #20个节点对应两个action的w
    b2 = self.bias_variable([self.action_dim])#两个action对应的常数项
    # input layer
    self.state_input = tf.placeholder("float",[None,self.state_dim]) #占位input，由于有bath，所以行数但不确定noun，列数是4，表示state里的4个因素
    # hidden layers
    h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1) #相乘，相加后，大于0的不变，小于0的变为0，共20列
    # Q Value layer
    self.Q_value = tf.matmul(h_layer,W2) + b2 # 隐含层向输出计算，共2列

  def create_training_method(self): 
    self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation， 占位，2列
    self.y_input = tf.placeholder("float",[None]) # 占位，一个数
    Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1) #这里理解计算两个action的总和
    self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action)) # 这里计算y 和Q之间的差值
    self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost) #权重更新比率是0.0001，选择一个优化器，这里选的是adamoptimizer，还可以选别的，或者自己写算法

  def perceive(self,state,action,reward,next_state,done):# 定义training中的perceive函数
    one_hot_action = np.zeros(self.action_dim) #返回两个0值数组 
    one_hot_action[action] = 1 #第一列为1，第二列为0
    self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
    #记忆库里面增加一个记忆deque([(array([ 0.01698751, -0.03674131,  0.01265532, -0.03304825]), 
    #array([1., 0.]), 0.1, array([ 0.01625269, -0.23204243,  0.01199435,  0.26360055]), False)])
    #episode:  0 Evaluation Average Reward: 1.0
    if len(self.replay_buffer) > REPLAY_SIZE: #如果增加了新记忆后，发现记忆库超过了reply buffer要求
      self.replay_buffer.popleft() #那么删除最先加进记忆库的记忆

    if len(self.replay_buffer) > BATCH_SIZE: #如果增加了新记忆后，发现记忆库超过了batch size要求，
      self.train_Q_network() #调用这个函数

  def train_Q_network(self):#定义这个函数
    self.time_step += 1  # step=step+1
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE) #从记忆池子中随机选规定size的batch
    state_batch = [data[0] for data in minibatch] # 从选出来的记忆条中分离出state
    action_batch = [data[1] for data in minibatch] # 从选出来的记忆条中分离出action
    reward_batch = [data[2] for data in minibatch]  # 从选出来的记忆中分离出reward
    next_state_batch = [data[3] for data in minibatch]  #。。。。。分离出next state

    # Step 2: calculate y
    y_batch = []  #定义y
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch}) #这里计算batch的Q值
    for i in range(0,BATCH_SIZE): #在batch中针对每一个sample
      done = minibatch[i][4] # 是minibatchlist中的第4个数组，true or false
      if done: #如果是true
        y_batch.append(reward_batch[i]) # y就等于当前的reward
      else : #如果不是true
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i])) # 本次加下一次Q值最大值

    self.optimizer.run(feed_dict={ 
      self.y_input:y_batch,
      self.action_input:action_batch,
      self.state_input:state_batch
      }) #将batch的所有数据输入优化过程

  def egreedy_action(self,state): #用了egreedy算法
    Q_value = self.Q_value.eval(feed_dict = { 
      self.state_input:[state] #
      })[0] #计算Q值
    if random.random() <= self.epsilon: #如果随机选择概率小于规定的epsilon，那么久随机选一个
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000 #逐步减小随机选择概率
        return random.randint(0,self.action_dim - 1) 
    else:
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000 #如果没有，那么返回Q值最大的一个索引号，这个索引号对应action
        return np.argmax(Q_value)

  def action(self,state): #又定义一遍action是返回最大Q值的action
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0]) 

  def weight_variable(self,shape): #返回服从正态分布的variable，两组，分别是4*20 和 
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape) #返回初始化的bias：20个和2个
    return tf.Variable(initial)
# ---------------------------------------------------------
# Hyper Parameters 主程序
ENV_NAME = 'CartPole-v0' #定义环境名称，这里我们定义ENV_NAME=CHOICE
EPISODE = 3000 # Episode limitation  # 一共是30000次试验
STEP = 300 # Step limitation in an episode #每次试验中有300步，我们没有300步，我们只有最多5步
TEST = 10 # The number of experiment test every 100 episode 3000次试验分成了10次小的试验类

def main(): #定义main函数
  # initialize OpenAI Gym env and dqn agent 初始化环境和agent
  
  env = gym.make(ENV_NAME) #这个地方我们需要自己设置环境：state 和 reward
  
  agent = DQN(env) # 初始化agent

  for episode in range(EPISODE): #对于每一个试验
    # initialize task
    state = env.reset()#重置实验环境，我们这里应该是加载进一个编号的env
    # Train
    for step in range(STEP): #对于一个试验里的一个step
      action = agent.egreedy_action(state) # e-greedy action for train，通过上面的行为确定action
      next_state,reward,done,_ = env.step(action) # 这里确定用了上面的action会导致下一个state,reward,和done
      # Define reward for agent
      reward = -1 if done else 0.1 #如果done==true，那么reward=-1，否则的话其他reward都是0，这里关于done，我们需要有一个判断
      agent.perceive(state,action,reward,next_state,done) # 调用agent.perceive函数，
      state = next_state# 新的state等于perceive里面的新的next state
      if done: #如果next state是done，那么直接break
        break
    # Test every 100 episodes
    if episode % 100 == 0: #如果实验次能够被100整除
      total_reward = 0 #那么累计reward0
      for i in range(TEST): #在每一个实验中
        state = env.reset() #重置环境
        for j in range(STEP): #对于每一个episode里面的step
          env.render() #调用演示
          action = agent.action(state) # direct action for test #这里不考虑优化训练时的action，而是直接测试action
          state,reward,done,_ = env.step(action) #确定下一个state
          total_reward += reward #计算reward
          if done: #如果done，就break
            break
      ave_reward = total_reward/TEST#计算总的reward
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward) #显示最后的reward

if __name__ == '__main__': #这里添加了入口，可以保证在在调用是不返回的上述结果
  main() #主程序结束
  
  
