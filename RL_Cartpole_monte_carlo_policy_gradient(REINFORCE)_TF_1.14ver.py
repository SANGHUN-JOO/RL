import gym
import numpy as np
import tensorflow as tf

class POLICY:
    def __init__(self, session, input_size, output_size):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.data_list = []
        self.dis = 0.99
        self.learning_rate = 0.0005

        self._build_network()
    
    def _build_network(self, hidden_size=13):
        self._X = tf.placeholder(tf.float32, [None, self.input_size], name="X")
        W1 = tf.get_variable("W1", shape=[input_size, hidden_size], initializer=tf.contrib.layers.xavier_initializer())
        L1 = tf.nn.relu(tf.matmul(self._X, W1))

        W2 = tf.get_variable("W2", shape=[hidden_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
        self._Pi = tf.nn.softmax(tf.matmul(L1, W2))
        
        self._Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32)
        self._G = tf.placeholder(tf.float32)

        self._loss = - (self._Y*tf.log(self._Pi))
        self._loss = self._loss * self._G
        self._loss = tf.reduce_mean(tf.reduce_sum(self._loss, axis=1))
        
        self._train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self._loss)

    def put_data(self, p_data):
        self.data_list.append(p_data)

    def reshape_state(self, p_state):
        return np.reshape(p_state, [1, self.input_size])

    def get_Pi(self, state):
        x = self.reshape_state(state)
        return self.session.run(self._Pi, feed_dict={self._X: x})

    def update_Pi(self):
        reward = 0
        #print(self.data_list)
        for cur_reward, state, action in self.data_list[::-1]:
            x = self.reshape_state(state)
            y = np.zeros(self.output_size)
            y[action] = 1
            
            y = np.reshape(y, [1, self.output_size])
            
            
            reward = cur_reward + (self.dis * reward)
            reward = np.reshape(reward, [1,1])

            self.session.run(self._train, feed_dict={self._X: x, self._Y: y, self._G: reward})

        self.data_list = []



env = gym.make('CartPole-v0')
env._max_episode_steps = 500

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

def single_play(p_POLICY, p_env=env):
    state = p_env.reset()
    reward_sum = 0

    while True:
        p_env.render()
        action = np.argmax(p_POLICY.get_Pi(state))
        new_state, reward, done, _ = p_env.step(action)
        reward_sum += reward
        if done or reward_sum>501:
            print("Total score: {}".format(reward_sum))
            break
        state = new_state

def main():
    num_episodes = 2000

    with tf.Session() as sess:
        pi = POLICY(sess, input_size, output_size)
        sess.run(tf.global_variables_initializer())

        for i in range(num_episodes):
            state = env.reset()
            done = False
            step_count = 0

            while not done:
                step_count += 1

                action_prob = pi.get_Pi(state)
                action = np.random.choice(np.arange(output_size), p=action_prob[0])
                
                new_state, reward, done, _ = env.step(action)
                if done:
                    reward = -100
                    pi.put_data([reward, state, action])
                    break
                
                pi.put_data([reward, state, action])
                
                state = new_state
                if step_count > 501:
                    print("Good!")
                    break
            
            pi.update_Pi()
            if i%10 == 1:
                print("Episode {0} : {1}".format(i, step_count))
        
        single_play(pi)

if __name__ == "__main__":
    main()
