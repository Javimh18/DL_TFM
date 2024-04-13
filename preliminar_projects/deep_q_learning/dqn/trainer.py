class Trainer:
    def __init__(self, env, agent, n_episodes, print_every=200) -> None:
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.curr_step = 0
        self.print_every = print_every
        # TODO: Add attributes for logging
        
    def train(self):
        step = 0
        ep_length = 0
        for e in range(self.n_episodes):
            # reset environment
            done, trunc = False, False
            state = self.env.reset()
            while (not done) and (not trunc):
                # 1. get action for state
                action = self.agent.perform_action(state)
                # 2. run action in environment
                next_state, reward, done, trunc, _ = self.env.step(action)
                # 3. collect experience in exp. replay buffer for Q-learning
                self.agent.store_transition(state, action, reward, next_state, done, trunc)
                # 4. Learn from collected experiences
                q, loss = self.agent.learn(step)
                # 5. Update the current state 
                state = next_state
                # 6. Update step value 
                step += 1
                
            ep_length = step - ep_length
            # TODO: It would be nice that, each N episodes, some information was printed out
            # or plotted in a log and a graph, for better understanding of the decisions
            if not (self.curr_step % self.print_every):
                print(f"INFO | Episode: {e} | Current Loss: {loss} | Mean Q_values: {q} | Episode Length: {ep_length}")
                
            self.curr_step += 1
        
        
        